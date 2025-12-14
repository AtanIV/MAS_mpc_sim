#!/usr/bin/env python3
"""
Visualize logged local subgraphs from MPC-GCBF testing.
Creates individual frame images and optionally an animation.
"""

# !/usr/bin/env python3
"""
Visualize logged local subgraphs from MPC-GCBF testing.
Creates individual frame images and optionally an animation.
Now includes status indicators from agent step logs.
"""

import pickle
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import Dict, List, Tuple, Optional
import glob
import json


class LocalGraphVisualizer:
    """Visualize local subgraphs with proper node type handling and status indicators."""

    def __init__(self,
                 figure_size: Tuple[int, int] = (12, 10),
                 dpi: int = 100,
                 axis_mode: str = 'auto',
                 axis_limits: Optional[Tuple[float, float, float, float]] = None,
                 padding: float = 0.5,
                 dt: float = 0.03,
                 mass: float = 0.1,
                 max_pred_steps: int = 16,
                 exaggeration_factor: float = 7.0):
        self.figure_size = figure_size
        self.dpi = dpi
        self.axis_mode = axis_mode
        self.axis_limits = axis_limits
        self.padding = padding

        # Dynamics parameters (match DoubleIntegrator defaults)
        self.dt = dt
        self.mass = mass
        self.max_pred_steps = max_pred_steps
        self.exaggeration_factor = exaggeration_factor

        # Color scheme matching GCBF conventions
        self.colors = {
            'ego_agent': '#FF4444',
            'other_agent': '#4444FF',
            'goal': '#44FF44',
            'lidar': '#FF0000',
            'padding': '#DDDDDD',
            'edge_agent': '#0066CC',
            'edge_goal': '#00CC66',
            'edge_lidar': '#333333'
        }

        # Status indicator colors
        self.status_colors = {
            'MPC': '#00CC00',  # Green - MPC succeeded
            'SAFE_QP': '#FF9900',  # Red - Safe QP fallback
            'POSITIVE_DERIVATIVE': '#FF0000'  # Orange - Positive directional derivative
        }

        self.markers = {
            'ego_agent': 'o',
            'other_agent': 's',
            'goal': '*',
            'lidar': 'o'
        }

        self.sizes = {
            'ego_agent': 200,
            'other_agent': 150,
            'goal': 300,
            'lidar': 20
        }

        # Cache for step logs
        self.step_logs_cache = {}

    # -------------------------------
    # Prediction helper functions
    # -------------------------------

    def _parse_control_sequence(self, control_field) -> Optional[np.ndarray]:
        """
        Parse the 'control_input' field from the CSV into a numpy array
        of shape (H, 2) or (1, 2).

        Returns None if parsing fails or the field is empty.
        """
        if control_field is None:
            return None

        # Sometimes this comes in as NaN
        if isinstance(control_field, float) and np.isnan(control_field):
            return None

        # Already an array-like?
        if isinstance(control_field, (list, tuple, np.ndarray)):
            arr = np.asarray(control_field, dtype=float)
        else:
            # Expect a JSON-encoded string from the logger
            s = str(control_field).strip()
            if not s:
                return None
            try:
                arr = np.asarray(json.loads(s), dtype=float)
            except Exception:
                # Fallback: try to pull out numbers crudely
                import re
                nums = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', s)
                if not nums:
                    return None
                arr = np.asarray(nums, dtype=float)

        if arr.ndim == 1:
            # (2,) -> single step 2D control
            if arr.size == 2:
                arr = arr.reshape(1, 2)
            else:
                # Try to interpret as flat list of pairs
                arr = arr.reshape(-1, 2)

        if arr.shape[-1] != 2:
            # Not a 2D control; give up
            return None

        return arr

    def _is_step_failure(self, row) -> bool:
        """
        Heuristic: treat step as 'failed' if:
          - controller_used is SAFE_QP / UNKNOWN / empty, or
          - mpc_status contains 'fail'/'error'
        """
        controller = str(row.get('controller_used', '')).upper()
        mpc_status = str(row.get('mpc_status', '')).lower()

        if controller in ('SAFE_QP', 'UNKNOWN', ''):
            return True
        if 'fail' in mpc_status or 'error' in mpc_status:
            return True
        return False

    def _rollout_positions(self,
                           initial_state: np.ndarray,
                           forces_seq: np.ndarray,
                           include_start: bool = False) -> Optional[np.ndarray]:
        """
        Roll out DoubleIntegrator positions (x, y) given initial state [x, y, vx, vy]
        and a sequence of forces (u_x, u_y).
        If include_start=True, the first entry is the current position.
        """
        if forces_seq is None:
            return None

        forces_seq = np.asarray(forces_seq, dtype=float)
        if forces_seq.ndim == 1:
            forces_seq = forces_seq.reshape(1, -1)

        # Limit horizon for plotting
        if self.max_pred_steps is not None and forces_seq.shape[0] > self.max_pred_steps:
            forces_seq = forces_seq[:self.max_pred_steps]

        x, y, vx, vy = map(float, initial_state[:4])
        dt = float(self.dt)
        m = float(self.mass)

        positions = []
        if include_start:
            positions.append((x, y))

        for f in forces_seq:
            ax, ay = f[0] / m, f[1] / m  # accel = force / mass

            x = x + vx * dt + 0.5 * ax * dt * dt
            y = y + vy * dt + 0.5 * ay * dt * dt
            vx = vx + ax * dt
            vy = vy + ay * dt

            positions.append((x, y))

        if not positions:
            return None
        return np.asarray(positions, dtype=float)


    def _get_neighbor_tail_sequence(self,
                                    df,
                                    prev_step_idx: int) -> Optional[np.ndarray]:
        """
        For a neighbor: use tail+zero of previous step's sequence, or zeros
        if that step failed.

        df: step log DataFrame for that neighbor
        prev_step_idx: index of the previous step in df
        """
        if df is None or prev_step_idx < 0 or prev_step_idx >= len(df):
            return None

        row = df.iloc[prev_step_idx]
        ctrl_arr = self._parse_control_sequence(row.get('control_input', None))
        if ctrl_arr is None:
            return None

        # On failure, return all zeros: zero inputs -> constant-velocity motion
        if self._is_step_failure(row):
            return np.zeros_like(ctrl_arr)

        # Success: build tail+zero
        H = ctrl_arr.shape[0]
        tail = np.zeros_like(ctrl_arr)
        if H > 1:
            tail[:-1] = ctrl_arr[1:]
        # last stays zero
        return tail

    def _exaggerate_positions(self, pos_seq: np.ndarray) -> np.ndarray:
        """
        Scale each step's delta by exaggeration_factor for visualization only.
        pos_seq: (T, 2)
        """
        if pos_seq is None or len(pos_seq) == 0:
            return pos_seq

        factor = float(self.exaggeration_factor)
        if factor == 1.0:
            return pos_seq  # no exaggeration

        # Differences between consecutive points
        deltas = np.diff(pos_seq, axis=0)    # shape: (T-1, 2)

        # Scale the deltas
        deltas_scaled = deltas * factor

        # Rebuild exaggerated trajectory
        exaggerated = [pos_seq[0]]
        current = pos_seq[0]
        for d in deltas_scaled:
            current = current + d
            exaggerated.append(current.copy())

        return np.asarray(exaggerated)


    def _plot_predicted_positions(self,
                                  ax,
                                  graph_data: Dict,
                                  log_dir: pathlib.Path,
                                  step_idx: int):
        """
        Plot predicted future positions:
          - Ego: this step's sequence (if step fails, use fallback control as single-step).
          - Neighbors: tail+zero of previous step (zeros on failure).
          - Dotted lines start at current position.
          - Points are small colored dots with black edge.
        """
        # Need local→global mapping
        local_agent_indices = graph_data.get('local_agent_indices', None)
        if local_agent_indices is None:
            return

        local_agent_indices = np.asarray(local_agent_indices, dtype=int)

        node_types = np.asarray(graph_data['node_type'])
        states = np.asarray(graph_data['states'])

        # Indices of agent nodes in this local graph
        agent_node_indices = np.where(node_types == 0)[0]
        if len(agent_node_indices) == 0:
            return

        # Align lengths defensively
        n_align = min(len(agent_node_indices), len(local_agent_indices))
        if n_align == 0:
            return

        agent_node_indices = agent_node_indices[:n_align]
        local_agent_indices = local_agent_indices[:n_align]

        ego_global_idx = graph_data.get('agent_idx', None)
        if ego_global_idx is None:
            return

        # Small cache so we don't reload CSVs repeatedly in one call
        df_cache: Dict[int, Optional[pd.DataFrame]] = {}

        def get_df(global_idx: int) -> Optional[pd.DataFrame]:
            if global_idx not in df_cache:
                df_cache[global_idx] = self.load_step_logs(log_dir, int(global_idx))
            return df_cache[global_idx]

        # ---------------- Ego: current step prediction ----------------
        ego_pred_positions = None
        ego_current_xy = None

        for node_idx, global_idx in zip(agent_node_indices, local_agent_indices):
            if global_idx == ego_global_idx:
                ego_current_xy = states[node_idx, :2]
                df_ego = get_df(global_idx)
                if df_ego is not None and 0 <= step_idx < len(df_ego):
                    row = df_ego.iloc[step_idx]
                    ctrl_arr = self._parse_control_sequence(row.get('control_input', None))
                    if ctrl_arr is not None:
                        # If this step "fails", still show a 1-step prediction
                        if self._is_step_failure(row):
                            ctrl_arr = ctrl_arr[:1]   # single-step fallback control
                        # Roll out and exaggerate
                        ego_pred_positions = self._rollout_positions(states[node_idx], ctrl_arr, include_start=True)
                        ego_pred_positions = self._exaggerate_positions(ego_pred_positions)
                break

        # ---------------- Neighbors: previous-step tail+zero ----------------
        neighbor_preds: List[Tuple[np.ndarray, np.ndarray]] = []
        prev_step = step_idx - 1

        for node_idx, global_idx in zip(agent_node_indices, local_agent_indices):
            if global_idx == ego_global_idx:
                continue  # skip ego here

            df_nei = get_df(global_idx)
            tail_seq = self._get_neighbor_tail_sequence(df_nei, prev_step)
            if tail_seq is None:
                continue

            pos_seq = self._rollout_positions(states[node_idx], tail_seq, include_start=True)
            if pos_seq is None or len(pos_seq) == 0:
                continue

            pos_seq = self._exaggerate_positions(pos_seq)
            neighbor_preds.append((states[node_idx, :2], pos_seq))

        # ---------------- Draw them: dotted lines + colored dots ----------------
        line_z = 15
        dot_z = 16

        # Ego prediction
        if ego_pred_positions is not None and len(ego_pred_positions) > 0:
            if ego_current_xy is not None:
                ego_line_xy = np.vstack([ego_current_xy, ego_pred_positions])
            else:
                ego_line_xy = ego_pred_positions

            # Dotted line from current position through predicted positions
            ax.plot(
                ego_line_xy[:, 0],
                ego_line_xy[:, 1],
                linestyle=':',
                linewidth=1.5,
                color='black',
                alpha=0.9,
                zorder=line_z,
            )
            # Small colored dots with black edge at predicted positions
            ax.scatter(
                ego_pred_positions[:, 0],
                ego_pred_positions[:, 1],
                s=12,
                facecolor=self.colors['ego_agent'],
                edgecolor='black',
                linewidth=0.8,
                alpha=0.95,
                zorder=dot_z,
                marker='o',
                label='Predicted (ego & neighbors)',
            )

        # Neighbor predictions
        for current_xy, pos_seq in neighbor_preds:
            if pos_seq is None or len(pos_seq) == 0:
                continue

            nei_line_xy = np.vstack([current_xy, pos_seq])

            ax.plot(
                nei_line_xy[:, 0],
                nei_line_xy[:, 1],
                linestyle=':',
                linewidth=1.0,
                color='black',
                alpha=0.7,
                zorder=line_z,
            )
            ax.scatter(
                pos_seq[:, 0],
                pos_seq[:, 1],
                s=10,
                facecolor=self.colors['other_agent'],
                edgecolor='black',
                linewidth=0.7,
                alpha=0.8,
                zorder=dot_z,
                marker='o',
            )



    # -------------------------------
    # Step visualization
    # -------------------------------

    def load_step_logs(self, log_dir: pathlib.Path, agent_idx: int) -> Optional[pd.DataFrame]:
        """
        Load agent step logs from CSV file.

        Args:
            log_dir: Root directory containing agent_step_logs
            agent_idx: Agent index

        Returns:
            DataFrame with step log data or None if not found
        """
        cache_key = (log_dir, agent_idx)
        if cache_key in self.step_logs_cache:
            return self.step_logs_cache[cache_key]

        # Look for agent_step_logs directory
        step_log_dirs = list(log_dir.glob("**/agent_step_logs"))

        if not step_log_dirs:
            print(f"Warning: No agent_step_logs directory found in {log_dir}")
            return None

        step_log_dir = step_log_dirs[0]
        csv_path = step_log_dir / f"agent_{agent_idx:02d}.csv"

        if not csv_path.exists():
            print(f"Warning: Step log not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            self.step_logs_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error loading step logs from {csv_path}: {e}")
            return None

    def get_step_status(self, df: pd.DataFrame, step_idx: int) -> Dict:
        """
        Extract status information for a specific step.

        Args:
            df: Step log DataFrame
            step_idx: Step index

        Returns:
            Dictionary with status info
        """
        if df is None or step_idx >= len(df):
            return {
                'controller': 'UNKNOWN',
                'status': 'N/A',
                'cbf_initial': None,
                'control_input': None,
                'color': '#888888'
            }

        row = df.iloc[step_idx]

        # Determine controller type and status
        controller = row.get('controller_used', 'UNKNOWN')
        mpc_status = row.get('mpc_status', '')

        # Determine status color
        if controller == 'MPC':
            status_key = 'MPC'
            status_text = 'MPC Success'
        elif controller == 'SAFE_QP':
            if 'Positive directional derivative' in mpc_status:
                status_key = 'POSITIVE_DERIVATIVE'
                status_text = 'Positive Deriv'
            else:
                status_key = 'SAFE_QP'
                status_text = f'Safe QP: {mpc_status}'
        else:
            status_key = 'SAFE_QP'
            status_text = f'{controller}'

        return {
            'controller': controller,
            'status_text': status_text,
            'status_key': status_key,
            'cbf_initial': row.get('cbf_initial', None),
            'control_input': row.get('control_input', None),
            'mpc_iterations': row.get('mpc_iterations', None),
            'color': self.status_colors.get(status_key, '#888888')
        }

    def _add_status_indicator(self, ax, status_info: Dict, position: str = 'bottom'):
        """
        Add status indicator to the plot.

        Args:
            ax: Matplotlib axis
            status_info: Status information dictionary
            position: 'top' or 'bottom' for indicator placement
        """
        if position == 'bottom':
            y_pos = 0.02
            va = 'bottom'
        else:
            y_pos = 0.98
            va = 'top'

        # Create status text lines
        status_lines = []

        # First line with status (dot will be added separately before this)
        status_lines.append(f"  {status_info['status_text']}")  # Added spacing for the dot

        # Add CBF value if available
        if status_info['cbf_initial'] is not None:
            try:
                cbf_val = float(status_info['cbf_initial'])
                status_lines.append(f"  CBF₀: {cbf_val:.4f}")
            except (ValueError, TypeError):
                pass

        # Add iterations if available
        if status_info.get('mpc_iterations') is not None:
            status_lines.append(f"  Iter: {status_info['mpc_iterations']}")

        # Add control input preview if available
        if status_info['control_input'] is not None:
            try:
                # Parse control input string
                control_str = str(status_info['control_input']).strip()

                # Try to parse as list/array to format nicely
                import ast
                import re

                # Remove various array/numpy wrappers
                control_str = re.sub(r'array\s*\(', '', control_str)
                control_str = re.sub(r'\)\s*$', '', control_str)
                control_str = control_str.strip()

                try:
                    # Try to evaluate as Python literal
                    control_vals = ast.literal_eval(control_str)

                    # If it's a list/tuple with multiple values, show one per line
                    if isinstance(control_vals, (list, tuple)):
                        if len(control_vals) > 1:
                            status_lines.append("  Control:")
                            for i, val in enumerate(control_vals):
                                if isinstance(val, (int, float)):
                                    status_lines.append(f"    u[{i}]: {val:.4f}")
                                else:
                                    status_lines.append(f"    u[{i}]: {val}")
                        elif len(control_vals) == 1:
                            val = control_vals[0]
                            if isinstance(val, (int, float)):
                                status_lines.append(f"  Control: {val:.4f}")
                            else:
                                status_lines.append(f"  Control: {val}")
                    else:
                        # Single value
                        if isinstance(control_vals, (int, float)):
                            status_lines.append(f"  Control: {control_vals:.4f}")
                        else:
                            status_lines.append(f"  Control: {control_vals}")

                except (ValueError, SyntaxError) as e:
                    # If parsing fails, try to extract numbers manually
                    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', control_str)
                    if len(numbers) > 1:
                        status_lines.append("  Control:")
                        for i, num in enumerate(numbers):
                            try:
                                val = float(num)
                                status_lines.append(f"    u[{i}]: {val:.4f}")
                            except ValueError:
                                status_lines.append(f"    u[{i}]: {num}")
                    elif len(numbers) == 1:
                        try:
                            val = float(numbers[0])
                            status_lines.append(f"  Control: {val:.4f}")
                        except ValueError:
                            status_lines.append(f"  Control: {numbers[0]}")
                    else:
                        # Last resort: just show truncated string
                        if len(control_str) > 60:
                            control_str = control_str[:60] + "..."
                        status_lines.append(f"  Control: {control_str}")
            except Exception as e:
                # If all else fails, show the raw string (truncated)
                control_str = str(status_info['control_input'])
                if len(control_str) > 60:
                    control_str = control_str[:60] + "..."
                status_lines.append(f"  Control: {control_str}")

        # Join all lines
        status_text = "\n".join(status_lines)

        # Map status to lighter background colors
        background_colors = {
            'MPC': '#CCFFCC',  # Light green
            'SAFE_QP': '#FFEECC',  # Light red
            'POSITIVE_DERIVATIVE': '#FFCCCC'  # Light orange
        }

        bg_color = background_colors.get(status_info['status_key'], '#EEEEEE')

        # Add text box with thin black border and colored background
        bbox_props = dict(
            boxstyle='round,pad=0.5',
            facecolor=bg_color,
            edgecolor='black',
            linewidth=1,
            alpha=0.9
        )

        # Create the main text with black color
        ax.text(0.02, y_pos, status_text,
                transform=ax.transAxes,
                verticalalignment=va,
                bbox=bbox_props,
                fontsize=9,
                family='monospace',
                color='black',
                fontweight='normal')

        # Add colored dot indicator at the beginning of the first line
        # Position it just before the status text
        ax.text(0.025, y_pos, '●',
                transform=ax.transAxes,
                verticalalignment=va,
                fontsize=12,
                color=status_info['color'],
                fontweight='bold',
                zorder=100)

    def _set_axis_limits(self, ax, states: np.ndarray, node_types: np.ndarray):
        """Set axis limits based on mode."""
        if self.axis_mode == 'fixed' and self.axis_limits:
            xmin, xmax, ymin, ymax = self.axis_limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        elif self.axis_mode == 'tight':
            logical_mask = node_types != -1
            if np.any(logical_mask):
                positions = states[logical_mask, :2]
                ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
                ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())

        elif self.axis_mode == 'equal':
            logical_mask = node_types != -1
            if np.any(logical_mask):
                positions = states[logical_mask, :2]
                xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
                ymin, ymax = positions[:, 1].min(), positions[:, 1].max()

                x_range = xmax - xmin
                y_range = ymax - ymin
                max_range = max(x_range, y_range)

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                half_range = max_range / 2 + self.padding

                ax.set_xlim(x_center - half_range, x_center + half_range)
                ax.set_ylim(y_center - half_range, y_center + half_range)

        else:  # 'auto'
            logical_mask = node_types != -1
            if np.any(logical_mask):
                positions = states[logical_mask, :2]
                xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
                ymin, ymax = positions[:, 1].min(), positions[:, 1].max()

                x_padding = (xmax - xmin) * 0.1 + self.padding
                y_padding = (ymax - ymin) * 0.1 + self.padding

                ax.set_xlim(xmin - x_padding, xmax + x_padding)
                ax.set_ylim(ymin - y_padding, ymax + y_padding)

    def load_graph(self, pkl_path: pathlib.Path) -> Dict:
        """Load a single graph from pickle file."""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            return None

    def extract_node_groups(self, graph_data: Dict) -> Dict[str, np.ndarray]:
        """Extract nodes by type, handling padding correctly."""
        node_types = graph_data['node_type']
        states = graph_data['states']

        logical_mask = node_types != -1
        logical_indices = np.where(logical_mask)[0]
        logical_types = node_types[logical_mask]

        agent_mask = logical_types == 0
        goal_mask = logical_types == 1
        lidar_mask = logical_types == 2

        agent_indices = logical_indices[agent_mask]
        goal_indices = logical_indices[goal_mask]
        lidar_indices = logical_indices[lidar_mask]
        padding_indices = np.where(node_types == -1)[0]

        ego_idx = agent_indices[0] if len(agent_indices) > 0 else None
        other_agent_indices = agent_indices[1:] if len(agent_indices) > 1 else np.array([])

        return {
            'ego_agent': np.array([ego_idx]) if ego_idx is not None else np.array([]),
            'other_agents': other_agent_indices,
            'goals': goal_indices,
            'lidar': lidar_indices,
            'padding': padding_indices
        }

    def extract_edges(self, graph_data: Dict) -> Dict[str, List[Tuple[int, int]]]:
        """Extract edges by type."""
        senders = graph_data['senders']
        receivers = graph_data['receivers']
        node_types = graph_data['node_type']

        edges = {
            'agent_agent': [],
            'agent_goal': [],
            'agent_lidar': []
        }

        for s, r in zip(senders, receivers):
            s_type = node_types[s]
            r_type = node_types[r]

            if s_type == -1 or r_type == -1:
                continue

            if s_type == 0 and r_type == 0:
                edges['agent_agent'].append((s, r))
            elif (s_type == 0 and r_type == 1) or (s_type == 1 and r_type == 0):
                edges['agent_goal'].append((s, r))
            elif (s_type == 0 and r_type == 2) or (s_type == 2 and r_type == 0):
                edges['agent_lidar'].append((s, r))

        return edges

    def plot_graph(self,
                   graph_data: Dict,
                   save_path: Optional[pathlib.Path] = None,
                   show: bool = False,
                   title: Optional[str] = None,
                   override_axis_mode: Optional[str] = None,
                   override_axis_limits: Optional[Tuple[float, float, float, float]] = None,
                   log_dir: Optional[pathlib.Path] = None) -> plt.Figure:
        """
        Create visualization of a single graph with status indicator and state info.

        Args:
            graph_data: Loaded graph dictionary
            save_path: Where to save figure
            show: Whether to display interactively
            title: Custom title for plot
            override_axis_mode: Override default axis mode
            override_axis_limits: Override axis limits
            log_dir: Root log directory for loading step logs
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        states = graph_data['states']
        agent_idx = graph_data.get('agent_idx', 'Unknown')
        step_idx = graph_data.get('step_idx', 'Unknown')

        # Load step status if available
        status_info = None
        if log_dir is not None and isinstance(agent_idx, int) and isinstance(step_idx, int):
            step_logs = self.load_step_logs(log_dir, agent_idx)
            status_info = self.get_step_status(step_logs, step_idx)

        # Extract node groups
        node_groups = self.extract_node_groups(graph_data)

        # Plot edges first
        edges = self.extract_edges(graph_data)

        if edges['agent_agent']:
            lines = [[states[s, :2], states[r, :2]] for s, r in edges['agent_agent']]
            lc = LineCollection(lines, colors=self.colors['edge_agent'],
                                linewidths=1.5, alpha=0.8, label='Agent-Agent')
            ax.add_collection(lc)

        if edges['agent_goal']:
            lines = [[states[s, :2], states[r, :2]] for s, r in edges['agent_goal']]
            lc = LineCollection(lines, colors=self.colors['edge_goal'],
                                linewidths=2.0, alpha=0.6, linestyles='dashed',
                                label='Agent-Goal')
            ax.add_collection(lc)

        if edges['agent_lidar']:
            lines = [[states[s, :2], states[r, :2]] for s, r in edges['agent_lidar']]
            lc = LineCollection(lines, colors=self.colors['edge_lidar'],
                                linewidths=1.2, alpha=0.5, label='Agent-LiDAR')
            ax.add_collection(lc)

        # Plot nodes
        if len(node_groups['ego_agent']) > 0:
            ego_pos = states[node_groups['ego_agent'][0], :2]
            ego_vel = states[node_groups['ego_agent'][0], 2:4]

            ax.scatter(ego_pos[0], ego_pos[1],
                       c=self.colors['ego_agent'],
                       marker=self.markers['ego_agent'],
                       s=self.sizes['ego_agent'],
                       edgecolors='black', linewidths=0.3,
                       label='Ego Agent', zorder=10)

            # vel_scale = 0.15
            # ax.arrow(ego_pos[0], ego_pos[1],
            #          ego_vel[0] * vel_scale, ego_vel[1] * vel_scale,
            #          head_width=0.05, head_length=0.03,
            #          fc=self.colors['ego_agent'], ec='black',
            #          linewidth=1.5, zorder=9)

        if len(node_groups['other_agents']) > 0:
            other_pos = states[node_groups['other_agents'], :2]
            other_vel = states[node_groups['other_agents'], 2:4]

            ax.scatter(other_pos[:, 0], other_pos[:, 1],
                       c=self.colors['other_agent'],
                       marker=self.markers['other_agent'],
                       s=self.sizes['other_agent'],
                       edgecolors='black', linewidths=0.3,
                       label='Other Agents', zorder=8)

            # vel_scale = 0.15
            # for pos, vel in zip(other_pos, other_vel):
            #     ax.arrow(pos[0], pos[1],
            #              vel[0] * vel_scale, vel[1] * vel_scale,
            #              head_width=0.04, head_length=0.025,
            #              fc=self.colors['other_agent'], ec='black',
            #              linewidth=1.0, alpha=0.7, zorder=7)

        # Plot predicted future positions:
        #   - Ego: this step's sequence
        #   - Neighbors: tail+zero of previous step
        if log_dir is not None and isinstance(agent_idx, int) and isinstance(step_idx, int):
            try:
                self._plot_predicted_positions(ax, graph_data, log_dir, step_idx)
            except Exception as e:
                print(f"Warning: failed to plot predicted positions: {e}")


        if len(node_groups['goals']) > 0:
            goal_pos = states[node_groups['goals'], :2]
            ax.scatter(goal_pos[:, 0], goal_pos[:, 1],
                       c=self.colors['goal'],
                       marker=self.markers['goal'],
                       s=self.sizes['goal'],
                       edgecolors='black', linewidths=1.5,
                       label='Goals', zorder=7)

        if len(node_groups['lidar']) > 0:
            lidar_pos = states[node_groups['lidar'], :2]
            ax.scatter(lidar_pos[:, 0], lidar_pos[:, 1],
                       c=self.colors['lidar'],
                       marker=self.markers['lidar'],
                       s=self.sizes['lidar'],
                       alpha=0.6,
                       label='LiDAR Hits', zorder=5)

        if len(node_groups['padding']) > 0:
            padding_pos = states[node_groups['padding'], :2]
            ax.scatter(padding_pos[:, 0], padding_pos[:, 1],
                       c=self.colors['padding'],
                       marker='.',
                       s=50,
                       alpha=0.3,
                       label='Padding', zorder=1)

        # Set axis limits
        axis_mode = override_axis_mode if override_axis_mode else self.axis_mode
        axis_limits = override_axis_limits if override_axis_limits else self.axis_limits

        old_mode = self.axis_mode
        old_limits = self.axis_limits
        self.axis_mode = axis_mode
        self.axis_limits = axis_limits

        self._set_axis_limits(ax, states, graph_data['node_type'])

        self.axis_mode = old_mode
        self.axis_limits = old_limits

        # Formatting
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Local Subgraph - Agent {agent_idx}, Step {step_idx}',
                         fontsize=14, fontweight='bold')

        # Add ego agent state info at upper left corner
        if len(node_groups['ego_agent']) > 0:
            ego_idx = node_groups['ego_agent'][0]
            ego_state = states[ego_idx]

            state_text = "Ego Agent State:\n"
            state_text += f"Position:\n"
            state_text += f"  x: {ego_state[0]:.4f} m\n"
            state_text += f"  y: {ego_state[1]:.4f} m\n"
            state_text += f"Velocity:\n"
            state_text += f"  vx: {ego_state[2]:.4f} m/s\n"
            state_text += f"  vy: {ego_state[3]:.4f} m/s\n"

            # Calculate speed
            speed = np.sqrt(ego_state[2] ** 2 + ego_state[3] ** 2)
            state_text += f"Speed: {speed:.4f} m/s"

            ax.text(0.02, 0.98, state_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')

        # Add node count info text (at middle-right)
        info_text = f"Nodes: {len(states)} total\n"
        info_text += f"  Agents: {len(node_groups['ego_agent']) + len(node_groups['other_agents'])}\n"
        info_text += f"  Goals: {len(node_groups['goals'])}\n"
        info_text += f"  LiDAR: {len(node_groups['lidar'])}\n"
        info_text += f"  Padding: {len(node_groups['padding'])}\n"
        info_text += f"Edges: {sum(len(v) for v in edges.values())}"

        ax.text(0.98, 0.50, info_text,
                transform=ax.transAxes,
                verticalalignment='center',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')

        # Add status indicator at bottom
        if status_info is not None:
            self._add_status_indicator(ax, status_info, position='bottom')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_agent_trajectory(self,
                                   log_dir: pathlib.Path,
                                   agent_idx: int,
                                   output_dir: Optional[pathlib.Path] = None,
                                   step_range: Optional[Tuple[int, int]] = None,
                                   create_animation: bool = False):
        """Visualize all graphs for a single agent."""
        agent_dirs = list(log_dir.glob(f"ep*/agent{agent_idx:02d}"))

        if not agent_dirs:
            print(f"No data found for agent {agent_idx}")
            return

        agent_dir = agent_dirs[0]
        episode_name = agent_dir.parent.name

        print(f"Processing {agent_dir}")

        if output_dir is None:
            output_dir = log_dir / "visualizations" / episode_name / f"agent{agent_idx:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        step_files = sorted(agent_dir.glob("step*.pkl"))

        if not step_files:
            print(f"No step files found in {agent_dir}")
            return

        print(f"Found {len(step_files)} steps")

        if step_range:
            start, end = step_range
            step_files = [f for f in step_files
                          if start <= int(f.stem.replace('step', '')) <= end]
            print(f"Filtered to {len(step_files)} steps in range [{start}, {end}]")

        # Get root log directory for step logs
        root_log_dir = log_dir.parent if log_dir.name == "local_graph_logs" else log_dir

        image_paths = []
        for step_file in step_files:
            graph_data = self.load_graph(step_file)
            if graph_data is None:
                continue

            step_num = int(step_file.stem.replace('step', ''))
            output_path = output_dir / f"step{step_num:05d}.png"

            title = f"Agent {agent_idx} - Step {step_num} ({episode_name})"
            self.plot_graph(graph_data, save_path=output_path, title=title,
                            log_dir=root_log_dir)
            image_paths.append(output_path)

        print(f"Created {len(image_paths)} visualizations in {output_dir}")

        if create_animation and image_paths:
            try:
                from PIL import Image

                gif_path = output_dir / f"agent{agent_idx:02d}_trajectory.gif"

                images = [Image.open(p) for p in image_paths]
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=200,
                    loop=0
                )

                print(f"Created animation: {gif_path}")

            except ImportError:
                print("PIL not available - skipping animation creation")
            except Exception as e:
                print(f"Animation creation failed: {e}")

    def compare_agents(self,
                       log_dir: pathlib.Path,
                       step_num: int,
                       agent_indices: Optional[List[int]] = None,
                       output_path: Optional[pathlib.Path] = None):
        """Create side-by-side comparison of multiple agents at same timestep."""
        if agent_indices is None:
            agent_dirs = sorted(log_dir.glob("ep*/agent*"))
            agent_indices = [int(d.name.replace('agent', '')) for d in agent_dirs]
        else:
            agent_dirs = [log_dir / f"ep00/agent{i:02d}" for i in agent_indices]

        n_agents = len(agent_indices)

        if n_agents == 0:
            print("No agents found")
            return

        cols = min(3, n_agents)
        rows = (n_agents + cols - 1) // cols

        fig = plt.figure(figsize=(self.figure_size[0] * cols / 2,
                                  self.figure_size[1] * rows / 2),
                         dpi=self.dpi)

        for idx, (agent_idx, agent_dir) in enumerate(zip(agent_indices, agent_dirs)):
            step_file = agent_dir / f"step{step_num:05d}.pkl"

            if not step_file.exists():
                print(f"Step {step_num} not found for agent {agent_idx}")
                continue

            graph_data = self.load_graph(step_file)
            if graph_data is None:
                continue

            ax = fig.add_subplot(rows, cols, idx + 1)

            states = graph_data['states']
            node_types = graph_data['node_type']

            for node_type, color, marker in [(0, self.colors['ego_agent'], 'o'),
                                             (1, self.colors['goal'], '*'),
                                             (2, self.colors['lidar'], 'x')]:
                mask = node_types == node_type
                if np.any(mask):
                    pos = states[mask, :2]
                    ax.scatter(pos[:, 0], pos[:, 1], c=color, marker=marker, s=50)

            ax.set_title(f"Agent {agent_idx}", fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Agent Comparison - Step {step_num}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved comparison: {output_path}")
        else:
            plt.show()

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize MPC-GCBF local subgraphs with status indicators')
    parser.add_argument('log_dir', type=str,
                        help='Path to local_graph_logs directory')
    parser.add_argument('--agent', type=int, default=0,
                        help='Agent index to visualize')
    parser.add_argument('--step-range', type=int, nargs=2, metavar=('START', 'END'),
                        help='Range of steps to visualize')
    parser.add_argument('--output-dir', type=str,
                        help='Custom output directory for visualizations')
    parser.add_argument('--animate', action='store_true',
                        help='Create animated GIF of trajectory')
    parser.add_argument('--compare-agents', action='store_true',
                        help='Create side-by-side comparison of all agents at a step')
    parser.add_argument('--compare-step', type=int, default=0,
                        help='Step number for agent comparison')
    parser.add_argument('--single-step', type=int,
                        help='Visualize only a single step')
    parser.add_argument('--axis-mode', type=str, default='auto',
                        choices=['auto', 'equal', 'tight', 'fixed'],
                        help='Axis scaling mode')
    parser.add_argument('--axis-limits', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Fixed axis limits')
    parser.add_argument('--padding', type=float, default=0.5,
                        help='Padding around data in auto/equal modes')
    parser.add_argument('--figure-size', type=int, nargs=2, default=[12, 10],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Figure resolution')

    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        return

    visualizer = LocalGraphVisualizer(
        figure_size=tuple(args.figure_size),
        dpi=args.dpi,
        axis_mode=args.axis_mode,
        axis_limits=tuple(args.axis_limits) if args.axis_limits else None,
        padding=args.padding
    )

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

    if args.compare_agents:
        compare_output = (output_dir or log_dir / "visualizations") / f"comparison_step{args.compare_step:05d}.png"
        visualizer.compare_agents(log_dir, args.compare_step, output_path=compare_output)

    elif args.single_step is not None:
        agent_dir = list(log_dir.glob(f"ep*/agent{args.agent:02d}"))[0]
        step_file = agent_dir / f"step{args.single_step:05d}.pkl"

        if not step_file.exists():
            print(f"Error: {step_file} does not exist")
            return

        graph_data = visualizer.load_graph(step_file)
        output_path = (
                                  output_dir or log_dir / "visualizations") / f"agent{args.agent:02d}_step{args.single_step:05d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        root_log_dir = log_dir.parent if log_dir.name == "local_graph_logs" else log_dir
        visualizer.plot_graph(graph_data, save_path=output_path,
                              title=f"Agent {args.agent} - Step {args.single_step}",
                              log_dir=root_log_dir)

    else:
        step_range = tuple(args.step_range) if args.step_range else None
    visualizer.visualize_agent_trajectory(
        log_dir,
        args.agent,
        output_dir=output_dir,
        step_range=step_range,
        create_animation=args.animate
    )

if __name__ == '__main__':
    main()


# # Basic visualization of agent 0
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1022-0115/local_graph_logs --agent 1 --step-range 0 159 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1020-1745/local_graph_logs --agent 0 --step-range 0 159 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1023-1243/local_graph_logs --agent 1 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1024-1625/local_graph_logs --agent 1 --step-range 30 45 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1028-0148/local_graph_logs --agent 3 --step-range 0 250 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1030-0117/local_graph_logs --agent 0 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1030-1816/local_graph_logs --agent 0 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1030-1816/local_graph_logs --agent 0 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1031-1711/local_graph_logs --agent 0 --step-range 0 119 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1104-1509/local_graph_logs --agent 0 --step-range 0 119 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy/1105-0949/local_graph_logs --agent 3 --step-range 0 200 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1113-1828/local_graph_logs --agent 1 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1126-1920/local_graph_logs --agent 1 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4

# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1117-1623/local_graph_logs --agent 1 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy/1114-0203/local_graph_logs --agent 4 --step-range 0 300 --axis-mode fixed --axis-limits -0 4 -0 4



#
# # With animation
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1020-1558/local_graph_logs --agent 0 --step-range 0 50 --animate
#
# # Compare all agents at step 10
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1020-1558/local_graph_logs --compare-agents --compare-step 10
#
# # View single step interactively
# python visualize_local_graphs.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1020-1558/local_graph_logs --agent 0 --single-step 5

