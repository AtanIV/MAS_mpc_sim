#!/usr/bin/env python3
"""
Visualize logged *full* graphs from MPC–GCBF testing, with:

- all agents/goals/LiDAR nodes,
- MPC-predicted trajectories (from agent_step_logs),
- obstacle polygons (from obstacle_vertices or obstacles.csv),
- per-agent colored trails of past positions (fading over time).
"""

import argparse
import json
import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


class FullGraphVisualizer:
    """Visualize full graphs with agents, trajectories, obstacles, and trails."""

    def __init__(
        self,
        figure_size: Tuple[int, int] = (12, 10),
        dpi: int = 100,
        axis_mode: str = "auto",
        axis_limits: Optional[Tuple[float, float, float, float]] = None,
        padding: float = 0.5,
        dt: float = 0.03,
        mass: float = 0.1,
        max_pred_steps: int = 16,
        exaggeration_factor: float = 7.0,
        vel_arrow_scale: float = 0.15,
        max_trail_steps: int = 10,
    ):
        self.figure_size = figure_size
        self.dpi = dpi
        self.axis_mode = axis_mode
        self.axis_limits = axis_limits
        self.padding = padding

        self.dt = dt
        self.mass = mass
        self.max_pred_steps = max_pred_steps
        self.exaggeration_factor = exaggeration_factor
        self.vel_arrow_scale = vel_arrow_scale

        # Trail settings
        self.max_trail_steps = max_trail_steps
        # “transparency 20% -> 90%” => opacity 0.8 -> 0.1
        self.trail_alpha_recent = 0.8
        self.trail_alpha_old = 0.1

        self.colors = {
            "agent": "#4444FF",
            "goal": "#44FF44",
            "lidar": "#FF0000",
            "padding": "#DDDDDD",
            "edge_agent": "#0066CC",
            "edge_goal": "#00CC66",
            "edge_lidar": "#333333",
            "predicted": "#FF8800",
            "obstacle": "#AA0000",
        }

        self.markers = {"agent": "o", "goal": "*", "lidar": "o"}
        self.sizes = {"agent": 80, "goal": 200, "lidar": 15}

        # Colormap for per-agent trail colors
        self.trail_cmap = plt.get_cmap("tab10")

        # Caches
        self.step_logs_cache: Dict[Tuple[pathlib.Path, int], Optional[pd.DataFrame]] = {}
        self.obstacle_logs_cache: Dict[pathlib.Path, Optional[pd.DataFrame]] = {}

        # Disturbance logs (for drawing a global disturbance arrow)
        self.disturbance_logs_cache: Dict[pathlib.Path, Optional[pd.DataFrame]] = {}
        self.disturbance_vector_cache: Dict[Tuple[pathlib.Path, int, int], Optional[np.ndarray]] = {}


    # ------------------------------------------------------------------
    # Axis control
    # ------------------------------------------------------------------

    def _set_axis_limits(self, ax, states: np.ndarray, node_types: np.ndarray):
        if self.axis_mode == "fixed" and self.axis_limits:
            xmin, xmax, ymin, ymax = self.axis_limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            return

        logical_mask = node_types != -1
        if not np.any(logical_mask):
            return

        positions = states[logical_mask, :2]

        if self.axis_mode == "tight":
            ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
            ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
        elif self.axis_mode == "equal":
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
        else:  # auto
            xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
            ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
            x_pad = (xmax - xmin) * 0.1 + self.padding
            y_pad = (ymax - ymin) * 0.1 + self.padding
            ax.set_xlim(xmin - x_pad, xmax + x_pad)
            ax.set_ylim(ymin - y_pad, ymax + y_pad)

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_graph(pkl_path: pathlib.Path) -> Optional[Dict]:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"[FullGraphVis] Error loading {pkl_path}: {e}")
            return None

    @staticmethod
    def extract_node_groups(graph_data: Dict) -> Dict[str, np.ndarray]:
        node_types = np.asarray(graph_data["node_type"])
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

        return {
            "agents": agent_indices,
            "goals": goal_indices,
            "lidar": lidar_indices,
            "padding": padding_indices,
        }

    @staticmethod
    def extract_edges(graph_data: Dict) -> Dict[str, List[Tuple[int, int]]]:
        senders = np.asarray(graph_data["senders"])
        receivers = np.asarray(graph_data["receivers"])
        node_types = np.asarray(graph_data["node_type"])

        edges = {"agent_agent": [], "agent_goal": [], "agent_lidar": []}

        for s, r in zip(senders, receivers):
            s_type = node_types[s]
            r_type = node_types[r]
            if s_type == -1 or r_type == -1:
                continue
            if s_type == 0 and r_type == 0:
                edges["agent_agent"].append((s, r))
            elif (s_type == 0 and r_type == 1) or (s_type == 1 and r_type == 0):
                edges["agent_goal"].append((s, r))
            elif (s_type == 0 and r_type == 2) or (s_type == 2 and r_type == 0):
                edges["agent_lidar"].append((s, r))

        return edges

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------

    def _get_obstacle_df(self, run_root: pathlib.Path) -> Optional[pd.DataFrame]:
        if run_root in self.obstacle_logs_cache:
            return self.obstacle_logs_cache[run_root]

        csv_path = run_root / "obstacles.csv"
        if not csv_path.exists():
            print(f"[FullGraphVis] No obstacles.csv found at {csv_path}")
            self.obstacle_logs_cache[run_root] = None
            return None

        try:
            df = pd.read_csv(csv_path)
            self.obstacle_logs_cache[run_root] = df
            return df
        except Exception as e:
            print(f"[FullGraphVis] Failed to read {csv_path}: {e}")
            self.obstacle_logs_cache[run_root] = None
            return None

    def _plot_obstacles_for_step(
        self,
        ax,
        run_root: pathlib.Path,
        episode: int,
        step: int,
        graph_data: Optional[Dict] = None,
    ):
        """
        Draw obstacles for a given (episode, step).

        Priority:
          1) graph_data["obstacle_vertices"] (preferred).
          2) obstacles.csv new format with v0_x..v3_y.
          3) obstacles.csv old format with pos_x,pos_y (small squares).
        """
        # 1) From pickle: obstacle_vertices
        if graph_data is not None and "obstacle_vertices" in graph_data:
            verts = graph_data["obstacle_vertices"]
            if verts is not None:
                verts = np.asarray(verts)
                for i, obs_verts in enumerate(verts):
                    try:
                        poly = patches.Polygon(
                            obs_verts,
                            linewidth=2.0,
                            edgecolor=self.colors["obstacle"],
                            facecolor=self.colors["obstacle"],
                            alpha=0.3,
                            closed=True,
                            zorder=3,
                        )
                        ax.add_patch(poly)
                        center = obs_verts.mean(axis=0)
                        ax.text(
                            center[0],
                            center[1],
                            f"Obs{i}",
                            fontsize=9,
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                alpha=0.8,
                            ),
                            zorder=5,
                        )
                    except Exception as e:
                        print(f"[FullGraphVis] Failed to draw obstacle polygon {i}: {e}")
                return

        # 2) Fallback: obstacles.csv
        df = self._get_obstacle_df(run_root)
        if df is None:
            return

        sub = df[(df["episode"] == episode) & (df["step"] == step)]
        if sub.empty:
            return

        columns = set(df.columns)

        # New format with vertices
        if {"v0_x", "v0_y", "v1_x", "v1_y", "v2_x", "v2_y", "v3_x", "v3_y"}.issubset(
            columns
        ):
            for _, row in sub.iterrows():
                verts = np.array(
                    [
                        [row["v0_x"], row["v0_y"]],
                        [row["v1_x"], row["v1_y"]],
                        [row["v2_x"], row["v2_y"]],
                        [row["v3_x"], row["v3_y"]],
                    ],
                    dtype=float,
                )
                try:
                    poly = patches.Polygon(
                        verts,
                        linewidth=2.0,
                        edgecolor=self.colors["obstacle"],
                        facecolor=self.colors["obstacle"],
                        alpha=0.3,
                        closed=True,
                        zorder=3,
                    )
                    ax.add_patch(poly)

                    cx = row.get("center_x", verts[:, 0].mean())
                    cy = row.get("center_y", verts[:, 1].mean())
                    obs_id = int(row.get("obstacle_id", -1))
                    ax.text(
                        cx,
                        cy,
                        f"Obs{obs_id}",
                        fontsize=9,
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.8,
                        ),
                        zorder=5,
                    )
                except Exception as e:
                    print(f"[FullGraphVis] Failed to draw obstacle polygon row: {e}")
        # Old format with only positions
        elif {"pos_x", "pos_y"}.issubset(columns):
            xy = sub[["pos_x", "pos_y"]].to_numpy()
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=40,
                marker="s",
                facecolor=self.colors["obstacle"],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.6,
                label="Obstacles",
                zorder=4,
            )

    # ------------------------------------------------------------------
    # Disturbances (for global direction arrow)
    # ------------------------------------------------------------------

    def _get_disturbance_df(self, run_root: pathlib.Path) -> Optional[pd.DataFrame]:
        """Load disturbances.csv once per run_root."""
        if run_root in self.disturbance_logs_cache:
            return self.disturbance_logs_cache[run_root]

        csv_path = run_root / "disturbances.csv"
        if not csv_path.exists():
            print(f"[FullGraphVis] No disturbances.csv found at {csv_path}")
            self.disturbance_logs_cache[run_root] = None
            return None

        try:
            df = pd.read_csv(csv_path)
            self.disturbance_logs_cache[run_root] = df
            return df
        except Exception as e:
            print(f"[FullGraphVis] Failed to read {csv_path}: {e}")
            self.disturbance_logs_cache[run_root] = None
            return None

    def _get_episode_disturbance_vector(
            self,
            run_root: pathlib.Path,
            episode: int,
            step: int,
    ) -> Optional[np.ndarray]:
        """
        Get the disturbance vector for a specific (episode, step) from
        disturbances.csv, using true_dist_x / true_dist_y.

        We average over agents at that step so the arrow is a global disturbance
        indicator for the scene.
        """
        cache_key = (run_root, int(episode), int(step))
        if cache_key in self.disturbance_vector_cache:
            return self.disturbance_vector_cache[cache_key]

        df = self._get_disturbance_df(run_root)
        if df is None:
            self.disturbance_vector_cache[cache_key] = None
            return None

        sub = df[
            (df["episode"] == int(episode)) &
            (df["step"] == int(step))
            ]
        if sub.empty or not {"true_dist_x", "true_dist_y"}.issubset(sub.columns):
            self.disturbance_vector_cache[cache_key] = None
            return None

        # Average over agents for that step (or filter agent_id here if you want just one agent)
        vx = float(sub["true_dist_x"].mean())
        vy = float(sub["true_dist_y"].mean())
        vec = np.array([vx, vy], dtype=float)

        if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-6:
            self.disturbance_vector_cache[cache_key] = None
        else:
            self.disturbance_vector_cache[cache_key] = vec

        return self.disturbance_vector_cache[cache_key]

    def _draw_disturbance_arrow(self, ax, vec: np.ndarray):
        """
        Draw a small disturbance indicator in the top-left:

        - A circle whose radius corresponds to disturbance magnitude 0.8
        - An arrow from the center scaled by ||vec|| / 0.8 (clipped to the circle)
        """
        if vec is None:
            return
        mag = float(np.linalg.norm(vec))
        if not np.isfinite(mag) or mag < 1e-6:
            return

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = xmax - xmin
        dy = ymax - ymin

        # Smaller circle at upper-left
        circle_radius = 0.04 * min(dx, dy)
        cx = xmin + 0.06 * dx  # was 0.14
        cy = ymax - 0.06 * dy  # was 0.14

        # Outer ring representing magnitude = 0.8
        ring = patches.Circle(
            (cx, cy),
            radius=circle_radius,
            facecolor="none",
            edgecolor="k",
            linewidth=1.0,
            alpha=0.9,
            zorder=19,
        )
        ax.add_patch(ring)

        # Normalize disturbance direction
        v = vec / mag

        # Scale arrow length: ||d|| = 0.8 -> arrow reaches circle boundary
        ref_mag = 8
        scale = min(mag / ref_mag, 1.0)
        arrow_len = circle_radius * scale

        ax.arrow(
            cx,
            cy,
            v[0] * arrow_len,
            v[1] * arrow_len,
            # Arrow head now 10% of previous size (0.3 → 0.03, 0.4 → 0.04)
            head_width=0.15 * circle_radius,
            head_length=0.15 * circle_radius,
            fc="k",
            ec="k",
            linewidth=1.2,
            alpha=0.9,
            zorder=20,
            length_includes_head=True,
        )

        # Tiny label under the circle
        ax.text(
            cx,
            cy - 1.1 * circle_radius,
            r"$d$ (ring = 0.8)",
            ha="center",
            va="top",
            fontsize=8,
            color="k",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8),
            zorder=21,
        )

    # ------------------------------------------------------------------
    # Agent step logs + predicted trajectories
    # ------------------------------------------------------------------

    def load_step_logs(
        self, run_root: pathlib.Path, agent_idx: int
    ) -> Optional[pd.DataFrame]:
        cache_key = (run_root, agent_idx)
        if cache_key in self.step_logs_cache:
            return self.step_logs_cache[cache_key]

        step_log_dirs = list(run_root.glob("**/agent_step_logs"))
        if not step_log_dirs:
            print(f"[FullGraphVis] No agent_step_logs directory found in {run_root}")
            self.step_logs_cache[cache_key] = None
            return None

        step_log_dir = step_log_dirs[0]
        csv_path = step_log_dir / f"agent_{agent_idx:02d}.csv"
        if not csv_path.exists():
            print(f"[FullGraphVis] Step log not found: {csv_path}")
            self.step_logs_cache[cache_key] = None
            return None

        try:
            df = pd.read_csv(csv_path)
            self.step_logs_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"[FullGraphVis] Error loading step logs from {csv_path}: {e}")
            self.step_logs_cache[cache_key] = None
            return None

    @staticmethod
    def _is_step_failure(row) -> bool:
        controller = str(row.get("controller_used", "")).upper()
        mpc_status = str(row.get("mpc_status", "")).lower()
        if controller in ("SAFE_QP", "UNKNOWN", ""):
            return True
        if "fail" in mpc_status or "error" in mpc_status:
            return True
        return False

    @staticmethod
    def _parse_control_sequence(control_field) -> Optional[np.ndarray]:
        if control_field is None:
            return None
        if isinstance(control_field, float) and np.isnan(control_field):
            return None

        if isinstance(control_field, (list, tuple, np.ndarray)):
            arr = np.asarray(control_field, dtype=float)
        else:
            s = str(control_field).strip()
            if not s:
                return None
            try:
                arr = np.asarray(json.loads(s), dtype=float)
            except Exception:
                import re

                nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", s)
                if not nums:
                    return None
                arr = np.asarray(nums, dtype=float)

        if arr.ndim == 1:
            if arr.size == 2:
                arr = arr.reshape(1, 2)
            else:
                arr = arr.reshape(-1, 2)

        if arr.shape[-1] != 2:
            return None

        return arr

    def _rollout_positions(
        self,
        initial_state: np.ndarray,
        forces_seq: np.ndarray,
        include_start: bool = False,
    ) -> Optional[np.ndarray]:
        if forces_seq is None:
            return None
        forces_seq = np.asarray(forces_seq, dtype=float)
        if forces_seq.ndim == 1:
            forces_seq = forces_seq.reshape(1, -1)

        if self.max_pred_steps is not None and forces_seq.shape[0] > self.max_pred_steps:
            forces_seq = forces_seq[: self.max_pred_steps]

        x, y, vx, vy = map(float, initial_state[:4])
        dt = float(self.dt)
        m = float(self.mass)

        positions = []
        if include_start:
            positions.append((x, y))

        for fx, fy in forces_seq:
            ax = fx / m
            ay = fy / m
            x = x + vx * dt + 0.5 * ax * dt * dt
            y = y + vy * dt + 0.5 * ay * dt * dt
            vx = vx + ax * dt
            vy = vy + ay * dt
            positions.append((x, y))

        if not positions:
            return None

        return np.asarray(positions, dtype=float)

    def _exaggerate_positions(self, pos_seq: np.ndarray) -> np.ndarray:
        if pos_seq is None or len(pos_seq) == 0:
            return pos_seq
        factor = float(self.exaggeration_factor)
        if factor == 1.0:
            return pos_seq
        deltas = np.diff(pos_seq, axis=0)
        deltas_scaled = deltas * factor
        exaggerated = [pos_seq[0]]
        current = pos_seq[0]
        for d in deltas_scaled:
            current = current + d
            exaggerated.append(current.copy())
        return np.asarray(exaggerated)

    def _plot_predicted_positions_all_agents(
        self,
        ax,
        graph_data: Dict,
        run_root: pathlib.Path,
    ):
        states = np.asarray(graph_data["states"])
        node_types = np.asarray(graph_data["node_type"])
        step_idx = int(graph_data.get("step", 0))

        agent_indices = np.where(node_types == 0)[0]
        if len(agent_indices) == 0:
            return

        for node_idx in agent_indices:
            global_agent_idx = int(node_idx)  # assumes same ordering
            df = self.load_step_logs(run_root, global_agent_idx)
            if df is None or not (0 <= step_idx < len(df)):
                continue

            row = df.iloc[step_idx]
            ctrl_arr = self._parse_control_sequence(row.get("control_input", None))
            if ctrl_arr is None:
                continue

            # If step considered "failure", only use 1-step prediction
            if self._is_step_failure(row):
                ctrl_arr = ctrl_arr[:1]

            # include_start=True so first step gets exaggerated
            pos_seq = self._rollout_positions(
                states[node_idx], ctrl_arr, include_start=True
            )
            if pos_seq is None or len(pos_seq) < 2:
                continue

            pos_seq = self._exaggerate_positions(pos_seq)

            # pos_seq starts at current agent state (exaggerated path)
            line_xy = pos_seq

            ax.plot(
                line_xy[:, 0],
                line_xy[:, 1],
                linestyle=":",
                linewidth=1.0,
                color="black",
                alpha=0.7,
                zorder=15,
            )

            # Scatter only future points (skip index 0 which is the current state)
            future_xy = pos_seq[1:]
            ax.scatter(
                future_xy[:, 0],
                future_xy[:, 1],
                s=10,
                facecolor=self.colors["predicted"],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                zorder=16,
            )

    # ------------------------------------------------------------------
    # Helpers for matching agent marker radius in data units
    # ------------------------------------------------------------------

    def _agent_marker_radius_data(self, ax) -> float:
        """
        Approximate the agent scatter circle radius in *data* units.

        Matplotlib scatter 's' is marker area in points^2.
        We:
          - estimate radius in points from s,
          - convert to pixels,
          - map pixel offset back into data coordinates via the axis transform.
        """
        s = float(self.sizes["agent"])  # points^2

        # Approximate radius in points: area ~ pi r^2
        radius_pt = np.sqrt(s / np.pi)

        # Points -> pixels: 1 pt = dpi / 72 px
        radius_px = radius_pt * (ax.figure.dpi / 72.0)

        # Use transform at some reference data point (0,0)
        center_data = np.array([0.0, 0.0])
        center_disp = ax.transData.transform(center_data)

        offset_disp = center_disp + np.array([radius_px, 0.0])
        offset_data = ax.transData.inverted().transform(offset_disp)

        radius_data = float(offset_data[0] - center_data[0])
        return radius_data

    def _agent_trail_linewidth_points(self) -> float:
        """
        Approximate a line width in points so that the line visually matches
        the agent scatter marker diameter.

        scatter 's' is marker area in points^2:
            area ~ pi * r^2  =>  r_pt = sqrt(s/pi), diameter = 2*r_pt
        """
        s = float(self.sizes["agent"])  # points^2
        radius_pt = np.sqrt(s / np.pi)
        return 2.0 * radius_pt

    # ------------------------------------------------------------------
    # Agent trails (past trajectory) as continuous thick lines with
    # rounded ends and smooth opacity along time.
    # ------------------------------------------------------------------

    def _plot_agent_trails(
            self,
            ax,
            agent_trail_history: Optional[Dict[int, List[Tuple[int, float, float]]]],
            current_step: int,
            node_groups: Dict[str, np.ndarray],
    ):
        """
        Draw fading trails for each agent as continuous thick lines whose
        width visually matches the agent markers, with round caps.

        agent_trail_history[agent_idx] = list of (step, x, y).

        Opacity schedule per segment (using the newer endpoint's step):
          age = current_step - step_new

          - if age <= max_trail_steps:
                opacity: 0.9 (age=0, current) -> 0.2 (age=max_trail_steps)
          - if age > max_trail_steps:
                opacity = 0.2  (older parts stay faint, don't accumulate)

        To avoid color stacking on slow motion:
          - we draw each segment as its own LineCollection with a slightly
            larger zorder the newer it is, so the newest segment covers
            older ones instead of blending with them.
        """
        if agent_trail_history is None:
            return

        max_steps = self.max_trail_steps
        opacity_newest = 0.5  # 90% opaque at current step
        opacity_old = 0.1  # 20% opaque for 50 steps ago and earlier

        trail_width = self._agent_trail_linewidth_points()
        min_dist = 1e-4  # skip essentially zero-length segments

        for agent_idx in node_groups["agents"]:
            history = agent_trail_history.get(int(agent_idx), None)
            if not history or len(history) < 2:
                continue

            # Only positions up to current_step
            hist_sorted = sorted(
                [h for h in history if h[0] <= current_step],
                key=lambda x: x[0],
            )
            if len(hist_sorted) < 2:
                continue

            base_color = self.trail_cmap(int(agent_idx) % self.trail_cmap.N)

            # Draw segments oldest -> newest, with increasing zorder
            for i in range(len(hist_sorted) - 1):
                step_k, xk, yk = hist_sorted[i]
                step_k1, xk1, yk1 = hist_sorted[i + 1]

                # Skip degenerate segments
                if (xk - xk1) ** 2 + (yk - yk1) ** 2 < min_dist ** 2:
                    continue

                age = current_step - step_k1
                if age <= max_steps:
                    t = max(0.0, min(1.0, age / max_steps))
                    opacity = opacity_newest - (opacity_newest - opacity_old) * t
                else:
                    opacity = opacity_old

                rgba = (base_color[0], base_color[1], base_color[2], opacity)

                segments = [[[xk, yk], [xk1, yk1]]]

                # zorder increases with time so newer segments cover older ones
                # (scaling constant just needs to be small but positive)
                z = 4.5 + 0.0001 * step_k1

                lc = LineCollection(
                    segments,
                    colors=[rgba],
                    linewidths=trail_width,
                    capstyle="round",
                    joinstyle="round",
                    antialiased=True,
                    zorder=z,
                )
                ax.add_collection(lc)

    # ------------------------------------------------------------------
    # Plot a single full graph
    # ------------------------------------------------------------------

    def plot_graph(
        self,
        graph_data: Dict,
        save_path: Optional[pathlib.Path] = None,
        show: bool = False,
        title: Optional[str] = None,
        override_axis_mode: Optional[str] = None,
        override_axis_limits: Optional[Tuple[float, float, float, float]] = None,
        run_root: Optional[pathlib.Path] = None,
        agent_trail_history: Optional[Dict[int, List[Tuple[int, float, float]]]] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        states = np.asarray(graph_data["states"])
        node_types = np.asarray(graph_data["node_type"])

        episode = int(graph_data.get("episode", 0))
        step = int(graph_data.get("step", 0))

        node_groups = self.extract_node_groups(graph_data)
        edges = self.extract_edges(graph_data)

        # Edges
        if edges["agent_agent"]:
            lines = [[states[s, :2], states[r, :2]] for s, r in edges["agent_agent"]]
            lc = LineCollection(
                lines,
                colors=self.colors["edge_agent"],
                linewidths=1.5,
                alpha=0.8,
                label="Agent-Agent",
            )
            ax.add_collection(lc)

        if edges["agent_goal"]:
            lines = [[states[s, :2], states[r, :2]] for s, r in edges["agent_goal"]]
            lc = LineCollection(
                lines,
                colors=self.colors["edge_goal"],
                linewidths=2.0,
                alpha=0.6,
                linestyles="dashed",
                label="Agent-Goal",
            )
            ax.add_collection(lc)

        if edges["agent_lidar"]:
            lines = [[states[s, :2], states[r, :2]] for s, r in edges["agent_lidar"]]
            lc = LineCollection(
                lines,
                colors=self.colors["edge_lidar"],
                linewidths=1.2,
                alpha=0.5,
                label="Agent-LiDAR",
            )
            ax.add_collection(lc)

        # Nodes
        if len(node_groups["agents"]) > 0:
            agent_pos = states[node_groups["agents"], :2]
            agent_vel = states[node_groups["agents"], 2:4]
            ax.scatter(
                agent_pos[:, 0],
                agent_pos[:, 1],
                c=self.colors["agent"],
                marker=self.markers["agent"],
                s=self.sizes["agent"],
                edgecolors="black",
                linewidths=0.3,
                label="Agents",
                zorder=10,
            )

            if self.vel_arrow_scale is not None and self.vel_arrow_scale > 0:
                for pos, vel in zip(agent_pos, agent_vel):
                    ax.arrow(
                        pos[0],
                        pos[1],
                        vel[0] * self.vel_arrow_scale,
                        vel[1] * self.vel_arrow_scale,
                        head_width=0.04,
                        head_length=0.025,
                        fc=self.colors["agent"],
                        ec="black",
                        linewidth=1.0,
                        alpha=0.7,
                        zorder=9,
                    )

        if len(node_groups["goals"]) > 0:
            goal_pos = states[node_groups["goals"], :2]
            ax.scatter(
                goal_pos[:, 0],
                goal_pos[:, 1],
                c=self.colors["goal"],
                marker=self.markers["goal"],
                s=self.sizes["goal"],
                edgecolors="black",
                linewidths=1.5,
                label="Goals",
                zorder=7,
            )

        if len(node_groups["lidar"]) > 0:
            lidar_indices = node_groups["lidar"]

            # Collect lidar nodes that actually participate in an agent–LiDAR edge
            connected_lidar = set()
            for s, r in edges["agent_lidar"]:
                if node_types[s] == 2:
                    connected_lidar.add(s)
                if node_types[r] == 2:
                    connected_lidar.add(r)

            # Filter to only those lidar nodes; if none, skip plotting
            if connected_lidar:
                mask = np.array(
                    [idx in connected_lidar for idx in lidar_indices], dtype=bool
                )
                lidar_indices = lidar_indices[mask]

            if len(lidar_indices) > 0:
                lidar_pos = states[lidar_indices, :2]
                ax.scatter(
                    lidar_pos[:, 0],
                    lidar_pos[:, 1],
                    c=self.colors["lidar"],
                    marker=self.markers["lidar"],
                    s=self.sizes["lidar"],
                    alpha=0.6,
                    label="LiDAR hits",
                    zorder=6,
                )


        if len(node_groups["padding"]) > 0:
            padding_pos = states[node_groups["padding"], :2]
            ax.scatter(
                padding_pos[:, 0],
                padding_pos[:, 1],
                c=self.colors["padding"],
                marker=".",
                s=40,
                alpha=0.3,
                label="Padding",
                zorder=1,
            )

        # Obstacles
        if run_root is not None:
            try:
                self._plot_obstacles_for_step(
                    ax, run_root, episode=episode, step=step, graph_data=graph_data
                )
            except Exception as e:
                print(f"[FullGraphVis] Failed to plot obstacles: {e}")

        # Predicted trajectories
        if run_root is not None:
            try:
                self._plot_predicted_positions_all_agents(ax, graph_data, run_root)
            except Exception as e:
                print(f"[FullGraphVis] Failed to plot predicted trajectories: {e}")

        # Axis handling
        old_mode, old_limits = self.axis_mode, self.axis_limits
        if override_axis_mode is not None:
            self.axis_mode = override_axis_mode
        if override_axis_limits is not None:
            self.axis_limits = override_axis_limits

        self._set_axis_limits(ax, states, node_types)

        self.axis_mode, self.axis_limits = old_mode, old_limits

        # Trails (now that axis limits are set, we can match marker screen size)
        self._plot_agent_trails(
            ax,
            agent_trail_history=agent_trail_history,
            current_step=step,
            node_groups=node_groups,
        )

        # Disturbance direction indicator (top-left arrow)
        if run_root is not None:
            try:
                dist_vec = self._get_episode_disturbance_vector(run_root, episode, step)
                if dist_vec is not None:
                    self._draw_disturbance_arrow(ax, dist_vec)
            except Exception as e:
                print(f"[FullGraphVis] Failed to draw disturbance arrow: {e}")

        # Formatting
        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        if title is not None:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            ax.set_title(
                f"Full Graph – Episode {episode}, Step {step}",
                fontsize=14,
                fontweight="bold",
            )

        # Info box
        info_text = f"Nodes: {len(states)} total\n"
        info_text += f"  Agents: {len(node_groups['agents'])}\n"
        info_text += f"  Goals: {len(node_groups['goals'])}\n"
        info_text += f"  LiDAR: {len(node_groups['lidar'])}\n"
        info_text += f"  Padding: {len(node_groups['padding'])}\n"
        info_text += f"Edges: {sum(len(v) for v in edges.values())}"

        ax.text(
            0.98,
            0.50,
            info_text,
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
            family="monospace",
        )

        plt.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"[FullGraphVis] Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    # ------------------------------------------------------------------
    # Episode-level driver
    # ------------------------------------------------------------------

    def visualize_episode(
        self,
        full_graph_root: pathlib.Path,
        episode: int,
        output_dir: Optional[pathlib.Path] = None,
        step_range: Optional[Tuple[int, int]] = None,
        create_animation: bool = False,
        fps: int = 30,
    ):

        """
        visualizes all steps for an episode, accumulating per-agent history
        to draw trails.
        """
        ep_dir = full_graph_root / f"ep{episode:02d}"
        if not ep_dir.exists():
            print(f"[FullGraphVis] No episode directory found: {ep_dir}")
            return

        step_files = sorted(ep_dir.glob("step*.pkl"))
        if not step_files:
            print(f"[FullGraphVis] No step*.pkl files in {ep_dir}")
            return

        if step_range is not None:
            start, end = step_range
            step_files = [
                f for f in step_files if start <= int(f.stem.replace("step", "")) <= end
            ]
            print(
                f"[FullGraphVis] Filtered to {len(step_files)} steps in "
                f"[{start}, {end}] for ep{episode:02d}"
            )

        if not step_files:
            print("[FullGraphVis] No steps left after filtering.")
            return

        if output_dir is None:
            output_dir = full_graph_root / "visualizations" / f"ep{episode:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_root = full_graph_root.parent

        # History: agent_idx -> list of (step, x, y)
        agent_trail_history: Dict[int, List[Tuple[int, float, float]]] = {}

        image_paths = []
        for step_file in step_files:
            graph_data = self.load_graph(step_file)
            if graph_data is None:
                continue

            # Step index for history (prefer stored 'step', fallback to filename)
            try:
                step_num = int(graph_data.get("step", int(step_file.stem.replace("step", ""))))
            except Exception:
                step_num = int(step_file.stem.replace("step", ""))

            states = np.asarray(graph_data["states"])
            node_types = np.asarray(graph_data["node_type"])
            agent_indices = np.where(node_types == 0)[0]

            # Update history for this step
            for a_idx in agent_indices:
                pos = states[a_idx, :2]
                agent_trail_history.setdefault(int(a_idx), []).append(
                    (step_num, float(pos[0]), float(pos[1]))
                )

            out_path = output_dir / f"step{step_num:05d}.png"
            title = f"Full Graph – Ep {episode}, Step {step_num}"

            self.plot_graph(
                graph_data,
                save_path=out_path,
                title=title,
                run_root=run_root,
                agent_trail_history=agent_trail_history,
            )
            image_paths.append(out_path)

        print(f"[FullGraphVis] Created {len(image_paths)} visualizations in {output_dir}")

        if create_animation and image_paths:
            # GIF
            try:
                from PIL import Image

                gif_path = output_dir / f"episode{episode:02d}_full_graph.gif"
                images = [Image.open(p) for p in image_paths]
                # duration is milliseconds per frame
                duration_ms = int(1000 / max(fps, 1))

                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration_ms,
                    loop=0,
                )
                print(f"[FullGraphVis] Created GIF: {gif_path}")
            except Exception as e:
                print(f"[FullGraphVis] GIF creation failed: {e}")

            # MP4
            try:
                import imageio.v2 as imageio

                mp4_path = output_dir / f"episode{episode:02d}_full_graph.mp4"
                with imageio.get_writer(mp4_path, fps=max(fps, 1)) as writer:
                    for p in image_paths:
                        frame = imageio.imread(p)
                        writer.append_data(frame)
                print(f"[FullGraphVis] Created MP4: {mp4_path}")
            except Exception as e:
                print(f"[FullGraphVis] MP4 creation failed: {e} "
                      "(install imageio and imageio-ffmpeg if needed)")



def main():
    parser = argparse.ArgumentParser(
        description="Visualize full graphs (all agents) with trajectories, obstacles, and trails"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to full_graph_logs directory (the one containing epXX/)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to visualize (default: 0 -> ep00)",
    )
    parser.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of steps to visualize (inclusive)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for visualizations",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animated GIF of the episode",
    )
    parser.add_argument(
        "--axis-mode",
        type=str,
        default="auto",
        choices=["auto", "equal", "tight", "fixed"],
        help="Axis scaling mode",
    )
    parser.add_argument(
        "--axis-limits",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Fixed axis limits if axis-mode=fixed",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.5,
        help="Padding around data in auto/equal modes",
    )
    parser.add_argument(
        "--figure-size",
        type=int,
        nargs=2,
        default=[18, 15],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure resolution (dots per inch)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for GIF/MP4 animations (default: 5)",
    )


    args = parser.parse_args()

    full_graph_root = pathlib.Path(args.log_dir)
    if not full_graph_root.exists():
        print(f"[FullGraphVis] Error: {full_graph_root} does not exist")
        return

    vis = FullGraphVisualizer(
        figure_size=tuple(args.figure_size),
        dpi=args.dpi,
        axis_mode=args.axis_mode,
        axis_limits=tuple(args.axis_limits) if args.axis_limits else None,
        padding=args.padding,
    )

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    step_range = tuple(args.step_range) if args.step_range else None

    vis.visualize_episode(
        full_graph_root=full_graph_root,
        episode=args.episode,
        output_dir=output_dir,
        step_range=step_range,
        create_animation=args.animate,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

