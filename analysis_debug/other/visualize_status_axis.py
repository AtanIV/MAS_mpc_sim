#!/usr/bin/env python3
"""
Generate bar graph visualization from SQP-GCBF agent step logs.
Creates a bar chart showing controller iterations for each step with color-coded status.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy as np
from typing import Optional, List, Tuple


class StatusBarGraphVisualizer:
    """Create bar graph visualization from agent step logs."""

    def __init__(self, figure_width: int = 16, figure_height: int = 6, dpi: int = 100):
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.dpi = dpi

        # Status colors - swapped SAFE_QP and POSITIVE_DERIVATIVE
        self.status_colors = {
            'SQP': '#00CC00',  # Green - SQP succeeded
            'POSITIVE_DERIVATIVE': '#FF0000',  # Red - Positive directional derivative
            'SAFE_QP': '#FF9900'  # Orange - Safe QP fallback (limit reached)
        }

    def load_step_log(self, csv_path: pathlib.Path) -> Optional[pd.DataFrame]:
        """Load agent step log from CSV file."""
        if not csv_path.exists():
            print(f"Error: Step log not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Error loading step log from {csv_path}: {e}")
            return None

    def categorize_status(self, row: pd.Series) -> Tuple[str, str]:
        """
        Categorize the controller status for a step.

        Args:
            row: DataFrame row for a single step

        Returns:
            Tuple of (status_key, status_label)
        """
        controller = row.get('controller_used', 'UNKNOWN')
        sqp_status = str(row.get('sqp_status', ''))

        if controller == 'SQP':
            return 'SQP', 'SQP Success'
        elif controller == 'SAFE_QP':
            if 'Positive directional derivative' in sqp_status or 'positive' in sqp_status.lower():
                return 'POSITIVE_DERIVATIVE', 'Positive Gradient'
            else:
                return 'SAFE_QP', 'Limit Reached'
        else:
            return 'SAFE_QP', 'Unknown'

    def create_bar_graph(self,
                         df: pd.DataFrame,
                         save_path: Optional[pathlib.Path] = None,
                         show: bool = False,
                         title: Optional[str] = None,
                         step_range: Optional[Tuple[int, int]] = None,
                         bar_width: float = 0.8) -> plt.Figure:
        """
        Create bar graph visualization showing iterations per step.

        Args:
            df: DataFrame with step log data
            save_path: Where to save figure
            show: Whether to display interactively
            title: Custom title for plot
            step_range: Optional (start, end) range of steps to visualize
            bar_width: Width of each bar (default: 0.8)
        """
        # Filter by step range if provided
        if step_range is not None:
            start, end = step_range
            df = df.iloc[start:end + 1]

        n_steps = len(df)

        if n_steps == 0:
            print("No steps to visualize")
            return None

        # Prepare data for plotting
        steps = []
        iterations = []
        colors = []
        status_labels = []

        for idx, (_, row) in enumerate(df.iterrows()):
            step_num = row.get('step', idx)
            status_key, status_label = self.categorize_status(row)
            color = self.status_colors.get(status_key, '#888888')

            # Get iteration count - use different heights for failed optimizations
            if status_key == 'SQP':
                iter_count = row.get('sqp_iterations', 0)
                if pd.isna(iter_count):
                    iter_count = 0
            elif status_key == 'SAFE_QP':
                # Limit reached: height = 100
                iter_count = 100
            else:
                # Positive gradient: height = 1
                iter_count = 50

            steps.append(step_num)
            iterations.append(int(iter_count))
            colors.append(color)
            status_labels.append(status_key)

        # Create figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height), dpi=self.dpi)

        # Create bar chart without black edges
        bars = ax.bar(steps, iterations, width=bar_width, color=colors,
                      edgecolor='none', alpha=0.85)

        # Customize axes
        ax.set_xlabel('Step', fontsize=16, fontweight='bold')
        ax.set_ylabel('Iterations', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis limits with some padding
        ax.set_xlim(min(steps) - 1, max(steps) + 1)

        # Set y-axis to start from 0
        ax.set_ylim(0, max(iterations) * 1.1 if max(iterations) > 0 else 10)

        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Create legend - swapped order to match color swap
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.status_colors['SQP'], edgecolor='none',
                  label='SQP Success', alpha=0.85),
            Patch(facecolor=self.status_colors['POSITIVE_DERIVATIVE'], edgecolor='none',
                  label='Positive Gradient', alpha=0.85),
            Patch(facecolor=self.status_colors['SAFE_QP'], edgecolor='none',
                  label='Limit Reached', alpha=0.85)
        ]
        # Place legend on top center
        ax.legend(handles=legend_elements, loc='upper center', fontsize=12,
                  framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, 1.15))

        # Calculate statistics
        status_counts = {}
        total_iterations = 0
        for idx, status_key in enumerate(status_labels):
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            if status_key == 'SQP':
                total_iterations += iterations[idx]

        sqp_count = status_counts.get('SQP', 0)
        pos_grad_count = status_counts.get('POSITIVE_DERIVATIVE', 0)
        limit_count = status_counts.get('SAFE_QP', 0)

        avg_iterations = total_iterations / sqp_count if sqp_count > 0 else 0

        stats_text = f"Total Steps: {n_steps}\n"
        stats_text += f"SQP Success: {sqp_count} ({sqp_count / n_steps * 100:.1f}%)\n"
        stats_text += f"Positive Gradient: {pos_grad_count} ({pos_grad_count / n_steps * 100:.1f}%)\n"
        stats_text += f"Limit Reached: {limit_count} ({limit_count / n_steps * 100:.1f}%)\n"
        stats_text += f"Avg Iterations (SQP): {avg_iterations:.2f}"

        # Position stats box at upper left
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                fontsize=11,
                family='monospace')

        # Title
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        else:
            # Get agent_idx from first row if available
            if len(df) > 0 and 'agent_idx' in df.columns:
                agent_idx = df.iloc[0]['agent_idx']
            else:
                agent_idx = 'Unknown'

            ax.set_title(f'Controller Iterations per Step - Agent {agent_idx}',
                         fontsize=18, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_multi_agent_graph(self,
                                 agent_dfs: List[Tuple[int, pd.DataFrame]],
                                 save_path: Optional[pathlib.Path] = None,
                                 show: bool = False,
                                 title: Optional[str] = None,
                                 step_range: Optional[Tuple[int, int]] = None,
                                 bar_width: float = 0.8,
                                 subplot_height: float = 4) -> plt.Figure:
        """
        Create multi-agent bar graph visualization with agents stacked vertically.

        Args:
            agent_dfs: List of tuples (agent_idx, dataframe) for each agent
            save_path: Where to save figure
            show: Whether to display interactively
            title: Custom title for plot
            step_range: Optional (start, end) range of steps to visualize
            bar_width: Width of each bar (default: 0.8)
            subplot_height: Height of each subplot in inches
        """
        n_agents = len(agent_dfs)
        if n_agents == 0:
            print("No agents to visualize")
            return None

        # Create figure with subplots
        fig_height = subplot_height * n_agents
        fig, axes = plt.subplots(n_agents, 1, figsize=(self.figure_width, fig_height),
                                 dpi=self.dpi, sharex=True)

        # Handle single agent case
        if n_agents == 1:
            axes = [axes]

        # Plot each agent
        for idx, (agent_idx, df) in enumerate(agent_dfs):
            ax = axes[idx]

            # Filter by step range if provided
            if step_range is not None:
                start, end = step_range
                df = df.iloc[start:end + 1]

            n_steps = len(df)
            if n_steps == 0:
                continue

            # Prepare data for plotting
            steps = []
            iterations = []
            colors = []
            status_labels = []

            for _, row in df.iterrows():
                step_num = row.get('step', len(steps))
                status_key, status_label = self.categorize_status(row)
                color = self.status_colors.get(status_key, '#888888')

                # Get iteration count
                if status_key == 'SQP':
                    iter_count = row.get('sqp_iterations', 0)
                    if pd.isna(iter_count):
                        iter_count = 0
                elif status_key == 'SAFE_QP':
                    iter_count = 100
                else:
                    iter_count = 50

                steps.append(step_num)
                iterations.append(int(iter_count))
                colors.append(color)
                status_labels.append(status_key)

            # Create bar chart
            ax.bar(steps, iterations, width=bar_width, color=colors,
                   edgecolor='none', alpha=0.85)

            # Customize axes
            ax.set_ylabel('Iterations', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=11)

            # Set x-axis limits with some padding
            ax.set_xlim(min(steps) - 1, max(steps) + 1)

            # Set y-axis to start from 0
            ax.set_ylim(0, max(iterations) * 1.3 if max(iterations) > 0 else 10)

            # Add grid for better readability
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

            # Add agent label
            ax.text(0.01, 0.95, f'Agent {agent_idx}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    fontsize=14,
                    fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # Calculate statistics for this agent
            status_counts = {}
            total_iterations = 0
            for i, status_key in enumerate(status_labels):
                status_counts[status_key] = status_counts.get(status_key, 0) + 1
                if status_key == 'SQP':
                    total_iterations += iterations[i]

            sqp_count = status_counts.get('SQP', 0)
            pos_grad_count = status_counts.get('POSITIVE_DERIVATIVE', 0)
            limit_count = status_counts.get('SAFE_QP', 0)
            avg_iterations = total_iterations / sqp_count if sqp_count > 0 else 0

            # Add statistics as text (smaller for multi-agent view)
            stats_text = f"Steps: {n_steps} | SQP: {sqp_count} ({sqp_count / n_steps * 100:.0f}%) | "
            stats_text += f"Pos Grad: {pos_grad_count} | Limit: {limit_count} | Avg Iter: {avg_iterations:.1f}"

            ax.text(0.99, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')

        # Add x-label only to bottom subplot
        axes[-1].set_xlabel('Step', fontsize=16, fontweight='bold')

        # Create legend on the top - placed above the first subplot, swapped order
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.status_colors['SQP'], edgecolor='none',
                  label='SQP Success', alpha=0.85),
            Patch(facecolor=self.status_colors['POSITIVE_DERIVATIVE'], edgecolor='none',
                  label='Positive Gradient', alpha=0.85),
            Patch(facecolor=self.status_colors['SAFE_QP'], edgecolor='none',
                  label='Limit Reached', alpha=0.85)
        ]
        axes[0].legend(handles=legend_elements, loc='upper center', fontsize=11,
                       framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, 1.25))

        # Overall title
        if title:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=0.998)
        else:
            fig.suptitle(f'Controller Iterations per Step - Multiple Agents',
                         fontsize=20, fontweight='bold', y=0.998)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_all_agents(self,
                             log_dir: pathlib.Path,
                             output_dir: Optional[pathlib.Path] = None,
                             agent_indices: Optional[List[int]] = None,
                             step_range: Optional[Tuple[int, int]] = None,
                             combined: bool = False):
        """
        Create bar graph visualizations for multiple agents.

        Args:
            log_dir: Root directory containing agent_step_logs
            output_dir: Where to save visualizations
            agent_indices: List of agent indices to visualize (None = all)
            step_range: Optional step range to visualize
            combined: If True, create a single image with all agents stacked vertically
        """
        # Find agent_step_logs directory
        step_log_dirs = list(log_dir.glob("**/agent_step_logs"))

        if not step_log_dirs:
            print(f"Error: No agent_step_logs directory found in {log_dir}")
            return

        step_log_dir = step_log_dirs[0]
        print(f"Found step logs in: {step_log_dir}")

        # Find all agent CSV files
        agent_files = sorted(step_log_dir.glob("agent_*.csv"))

        if not agent_files:
            print(f"No agent CSV files found in {step_log_dir}")
            return

        # Filter by agent indices if provided
        if agent_indices is not None:
            agent_files = [f for f in agent_files
                           if int(f.stem.split('_')[1]) in agent_indices]

        if output_dir is None:
            output_dir = log_dir / "status_bar_graphs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if combined:
            # Load all agent data
            agent_dfs = []
            for agent_file in agent_files:
                agent_idx = int(agent_file.stem.split('_')[1])
                df = self.load_step_log(agent_file)
                if df is None:
                    continue

                # Add agent_idx to dataframe if not present
                if 'agent_idx' not in df.columns:
                    df['agent_idx'] = agent_idx

                agent_dfs.append((agent_idx, df))

            if agent_dfs:
                output_path = output_dir / "all_agents_combined_bar_graph.png"
                title = f"Controller Iterations per Step - All Agents"
                self.create_multi_agent_graph(agent_dfs, save_path=output_path,
                                              title=title, step_range=step_range)
                print(f"Created combined visualization with {len(agent_dfs)} agents")
        else:
            # Create individual visualizations
            print(f"Creating bar graphs for {len(agent_files)} agents...")

            for agent_file in agent_files:
                agent_idx = int(agent_file.stem.split('_')[1])

                df = self.load_step_log(agent_file)
                if df is None:
                    continue

                # Add agent_idx to dataframe if not present
                if 'agent_idx' not in df.columns:
                    df['agent_idx'] = agent_idx

                output_path = output_dir / f"agent_{agent_idx:02d}_bar_graph.png"
                title = f"Controller Iterations per Step - Agent {agent_idx}"

                self.create_bar_graph(df, save_path=output_path, title=title,
                                      step_range=step_range)

            print(f"Created bar graph visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate controller bar graph visualizations from agent step logs'
    )
    parser.add_argument('log_dir', type=str,
                        help='Path to directory containing agent_step_logs')
    parser.add_argument('--agent', type=int,
                        help='Specific agent index to visualize')
    parser.add_argument('--agents', type=int, nargs='+',
                        help='Multiple agent indices to visualize')
    parser.add_argument('--output-dir', type=str,
                        help='Custom output directory for visualizations')
    parser.add_argument('--step-range', type=int, nargs=2, metavar=('START', 'END'),
                        help='Range of steps to visualize')
    parser.add_argument('--show', action='store_true',
                        help='Display visualization interactively')
    parser.add_argument('--combined', action='store_true',
                        help='Create a single image with all agents stacked vertically')
    parser.add_argument('--figure-width', type=int, default=16,
                        help='Figure width in inches')
    parser.add_argument('--figure-height', type=int, default=6,
                        help='Figure height in inches (per agent if combined)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Figure resolution')
    parser.add_argument('--bar-width', type=float, default=0.8,
                        help='Width of each bar')
    parser.add_argument('--subplot-height', type=float, default=4,
                        help='Height of each subplot in combined mode')

    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        return

    visualizer = StatusBarGraphVisualizer(
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi
    )

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

    # Determine which agents to visualize
    if args.agent is not None:
        agent_indices = [args.agent]
    elif args.agents is not None:
        agent_indices = args.agents
    else:
        agent_indices = None  # All agents

    step_range = tuple(args.step_range) if args.step_range else None

    # Single agent visualization with show option
    if args.agent is not None and args.show and not args.combined:
        step_log_dirs = list(log_dir.glob("**/agent_step_logs"))
        if not step_log_dirs:
            print(f"Error: No agent_step_logs directory found in {log_dir}")
            return

        csv_path = step_log_dirs[0] / f"agent_{args.agent:02d}.csv"
        df = visualizer.load_step_log(csv_path)

        if df is None:
            return

        # Add agent_idx to dataframe if not present
        if 'agent_idx' not in df.columns:
            df['agent_idx'] = args.agent

        title = f"Controller Iterations per Step - Agent {args.agent}"

        visualizer.create_bar_graph(df, show=True, title=title,
                                    step_range=step_range,
                                    bar_width=args.bar_width)
    # Combined multi-agent visualization with show option
    elif args.show and (args.combined or args.agents):
        step_log_dirs = list(log_dir.glob("**/agent_step_logs"))
        if not step_log_dirs:
            print(f"Error: No agent_step_logs directory found in {log_dir}")
            return

        step_log_dir = step_log_dirs[0]
        agent_files = sorted(step_log_dir.glob("agent_*.csv"))

        # Filter by agent indices if provided
        if agent_indices is not None:
            agent_files = [f for f in agent_files
                           if int(f.stem.split('_')[1]) in agent_indices]

        # Load all agent data
        agent_dfs = []
        for agent_file in agent_files:
            agent_idx = int(agent_file.stem.split('_')[1])
            df = visualizer.load_step_log(agent_file)
            if df is None:
                continue

            if 'agent_idx' not in df.columns:
                df['agent_idx'] = agent_idx

            agent_dfs.append((agent_idx, df))

        if agent_dfs:
            title = f"Controller Iterations per Step - Multiple Agents"
            visualizer.create_multi_agent_graph(agent_dfs, show=True, title=title,
                                                step_range=step_range,
                                                bar_width=args.bar_width,
                                                subplot_height=args.subplot_height)
    else:
        # Batch visualization (saves to files)
        visualizer.visualize_all_agents(log_dir, output_dir=output_dir,
                                        agent_indices=agent_indices,
                                        step_range=step_range,
                                        combined=args.combined)


if __name__ == '__main__':
    main()

# Example usage:
#
# # Visualize all agents individually
# python visualize_status_axis.py /path/to/logs
#
# # Visualize all agents in a single combined image
# python visualize_status_axis.py /path/to/logs --combined
#
# # Visualize specific agents in a combined image
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1031-1711 --agents 0 1 2 3 --combined
#
# # Show combined visualization interactively
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1031-1711 --agents 0 1 2 3 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1028-0148 --agents 0 1 2 3 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1104-1509 --agents 0 1 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1104-1610 --agents 0 1 2 3 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy/1104-1742 --agents 0 1 2 3 --step-range 0 100 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1105-0123 --agents 0 1 2 3 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1109-2237 --agents 0 1 2 3 --combined --show

# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1114-0203 --agents 0 1 2 3 --step-range 0 150 --combined --show

# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy/1111-0204 --agents 0 1 2 3 4 5 --combined --show
# python visualize_status_axis.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy/1118-0349 --agents 0 1 2 3 4 5 --combined --show
#


#
# # Single agent with interactive display
# python visualize_status_axis.py /path/to/logs --agent 0 --show
#
# # Specific step range with combined view
# python visualize_status_axis.py /path/to/logs --agents 0 1 2 --step-range 0 159 --combined --show
#
# # Adjust subplot height for combined view
# python visualize_status_axis.py /path/to/logs --combined --subplot-height 3
