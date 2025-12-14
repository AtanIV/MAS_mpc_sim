#!/usr/bin/env python3
"""
Generate status bar visualization from SQP-GCBF agent step logs.
Creates a horizontal timeline showing controller status for each step.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import pathlib
import numpy as np
from typing import Optional, List, Tuple


class StatusBarVisualizer:
    """Create horizontal status bar visualization from agent step logs."""

    def __init__(self, figure_width: int = 16, figure_height: int = 3, dpi: int = 100):
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.dpi = dpi

        # Status colors matching the local graph visualizer
        self.status_colors = {
            'SQP': '#00CC00',  # Green - SQP succeeded
            'SAFE_QP': '#FF0000',  # Red - Safe QP fallback (limit reached)
            'POSITIVE_DERIVATIVE': '#FF9900'  # Orange - Positive directional derivative
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

    def create_status_bar(self,
                          df: pd.DataFrame,
                          save_path: Optional[pathlib.Path] = None,
                          show: bool = False,
                          title: Optional[str] = None,
                          step_range: Optional[Tuple[int, int]] = None,
                          rect_width: float = 1.0,
                          rect_height: float = 0.8,
                          steps_per_row: int = 20) -> plt.Figure:
        """
        Create status bar visualization with multiple rows.

        Args:
            df: DataFrame with step log data
            save_path: Where to save figure
            show: Whether to display interactively
            title: Custom title for plot
            step_range: Optional (start, end) range of steps to visualize
            rect_width: Width of each status rectangle
            rect_height: Height of status rectangles
            steps_per_row: Number of steps to display per row (default: 20)
        """
        # Filter by step range if provided
        if step_range is not None:
            start, end = step_range
            df = df.iloc[start:end + 1]

        n_steps = len(df)

        if n_steps == 0:
            print("No steps to visualize")
            return None

        # Calculate number of rows needed
        n_rows = int(np.ceil(n_steps / steps_per_row))

        # Calculate figure dimensions
        row_width = steps_per_row * rect_width
        calculated_width = max(self.figure_width, row_width * 0.9)
        calculated_height = max(self.figure_height, n_rows * (rect_height + 1.0) + 1.5)

        fig, ax = plt.subplots(figsize=(calculated_width, calculated_height), dpi=self.dpi)

        # Process each step
        for idx, (_, row) in enumerate(df.iterrows()):
            step_num = row.get('step', idx)
            status_key, status_label = self.categorize_status(row)
            color = self.status_colors.get(status_key, '#888888')

            # Calculate row and column position
            row_idx = idx // steps_per_row
            col_idx = idx % steps_per_row

            # Draw rectangle (y increases downward, so invert row_idx)
            x_pos = col_idx * rect_width
            y_pos = (n_rows - 1 - row_idx) * (rect_height + 1.0)

            rect = patches.Rectangle((x_pos, y_pos), rect_width, rect_height,
                                     linewidth=1.5, edgecolor='black',
                                     facecolor=color, alpha=0.8)
            ax.add_patch(rect)

            # Add iteration count on top for successful SQP
            if status_key == 'SQP':
                iterations = row.get('sqp_iterations', None)
                if iterations is not None and not pd.isna(iterations):
                    ax.text(x_pos + rect_width / 2, y_pos + rect_height + 0.1,
                            f"{int(iterations)}",
                            ha='center', va='bottom',
                            fontsize=22,  # Increased from 8
                            fontweight='bold',
                            color='black')

            # Add step number below rectangle
            ax.text(x_pos + rect_width / 2, y_pos - 0.15,
                    f"{int(step_num)}",
                    ha='center', va='top',
                    fontsize=20,  # Increased from 7
                    color='black')

        # Draw arrow axis for each row
        for row_idx in range(n_rows):
            y_pos = (n_rows - 1 - row_idx) * (rect_height + 1.0)
            arrow_y = y_pos - 0.35

            # Determine arrow length for this row
            steps_in_row = min(steps_per_row, n_steps - row_idx * steps_per_row)
            arrow_end = steps_in_row * rect_width

            arrow_props = dict(arrowstyle='->', lw=2, color='black')
            ax.annotate('', xy=(arrow_end, arrow_y), xytext=(0, arrow_y),
                        arrowprops=arrow_props)

            # Add "Steps" label only on the bottom row
            if row_idx == n_rows - 1:
                ax.text(arrow_end / 2, arrow_y - 0.15,
                        'Steps',
                        ha='center', va='top',
                        fontsize=20,  # Increased from 10
                        fontweight='bold')

        # Create legend
        legend_elements = [
            patches.Patch(facecolor=self.status_colors['SQP'], edgecolor='black',
                          label='SQP Success'),
            patches.Patch(facecolor=self.status_colors['POSITIVE_DERIVATIVE'], edgecolor='black',
                          label='Positive Gradient'),
            patches.Patch(facecolor=self.status_colors['SAFE_QP'], edgecolor='black',
                          label='Limit Reached')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=24,  # Increased from 9
                  framealpha=0.9)

        # Calculate statistics
        status_counts = {}
        for _, row in df.iterrows():
            status_key, _ = self.categorize_status(row)
            status_counts[status_key] = status_counts.get(status_key, 0) + 1

        sqp_count = status_counts.get('SQP', 0)
        pos_grad_count = status_counts.get('POSITIVE_DERIVATIVE', 0)
        limit_count = status_counts.get('SAFE_QP', 0)

        stats_text = f"Total Steps: {n_steps}\n"
        stats_text += f"SQP Success: {sqp_count} ({sqp_count / n_steps * 100:.1f}%)\n"
        stats_text += f"Positive Gradient: {pos_grad_count} ({pos_grad_count / n_steps * 100:.1f}%)\n"
        stats_text += f"Limit Reached: {limit_count} ({limit_count / n_steps * 100:.1f}%)"

        # Position stats box at upper left (moved higher to avoid covering rectangles)
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=24,  # Increased from 9
                family='monospace')

        # Formatting
        ax.set_xlim(-0.5, row_width + 0.5)
        ax.set_ylim(-0.8, n_rows * (rect_height + 1.0) + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title - Fix agent index extraction
        if title:
            ax.set_title(title, fontsize=40, fontweight='bold', pad=20)  # Increased from 14
        else:
            # Get agent_idx from first row if available, otherwise try to extract from filename
            if len(df) > 0 and 'agent_idx' in df.columns:
                agent_idx = df.iloc[0]['agent_idx']
            else:
                agent_idx = 'Unknown'

            ax.set_title(f'Controller Status Timeline - Agent {agent_idx}',
                         fontsize=18, fontweight='bold', pad=20)  # Increased from 14

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
                             step_range: Optional[Tuple[int, int]] = None):
        """
        Create status bar visualizations for multiple agents.

        Args:
            log_dir: Root directory containing agent_step_logs
            output_dir: Where to save visualizations
            agent_indices: List of agent indices to visualize (None = all)
            step_range: Optional step range to visualize
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
            output_dir = log_dir / "status_bars"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating status bars for {len(agent_files)} agents...")

        for agent_file in agent_files:
            agent_idx = int(agent_file.stem.split('_')[1])

            df = self.load_step_log(agent_file)
            if df is None:
                continue

            # Add agent_idx to dataframe if not present
            if 'agent_idx' not in df.columns:
                df['agent_idx'] = agent_idx

            output_path = output_dir / f"agent_{agent_idx:02d}_status_bar.png"
            title = f"Controller Status Timeline - Agent {agent_idx}"

            self.create_status_bar(df, save_path=output_path, title=title,
                                   step_range=step_range)

        print(f"Created status bar visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate controller status bar visualizations from agent step logs'
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
    parser.add_argument('--figure-width', type=int, default=16,
                        help='Figure width in inches')
    parser.add_argument('--figure-height', type=int, default=3,
                        help='Figure height in inches')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Figure resolution')
    parser.add_argument('--rect-width', type=float, default=1.0,
                        help='Width of each status rectangle')
    parser.add_argument('--rect-height', type=float, default=0.8,
                        help='Height of status rectangles')

    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        return

    visualizer = StatusBarVisualizer(
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
    if args.agent is not None and args.show:
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

        title = f"Controller Status Timeline - Agent {args.agent}"

        # Show without saving when --show is used
        visualizer.create_status_bar(df, show=True, title=title,
                                     step_range=step_range,
                                     rect_width=args.rect_width,
                                     rect_height=args.rect_height)
    else:
        # Batch visualization (saves to files)
        visualizer.visualize_all_agents(log_dir, output_dir=output_dir,
                                        agent_indices=agent_indices,
                                        step_range=step_range)


if __name__ == '__main__':
    main()

# # Visualize all agents
# python visualize_status_bar.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1021-1609
#
# # Single agent with interactive display
# python visualize_status_bar.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1020-1745 --agent 0 --step-range 0 159 --show
#
# # Multiple specific agents
# python visualize_status_bar.py /path/to/logs --agents 0 1 2
#
# # Specific step range
# python visualize_status_bar.py /path/to/logs --agent 0 --step-range 0 50
#
# # Custom output directory
# python visualize_status_bar.py /path/to/logs --output-dir ./my_status_bars
#
# # Adjust rectangle size
# python visualize_status_bar.py /path/to/logs --agent 0 --rect-width 1.2 --rect-height 1.0
