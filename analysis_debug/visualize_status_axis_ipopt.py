#!/usr/bin/env python3
"""
Generate bar graph visualization from IPOPT-based agent step logs.

- One bar per step.
- Bar height = ipopt_iterations (or mpc_iterations fallback).
- Bar color = IPOPT status (from ipopt_status / mpc_status).
"""

import argparse
import pathlib
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StatusBarGraphVisualizer:
    """Create bar graph visualization from agent step logs."""

    def __init__(self,
                 figure_width: int = 16,
                 figure_height: int = 6,
                 dpi: int = 100):
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.dpi = dpi

        # ------------------------------------------------------------------ #
        # Color mapping based on your ipopt_status_map comments
        # Unmarked statuses (and anything unknown) -> grey
        # ------------------------------------------------------------------ #
        self.status_colors = {
            # Marked with specific colors
            "Solve_Succeeded":                 "#2ecc71",  # green
            "Solved_To_Acceptable_Level":      "#2ecc71",  # green
            "fail: Infeasible_Problem_Detected":     "#e74c3c",  # red
            "fail: Search_Direction_Becomes_Too_Small": "#e57373",  # light red
            "fail: User_Requested_Stop":             "#3498db",  # blue
            "fail: Feasible_Point_Found":            "#f1c40f",  # yellow
            "fail: Maximum_Iterations_Exceeded":     "#e67e22",  # orange
            "fail: Restoration_Failed":              "#9b59b6",  # light purple
            "fail: Error_In_Step_Computation":       "#8e5b3a",  # brown
            "fail: Not_Enough_Degrees_Of_Freedom":   "#e74c3c",  # dark purple

            # Everything else (explicit or not) falls back to grey.
            "Unknown":                         "#7f8c8d",  # grey
        }

        # Dark green for *_FeasibleRestored
        self.feasible_restored_color = "#196f3d"

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_step_log(csv_path: pathlib.Path) -> Optional[pd.DataFrame]:
        """Load a single agent_XX.csv log file."""
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist.")
            return None
        df = pd.read_csv(csv_path)
        if "step" not in df.columns:
            print(f"Warning: 'step' column missing in {csv_path}")
        return df

    # ------------------------------------------------------------------ #
    # Status parsing and coloring
    # ------------------------------------------------------------------ #
    def _pick_status_string(self, row: pd.Series) -> str:
        """
        Prefer ipopt_status; fall back to mpc_status if needed.
        """
        s = row.get("ipopt_status", None)
        if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
            s = row.get("mpc_status", "")
        return str(s).strip()

    def _get_status_color_and_label(self, raw_status: str) -> Tuple[str, str]:
        """
        Given the raw status string from the CSV, determine:
        - a color (hex)
        - a human-friendly label for legend/stats

        Rules:
        * If status ends with "_FeasibleRestored" -> dark green.
        * Otherwise, if base name is in status_colors -> use its color.
        * Everything else -> grey.
        """
        if raw_status is None:
            return self.status_colors["Unknown"], "Unknown"

        s = str(raw_status).strip()
        if not s:
            return self.status_colors["Unknown"], "Unknown"

        feasible_restored = False
        base = s

        if s.endswith("_FeasibleRestored"):
            feasible_restored = True
            base = s[: -len("_FeasibleRestored")]

        # Choose color
        if feasible_restored:
            color = self.feasible_restored_color
            label = base.replace("_", " ") + " (FeasibleRestored)"
        else:
            color = self.status_colors.get(base, self.status_colors["Unknown"])
            label = base.replace("_", " ") if base else "Unknown"

        return color, label

    def _is_successish(self, status: str) -> bool:
        """
        Define which statuses count as "success-ish" for stats.
        """
        if not status:
            return False

        if status.endswith("_FeasibleRestored"):
            return True

        base = status
        # main "good" statuses
        if base in {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Feasible_Point_Found",
        }:
            return True

        return False

    def _legend_sort_key(self, status: str):
        """
        Sort legend entries by a preferred order based on base name.
        """
        base = status
        if base.endswith("_FeasibleRestored"):
            base = base[: -len("_FeasibleRestored")]

        order_map = {
            "Solve_Succeeded": 0,
            "Solved_To_Acceptable_Level": 1,
            "Feasible_Point_Found": 2,
            "Infeasible_Problem_Detected": 3,
            "Search_Direction_Becomes_Too_Small": 4,
            "Maximum_Iterations_Exceeded": 5,
            "Restoration_Failed": 6,
            "Error_In_Step_Computation": 7,
            "Not_Enough_Degrees_Of_Freedom": 8,
            "User_Requested_Stop": 9,
        }
        return (order_map.get(base, 100), status)

    # ------------------------------------------------------------------ #
    # Single-agent plot
    # ------------------------------------------------------------------ #
    def create_bar_graph(self,
                         df: pd.DataFrame,
                         save_path: Optional[pathlib.Path] = None,
                         show: bool = False,
                         title: Optional[str] = None,
                         step_range: Optional[Tuple[int, int]] = None,
                         step_range_axis: Optional[Tuple[int, int]] = None,
                         bar_width: float = 0.8) -> plt.Figure:
        """
        Create bar graph visualization showing IPOPT iterations per step.

        - x-axis: step
        - y-axis: ipopt_iterations (or mpc_iterations fallback)
        - color: ipopt_status (normalized)

        step_range: filter data to these steps.
        step_range_axis: force x-axis to this range regardless of data.
        """

        # Optionally restrict to a step range (data)
        if step_range is not None:
            start, end = step_range
            df = df[(df["step"] >= start) & (df["step"] <= end)].copy()

        if df.empty:
            raise ValueError("No data available to plot after applying step_range.")

        # Prepare plotting data
        steps: List[int] = []
        iterations: List[int] = []
        colors: List[str] = []
        status_keys: List[str] = []
        labels_for_legend: List[str] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            step_num = int(row.get("step", idx))

            # status & color
            raw_status = self._pick_status_string(row)
            color, label = self._get_status_color_and_label(raw_status)

            # iterations: prefer ipopt_iterations, fallback mpc_iterations
            iter_val = row.get("ipopt_iterations", None)
            if iter_val is None or (isinstance(iter_val, float) and pd.isna(iter_val)):
                iter_val = row.get("mpc_iterations", 0)
            if pd.isna(iter_val):
                iter_val = 0
            iter_val = int(iter_val)

            steps.append(step_num)
            iterations.append(iter_val)
            colors.append(color)
            status_keys.append(str(raw_status))
            labels_for_legend.append(label)

        # Create figure
        fig, ax = plt.subplots(
            figsize=(self.figure_width, self.figure_height),
            dpi=self.dpi,
        )

        # Bar chart
        ax.bar(steps, iterations, width=bar_width, color=colors,
               edgecolor="none", alpha=0.85)

        # Axes formatting
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel("IPOPT Iterations", fontsize=14)
        ax.set_xticks(steps)
        ax.tick_params(axis="x", labelrotation=0)

        # Force x-axis range if requested
        if step_range_axis is not None:
            x_start, x_end = step_range_axis
            ax.set_xlim(x_start, x_end)

        if title:
            ax.set_title(title, fontsize=16, pad=15)

        # Stats
        n_steps = len(steps)
        unique_statuses, counts = np.unique(status_keys, return_counts=True)
        stats_map = dict(zip(unique_statuses, counts))

        success_steps = sum(
            stats_map[s] for s in stats_map.keys() if self._is_successish(s)
        )
        avg_iter = float(np.mean(iterations)) if iterations else 0.0

        # Build stats text
        stats_lines = [
            f"Total Steps: {n_steps}",
            f"Success-ish: {success_steps} ({(success_steps / n_steps * 100):.1f}%)" if n_steps > 0 else "Success-ish: 0 (0.0%)",
            f"Avg Iterations: {avg_iter:.2f}",
            "",
        ]

        # Sort status keys for stats printing
        for s in sorted(stats_map.keys(), key=self._legend_sort_key):
            _, label = self._get_status_color_and_label(s)
            stats_lines.append(
                f"{label}: {stats_map[s]} ({stats_map[s] / n_steps * 100:.1f}%)"
            )

        stats_text = "\n".join(stats_lines)
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Legend at top (figure-level)
        from matplotlib.patches import Patch

        legend_elements = []
        # use unique raw status strings but sorted
        for s in sorted(set(status_keys), key=self._legend_sort_key):
            color, label = self._get_status_color_and_label(s)
            legend_elements.append(
                Patch(
                    facecolor=color,
                    edgecolor="none",
                    label=label,
                    alpha=0.85,
                )
            )

        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.91),
                ncol=3,
                fontsize=12,
                framealpha=0.9,
            )

        plt.tight_layout(rect=(0, 0, 1, 0.93))

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    # ------------------------------------------------------------------ #
    # Multi-agent (stacked) plot
    # ------------------------------------------------------------------ #
    def create_multi_agent_graph(
        self,
        agent_dfs: List[Tuple[int, pd.DataFrame]],
        save_path: Optional[pathlib.Path] = None,
        show: bool = False,
        title: Optional[str] = None,
        step_range: Optional[Tuple[int, int]] = None,
        step_range_axis: Optional[Tuple[int, int]] = None,
        bar_width: float = 0.8,
        subplot_height: float = 4.0,
    ) -> plt.Figure:
        """
        Create multi-agent bar graph visualization with agents stacked vertically.

        step_range: filter data to these steps.
        step_range_axis: force x-axis to this range on all subplots.
        """

        n_agents = len(agent_dfs)
        if n_agents == 0:
            raise ValueError("No agent data provided for multi-agent graph.")

        fig_height = subplot_height * n_agents
        fig, axes = plt.subplots(
            n_agents,
            1,
            figsize=(self.figure_width, fig_height),
            dpi=self.dpi,
            sharex=True,
        )

        if n_agents == 1:
            axes = [axes]

        # For building a global legend
        all_status_keys: List[str] = []

        for ax, (agent_idx, df) in zip(axes, agent_dfs):
            # Restrict to step range (data) if requested
            if step_range is not None:
                start, end = step_range
                df_plot = df[(df["step"] >= start) & (df["step"] <= end)].copy()
            else:
                df_plot = df.copy()

            if df_plot.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for Agent {agent_idx}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.axis("off")
                continue

            steps = []
            iterations = []
            colors = []
            status_keys = []

            for idx, (_, row) in enumerate(df_plot.iterrows()):
                step_num = int(row.get("step", idx))
                raw_status = self._pick_status_string(row)
                color, _ = self._get_status_color_and_label(raw_status)

                # iterations: prefer ipopt_iterations, fallback mpc_iterations
                iter_val = row.get("ipopt_iterations", None)
                if iter_val is None or (isinstance(iter_val, float) and pd.isna(iter_val)):
                    iter_val = row.get("mpc_iterations", 0)
                if pd.isna(iter_val):
                    iter_val = 0
                iter_val = int(iter_val)

                steps.append(step_num)
                iterations.append(iter_val)
                colors.append(color)
                status_keys.append(str(raw_status))

            all_status_keys.extend(status_keys)

            ax.bar(steps, iterations, width=bar_width, color=colors,
                   edgecolor="none", alpha=0.85)

            ax.set_ylabel("Iter", fontsize=12)
            ax.set_title(f"Agent {agent_idx}", fontsize=14)

            # Per-agent simple stats
            avg_iter = float(np.mean(iterations)) if iterations else 0.0
            n_steps = len(steps)

            stats_text = f"Steps: {n_steps} | Avg Iter: {avg_iter:.1f}"
            ax.text(
                0.99,
                0.95,
                stats_text,
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Force x-axis range if requested
            if step_range_axis is not None:
                x_start, x_end = step_range_axis
                ax.set_xlim(x_start, x_end)

        # X label on ALL subplots
        for ax in axes:
            ax.set_xlabel("Step", fontsize=12)
            ax.tick_params(axis="x", labelrotation=0)

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        # Global legend at top
        from matplotlib.patches import Patch

        unique_statuses_all = sorted(set(all_status_keys), key=self._legend_sort_key)
        legend_elements = []
        for s in unique_statuses_all:
            color, label = self._get_status_color_and_label(s)
            legend_elements.append(
                Patch(
                    facecolor=color,
                    edgecolor="none",
                    label=label,
                    alpha=0.85,
                )
            )

        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.96),
                ncol=3,
                fontsize=12,
                framealpha=0.9,
            )

        plt.tight_layout(rect=(0, 0, 1, 0.9))

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Generate IPOPT status / iteration bar graph from agent step logs"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to directory containing agent_step_logs (or a run dir that has it)",
    )
    parser.add_argument(
        "--agent",
        type=int,
        help="Specific agent index to visualize (single figure)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        help="Multiple agent indices to visualize (combined stacked figure)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for visualizations",
    )
    parser.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of steps to visualize (data filter)",
    )
    parser.add_argument(
        "--step-range-axis",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Force x-axis to show this step range, even if some steps have no data",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization interactively",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create a single image with all agents stacked vertically",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=16,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--figure-height",
        type=int,
        default=6,
        help="Figure height in inches (for single-agent view)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure resolution (dots per inch)",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.8,
        help="Width of each bar",
    )
    parser.add_argument(
        "--subplot-height",
        type=float,
        default=4.0,
        help="Height of each subplot in combined mode",
    )

    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        return

    visualizer = StatusBarGraphVisualizer(
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
    )

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

    # Try to find agent_step_logs directory within log_dir
    step_log_dirs = list(log_dir.glob("**/agent_step_logs"))
    if not step_log_dirs:
        print(f"Error: No agent_step_logs directory found in {log_dir}")
        return
    step_log_dir = step_log_dirs[0]

    # Load agent CSVs
    agent_files = sorted(step_log_dir.glob("agent_*.csv"))

    # Filter by agent indices if provided
    agent_indices = None
    if args.agent is not None:
        agent_indices = [args.agent]
    if args.agents is not None:
        agent_indices = args.agents

    if agent_indices is not None:
        agent_files = [
            f
            for f in agent_files
            if int(f.stem.split("_")[1]) in agent_indices
        ]

    if not agent_files:
        print(f"No agent CSV files found in {step_log_dir} after filtering.")
        return

    step_range = tuple(args.step_range) if args.step_range is not None else None
    step_range_axis = tuple(args.step_range_axis) if args.step_range_axis is not None else None

    # Single agent mode (one figure per agent)
    if args.agent is not None and not args.combined:
        csv_path = step_log_dir / f"agent_{args.agent:02d}.csv"
        df = visualizer.load_step_log(csv_path)
        if df is None:
            return

        title = f"IPOPT Iterations per Step - Agent {args.agent}"
        save_path = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"agent_{args.agent:02d}_status_bar.png"

        visualizer.create_bar_graph(
            df,
            save_path=save_path,
            show=args.show,
            title=title,
            step_range=step_range,
            step_range_axis=step_range_axis,
            bar_width=args.bar_width,
        )

    # Combined multi-agent mode
    else:
        agent_dfs: List[Tuple[int, pd.DataFrame]] = []
        for f in agent_files:
            agent_idx = int(f.stem.split("_")[1])
            df = visualizer.load_step_log(f)
            if df is None:
                continue
            agent_dfs.append((agent_idx, df))

        if not agent_dfs:
            print("No agent data could be loaded.")
            return

        title = "IPOPT Iterations per Step - Multi-Agent"

        if output_dir is None:
            output_dir = log_dir / "status_bar_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "multi_agent_status_bar.png"

        visualizer.create_multi_agent_graph(
            agent_dfs,
            save_path=save_path,
            show=args.show,
            title=title,
            step_range=step_range,
            step_range_axis=step_range_axis,
            bar_width=args.bar_width,
            subplot_height=args.subplot_height,
        )


if __name__ == "__main__":
    main()

