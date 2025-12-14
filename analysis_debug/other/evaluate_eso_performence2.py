#!/usr/bin/env python3
"""
ESO Performance Analysis Script

Analyzes ESO (Extended State Observer) performance from logged CSV data.
Compares ESO estimates vs true states and disturbances for a specified agent.
Creates separate plots for X and Y axes since they are decoupled.

USAGE:
1. Easy Configuration Mode:
   - Edit the configuration variables below (DATA_FOLDER, AGENT_ID, etc.)
   - Uncomment quick_run() in the main section
   - Run: python eso_analysis.py

2. Command Line Mode:
   - Run: python eso_analysis.py /path/to/logs --agent 0 --episode 1

3. Examples:
   - Analyze Agent 0: Set AGENT_ID = 0
   - Compare all agents: Set COMPARE_ALL = True
   - Analyze specific episode: Set EPISODE_ID = 1
   - Skip saving plots: Set SAVE_PLOTS = False
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class ESOAnalyzer:
    """Analyzer for ESO performance data from CSV logs."""

    def __init__(self, log_dir: str):
        """Initialize analyzer with log directory path."""
        self.log_dir = pathlib.Path(log_dir)
        self.agent_states = None
        self.eso_data = None
        self.disturbances = None
        self.mpc_status = None

        # Load all CSV files
        self._load_data()

    def _load_data(self):
        """Load all CSV files into pandas DataFrames."""
        try:
            self.agent_states = pd.read_csv(self.log_dir / "agent_states.csv")
            print(f"✅ Loaded agent_states.csv: {len(self.agent_states)} rows")
        except FileNotFoundError:
            print("❌ agent_states.csv not found")

        try:
            self.eso_data = pd.read_csv(self.log_dir / "eso_data.csv")
            print(f"✅ Loaded eso_data.csv: {len(self.eso_data)} rows")
        except FileNotFoundError:
            print("❌ eso_data.csv not found")

        try:
            self.disturbances = pd.read_csv(self.log_dir / "disturbances.csv")
            print(f"✅ Loaded disturbances.csv: {len(self.disturbances)} rows")
        except FileNotFoundError:
            print("❌ disturbances.csv not found")

        try:
            self.mpc_status = pd.read_csv(self.log_dir / "mpc_status.csv")
            print(f"✅ Loaded mpc_status.csv: {len(self.mpc_status)} rows")
        except FileNotFoundError:
            print("❌ mpc_status.csv not found")

    def get_available_agents(self) -> List[int]:
        """Get list of available agent IDs."""
        if self.agent_states is not None:
            return sorted(self.agent_states['agent_id'].unique())
        return []

    def get_available_episodes(self) -> List[int]:
        """Get list of available episode IDs."""
        if self.agent_states is not None:
            return sorted(self.agent_states['episode'].unique())
        return []

    def analyze_agent(self, agent_id: int, episode: Optional[int] = None,
                      save_plots: bool = True, show_plots: bool = True) -> Dict:
        """
        Analyze ESO performance for a specific agent.

        Args:
            agent_id: Agent ID to analyze
            episode: Specific episode to analyze (None for all episodes)
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots

        Returns:
            Dictionary containing analysis results and metrics
        """
        print(f"\n🔍 Analyzing Agent {agent_id}")
        if episode is not None:
            print(f"   Episode: {episode}")
        else:
            print(f"   Episodes: All available")

        # Filter data for the specified agent and episode
        agent_data = self._filter_data(agent_id, episode)

        if not agent_data:
            print(f"❌ No data found for agent {agent_id}")
            return {}

        # Calculate performance metrics
        metrics = self._calculate_metrics(agent_data)

        # Create plots
        if show_plots or save_plots:
            self._create_plots(agent_data, agent_id, episode, save_plots, show_plots)

        # Print summary
        self._print_summary(agent_id, episode, metrics)

        return metrics

    def _filter_data(self, agent_id: int, episode: Optional[int] = None) -> Dict:
        """Filter data for specific agent and episode."""
        data = {}

        # Filter conditions
        agent_filter = lambda df: df['agent_id'] == agent_id
        episode_filter = lambda df: df['episode'] == episode if episode is not None else df

        # Filter agent states
        if self.agent_states is not None:
            filtered = self.agent_states[agent_filter(self.agent_states)]
            if episode is not None:
                filtered = filtered[filtered['episode'] == episode]
            data['states'] = filtered

        # Filter ESO data
        if self.eso_data is not None:
            filtered = self.eso_data[agent_filter(self.eso_data)]
            if episode is not None:
                filtered = filtered[filtered['episode'] == episode]
            data['eso'] = filtered

        # Filter disturbances
        if self.disturbances is not None:
            filtered = self.disturbances[agent_filter(self.disturbances)]
            if episode is not None:
                filtered = filtered[filtered['episode'] == episode]
            data['disturbances'] = filtered

        # Filter MPC status
        if self.mpc_status is not None:
            filtered = self.mpc_status[agent_filter(self.mpc_status)]
            if episode is not None:
                filtered = filtered[filtered['episode'] == episode]
            data['mpc'] = filtered

        return data

    def _calculate_metrics(self, agent_data: Dict) -> Dict:
        """Calculate performance metrics for the agent."""
        metrics = {}

        if 'eso' in agent_data and not agent_data['eso'].empty:
            eso_df = agent_data['eso']

            # Separate X and Y axis data
            x_data = eso_df[eso_df['axis'] == 'x']
            y_data = eso_df[eso_df['axis'] == 'y']

            # Position estimation metrics
            metrics['position'] = {
                'x': {
                    'rmse': np.sqrt(np.mean(x_data['pos_error'] ** 2)) if not x_data.empty else 0,
                    'mae': np.mean(np.abs(x_data['pos_error'])) if not x_data.empty else 0,
                    'max_error': np.max(np.abs(x_data['pos_error'])) if not x_data.empty else 0
                },
                'y': {
                    'rmse': np.sqrt(np.mean(y_data['pos_error'] ** 2)) if not y_data.empty else 0,
                    'mae': np.mean(np.abs(y_data['pos_error'])) if not y_data.empty else 0,
                    'max_error': np.max(np.abs(y_data['pos_error'])) if not y_data.empty else 0
                }
            }

            # Velocity estimation metrics
            metrics['velocity'] = {
                'x': {
                    'rmse': np.sqrt(np.mean(x_data['vel_error'] ** 2)) if not x_data.empty else 0,
                    'mae': np.mean(np.abs(x_data['vel_error'])) if not x_data.empty else 0,
                    'max_error': np.max(np.abs(x_data['vel_error'])) if not x_data.empty else 0
                },
                'y': {
                    'rmse': np.sqrt(np.mean(y_data['vel_error'] ** 2)) if not y_data.empty else 0,
                    'mae': np.mean(np.abs(y_data['vel_error'])) if not y_data.empty else 0,
                    'max_error': np.max(np.abs(y_data['vel_error'])) if not y_data.empty else 0
                }
            }

            # Disturbance estimation metrics
            metrics['disturbance'] = {
                'x': {
                    'rmse': np.sqrt(np.mean(x_data['dist_error'] ** 2)) if not x_data.empty else 0,
                    'mae': np.mean(np.abs(x_data['dist_error'])) if not x_data.empty else 0,
                    'max_error': np.max(np.abs(x_data['dist_error'])) if not x_data.empty else 0
                },
                'y': {
                    'rmse': np.sqrt(np.mean(y_data['dist_error'] ** 2)) if not y_data.empty else 0,
                    'mae': np.mean(np.abs(y_data['dist_error'])) if not y_data.empty else 0,
                    'max_error': np.max(np.abs(y_data['dist_error'])) if not y_data.empty else 0
                }
            }

        return metrics

    def _create_plots(self, agent_data: Dict, agent_id: int, episode: Optional[int],
                      save_plots: bool, show_plots: bool):
        """Create comprehensive plots for ESO analysis."""
        if 'eso' not in agent_data or agent_data['eso'].empty:
            print("❌ No ESO data available for plotting")
            return

        eso_df = agent_data['eso']

        # Separate X and Y axis data
        x_data = eso_df[eso_df['axis'] == 'x'].sort_values(['episode', 'step'])
        y_data = eso_df[eso_df['axis'] == 'y'].sort_values(['episode', 'step'])

        if x_data.empty or y_data.empty:
            print("❌ Insufficient data for plotting")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'ESO Performance Analysis - Agent {agent_id}' +
                     (f' (Episode {episode})' if episode is not None else ' (All Episodes)'),
                     fontsize=16, fontweight='bold')

        # Time vectors
        time_x = x_data['step'].values
        time_y = y_data['step'].values

        # Plot 1: Position Estimation (X-axis)
        ax = axes[0, 0]
        ax.plot(time_x, x_data['true_pos'], 'b-', linewidth=2, label='True Position', alpha=0.8)
        ax.plot(time_x, x_data['eso_pos'], 'r--', linewidth=2, label='ESO Estimate', alpha=0.8)
        ax.fill_between(time_x, x_data['true_pos'], x_data['eso_pos'], alpha=0.3, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('X Position')
        ax.set_title('Position Estimation - X Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Position Estimation (Y-axis)
        ax = axes[0, 1]
        ax.plot(time_y, y_data['true_pos'], 'b-', linewidth=2, label='True Position', alpha=0.8)
        ax.plot(time_y, y_data['eso_pos'], 'r--', linewidth=2, label='ESO Estimate', alpha=0.8)
        ax.fill_between(time_y, y_data['true_pos'], y_data['eso_pos'], alpha=0.3, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Y Position')
        ax.set_title('Position Estimation - Y Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Velocity Estimation (X-axis)
        ax = axes[1, 0]
        ax.plot(time_x, x_data['true_vel'], 'b-', linewidth=2, label='True Velocity', alpha=0.8)
        ax.plot(time_x, x_data['eso_vel'], 'r--', linewidth=2, label='ESO Estimate', alpha=0.8)
        ax.fill_between(time_x, x_data['true_vel'], x_data['eso_vel'], alpha=0.3, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('X Velocity')
        ax.set_title('Velocity Estimation - X Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Velocity Estimation (Y-axis)
        ax = axes[1, 1]
        ax.plot(time_y, y_data['true_vel'], 'b-', linewidth=2, label='True Velocity', alpha=0.8)
        ax.plot(time_y, y_data['eso_vel'], 'r--', linewidth=2, label='ESO Estimate', alpha=0.8)
        ax.fill_between(time_y, y_data['true_vel'], y_data['eso_vel'], alpha=0.3, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Y Velocity')
        ax.set_title('Velocity Estimation - Y Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Disturbance Estimation (X-axis)
        ax = axes[2, 0]
        ax.plot(time_x, x_data['true_dist'], 'b-', linewidth=2, label='True Disturbance', alpha=0.8)
        ax.plot(time_x, x_data['eso_dist'], 'r--', linewidth=2, label='ESO Estimate', alpha=0.8)
        ax.fill_between(time_x, x_data['true_dist'], x_data['eso_dist'], alpha=0.3, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('X Disturbance')
        ax.set_title('Disturbance Estimation - X Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Disturbance Estimation (Y-axis)
        ax = axes[2, 1]
        ax.plot(time_y, y_data['true_dist'], 'b-', linewidth=2, label='True Disturbance', alpha=0.8)
        ax.plot(time_y, y_data['eso_dist'], 'r--', linewidth=2, label='ESO Estimate', alpha=0.8)
        ax.fill_between(time_y, y_data['true_dist'], y_data['eso_dist'], alpha=0.3, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Y Disturbance')
        ax.set_title('Disturbance Estimation - Y Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot if requested
        if save_plots:
            episode_str = f"_ep{episode}" if episode is not None else "_all_episodes"
            filename = f"eso_analysis_agent{agent_id}{episode_str}.png"
            filepath = self.log_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved: {filepath}")

        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Create error plots
        self._create_error_plots(x_data, y_data, agent_id, episode, save_plots, show_plots)

    def _create_error_plots(self, x_data: pd.DataFrame, y_data: pd.DataFrame,
                            agent_id: int, episode: Optional[int],
                            save_plots: bool, show_plots: bool):
        """Create separate error analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'ESO Estimation Errors - Agent {agent_id}' +
                     (f' (Episode {episode})' if episode is not None else ' (All Episodes)'),
                     fontsize=16, fontweight='bold')

        time_x = x_data['step'].values
        time_y = y_data['step'].values

        # Position errors
        ax = axes[0, 0]
        ax.plot(time_x, x_data['pos_error'], 'r-', linewidth=1.5, label='X-axis', alpha=0.8)
        ax.plot(time_y, y_data['pos_error'], 'b-', linewidth=1.5, label='Y-axis', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position Error')
        ax.set_title('Position Estimation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity errors
        ax = axes[0, 1]
        ax.plot(time_x, x_data['vel_error'], 'r-', linewidth=1.5, label='X-axis', alpha=0.8)
        ax.plot(time_y, y_data['vel_error'], 'b-', linewidth=1.5, label='Y-axis', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Velocity Error')
        ax.set_title('Velocity Estimation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Disturbance errors
        ax = axes[1, 0]
        ax.plot(time_x, x_data['dist_error'], 'r-', linewidth=1.5, label='X-axis', alpha=0.8)
        ax.plot(time_y, y_data['dist_error'], 'b-', linewidth=1.5, label='Y-axis', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Disturbance Error')
        ax.set_title('Disturbance Estimation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error histograms
        ax = axes[1, 1]
        all_pos_errors_x = np.abs(x_data['pos_error'])
        all_pos_errors_y = np.abs(y_data['pos_error'])
        all_vel_errors_x = np.abs(x_data['vel_error'])
        all_vel_errors_y = np.abs(y_data['vel_error'])

        ax.hist(all_pos_errors_x, bins=30, alpha=0.6, label='Position X', color='red')
        ax.hist(all_pos_errors_y, bins=30, alpha=0.6, label='Position Y', color='blue')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot if requested
        if save_plots:
            episode_str = f"_ep{episode}" if episode is not None else "_all_episodes"
            filename = f"eso_errors_agent{agent_id}{episode_str}.png"
            filepath = self.log_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 Error plot saved: {filepath}")

        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()

    def _print_summary(self, agent_id: int, episode: Optional[int], metrics: Dict):
        """Print analysis summary."""
        print(f"\n📈 ESO Performance Summary - Agent {agent_id}")
        print("=" * 60)

        if 'position' in metrics:
            print("🎯 Position Estimation Performance:")
            print(f"   X-axis: RMSE={metrics['position']['x']['rmse']:.4f}, "
                  f"MAE={metrics['position']['x']['mae']:.4f}, "
                  f"Max Error={metrics['position']['x']['max_error']:.4f}")
            print(f"   Y-axis: RMSE={metrics['position']['y']['rmse']:.4f}, "
                  f"MAE={metrics['position']['y']['mae']:.4f}, "
                  f"Max Error={metrics['position']['y']['max_error']:.4f}")

        if 'velocity' in metrics:
            print("\n🏃 Velocity Estimation Performance:")
            print(f"   X-axis: RMSE={metrics['velocity']['x']['rmse']:.4f}, "
                  f"MAE={metrics['velocity']['x']['mae']:.4f}, "
                  f"Max Error={metrics['velocity']['x']['max_error']:.4f}")
            print(f"   Y-axis: RMSE={metrics['velocity']['y']['rmse']:.4f}, "
                  f"MAE={metrics['velocity']['y']['mae']:.4f}, "
                  f"Max Error={metrics['velocity']['y']['max_error']:.4f}")

        if 'disturbance' in metrics:
            print("\n🌊 Disturbance Estimation Performance:")
            print(f"   X-axis: RMSE={metrics['disturbance']['x']['rmse']:.4f}, "
                  f"MAE={metrics['disturbance']['x']['mae']:.4f}, "
                  f"Max Error={metrics['disturbance']['x']['max_error']:.4f}")
            print(f"   Y-axis: RMSE={metrics['disturbance']['y']['rmse']:.4f}, "
                  f"MAE={metrics['disturbance']['y']['mae']:.4f}, "
                  f"Max Error={metrics['disturbance']['y']['max_error']:.4f}")

        print("=" * 60)

    def compare_all_agents(self, episode: Optional[int] = None,
                           save_plots: bool = True, show_plots: bool = False) -> Dict:
        """Compare ESO performance across all agents."""
        available_agents = self.get_available_agents()

        if not available_agents:
            print("❌ No agents found")
            return {}

        print(f"\n🔍 Comparing ESO performance across {len(available_agents)} agents")

        all_metrics = {}

        # Analyze each agent
        for agent_id in available_agents:
            print(f"\nAnalyzing Agent {agent_id}...")
            metrics = self.analyze_agent(agent_id, episode, save_plots=False, show_plots=False)
            all_metrics[agent_id] = metrics

        # Create comparison plots
        if show_plots or save_plots:
            self._create_comparison_plots(all_metrics, episode, save_plots, show_plots)

        # Print comparison summary
        self._print_comparison_summary(all_metrics)

        return all_metrics

    def _create_comparison_plots(self, all_metrics: Dict, episode: Optional[int],
                                 save_plots: bool, show_plots: bool):
        """Create comparison plots across agents."""
        if not all_metrics:
            return

        agents = list(all_metrics.keys())

        # Extract metrics for comparison
        pos_rmse_x = [all_metrics[agent]['position']['x']['rmse'] for agent in agents if
                      'position' in all_metrics[agent]]
        pos_rmse_y = [all_metrics[agent]['position']['y']['rmse'] for agent in agents if
                      'position' in all_metrics[agent]]
        vel_rmse_x = [all_metrics[agent]['velocity']['x']['rmse'] for agent in agents if
                      'velocity' in all_metrics[agent]]
        vel_rmse_y = [all_metrics[agent]['velocity']['y']['rmse'] for agent in agents if
                      'velocity' in all_metrics[agent]]
        dist_rmse_x = [all_metrics[agent]['disturbance']['x']['rmse'] for agent in agents if
                       'disturbance' in all_metrics[agent]]
        dist_rmse_y = [all_metrics[agent]['disturbance']['y']['rmse'] for agent in agents if
                       'disturbance' in all_metrics[agent]]

        if not pos_rmse_x:
            print("❌ No metrics available for comparison plots")
            return

        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'ESO Performance Comparison Across Agents' +
                     (f' (Episode {episode})' if episode is not None else ' (All Episodes)'),
                     fontsize=16, fontweight='bold')

        x_pos = np.arange(len(agents))
        width = 0.35

        # Position RMSE comparison
        ax = axes[0]
        ax.bar(x_pos - width / 2, pos_rmse_x, width, label='X-axis', alpha=0.8, color='red')
        ax.bar(x_pos + width / 2, pos_rmse_y, width, label='Y-axis', alpha=0.8, color='blue')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Position RMSE')
        ax.set_title('Position Estimation RMSE')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity RMSE comparison
        ax = axes[1]
        ax.bar(x_pos - width / 2, vel_rmse_x, width, label='X-axis', alpha=0.8, color='red')
        ax.bar(x_pos + width / 2, vel_rmse_y, width, label='Y-axis', alpha=0.8, color='blue')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Velocity RMSE')
        ax.set_title('Velocity Estimation RMSE')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Disturbance RMSE comparison
        ax = axes[2]
        ax.bar(x_pos - width / 2, dist_rmse_x, width, label='X-axis', alpha=0.8, color='red')
        ax.bar(x_pos + width / 2, dist_rmse_y, width, label='Y-axis', alpha=0.8, color='blue')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Disturbance RMSE')
        ax.set_title('Disturbance Estimation RMSE')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot if requested
        if save_plots:
            episode_str = f"_ep{episode}" if episode is not None else "_all_episodes"
            filename = f"eso_comparison{episode_str}.png"
            filepath = self.log_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 Comparison plot saved: {filepath}")

        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()

    def _print_comparison_summary(self, all_metrics: Dict):
        """Print comparison summary across agents."""
        if not all_metrics:
            return

        print(f"\n📊 ESO Performance Comparison Summary")
        print("=" * 80)

        # Find best and worst performing agents
        agents = list(all_metrics.keys())

        if 'position' in all_metrics[agents[0]]:
            # Position performance
            pos_rmse_x = {agent: all_metrics[agent]['position']['x']['rmse'] for agent in agents}
            pos_rmse_y = {agent: all_metrics[agent]['position']['y']['rmse'] for agent in agents}

            best_pos_x = min(pos_rmse_x, key=pos_rmse_x.get)
            worst_pos_x = max(pos_rmse_x, key=pos_rmse_x.get)
            best_pos_y = min(pos_rmse_y, key=pos_rmse_y.get)
            worst_pos_y = max(pos_rmse_y, key=pos_rmse_y.get)

            print(f"🎯 Position Estimation:")
            print(f"   X-axis: Best=Agent {best_pos_x} (RMSE={pos_rmse_x[best_pos_x]:.4f}), "
                  f"Worst=Agent {worst_pos_x} (RMSE={pos_rmse_x[worst_pos_x]:.4f})")
            print(f"   Y-axis: Best=Agent {best_pos_y} (RMSE={pos_rmse_y[best_pos_y]:.4f}), "
                  f"Worst=Agent {worst_pos_y} (RMSE={pos_rmse_y[worst_pos_y]:.4f})")

        print("=" * 80)


# =============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# =============================================================================
DATA_FOLDER = Path(
    #"/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_gcbf_results_universal_compensation/0728-2047")
    "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_gcbf_results_universal_compensation/0729-2147")
AGENT_ID = 3  # Agent ID to analyze (set to None to analyze all agents)
EPISODE_ID = None  # Episode ID to analyze (set to None for all episodes)
SAVE_PLOTS = True  # Whether to save plots to files
SHOW_PLOTS = True  # Whether to display plots interactively
COMPARE_ALL = False  # Set to True to compare all agents instead of analyzing single agent


# =============================================================================


def main():
    """Main function - uses configuration variables above."""
    parser = argparse.ArgumentParser(description="Analyze ESO performance from CSV logs")

    parser.add_argument("log_dir", type=str, nargs='?', default=str(DATA_FOLDER),
                        help=f"Directory containing CSV log files (default: {DATA_FOLDER})")
    parser.add_argument("--agent", type=int, default=AGENT_ID,
                        help=f"Specific agent ID to analyze (default: {AGENT_ID})")
    parser.add_argument("--episode", type=int, default=EPISODE_ID,
                        help=f"Specific episode to analyze (default: {EPISODE_ID})")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots to files")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--compare-all", action="store_true", help="Compare all agents")

    args = parser.parse_args()

    # Use configuration variables if command line args not provided
    log_dir = args.log_dir
    agent_id = args.agent
    episode_id = args.episode
    save_plots = not args.no_save and SAVE_PLOTS
    show_plots = not args.no_show and SHOW_PLOTS
    compare_all = args.compare_all or COMPARE_ALL

    # Create analyzer
    analyzer = ESOAnalyzer(log_dir)

    # Check available data
    available_agents = analyzer.get_available_agents()
    available_episodes = analyzer.get_available_episodes()

    print(f"📁 Log directory: {log_dir}")
    print(f"🤖 Available agents: {available_agents}")
    print(f"📺 Available episodes: {available_episodes}")
    print(f"🔧 Configuration: Agent={agent_id}, Episode={episode_id}, Save={save_plots}, Show={show_plots}")

    if compare_all:
        # Compare all agents
        print(f"\n🔍 Comparing all agents...")
        analyzer.compare_all_agents(episode_id, save_plots, show_plots)
    elif agent_id is not None:
        # Analyze specific agent
        if agent_id in available_agents:
            print(f"\n🔍 Analyzing Agent {agent_id}...")
            analyzer.analyze_agent(agent_id, episode_id, save_plots, show_plots)
        else:
            print(f"❌ Agent {agent_id} not found. Available agents: {available_agents}")
    else:
        # Analyze all agents individually
        print(f"\n🔍 Analyzing all agents individually...")
        for agent in available_agents:
            analyzer.analyze_agent(agent, episode_id, save_plots, show_plots)


def quick_run():
    """Quick run using only configuration variables (no command line args)."""
    print("🚀 Quick Run Mode - Using Configuration Variables")

    # Create analyzer
    analyzer = ESOAnalyzer(DATA_FOLDER)

    # Check available data
    available_agents = analyzer.get_available_agents()
    available_episodes = analyzer.get_available_episodes()

    print(f"📁 Log directory: {DATA_FOLDER}")
    print(f"🤖 Available agents: {available_agents}")
    print(f"📺 Available episodes: {available_episodes}")
    print(f"🔧 Configuration: Agent={AGENT_ID}, Episode={EPISODE_ID}, Save={SAVE_PLOTS}, Show={SHOW_PLOTS}")

    if COMPARE_ALL:
        # Compare all agents
        print(f"\n🔍 Comparing all agents...")
        analyzer.compare_all_agents(EPISODE_ID, SAVE_PLOTS, SHOW_PLOTS)
    elif AGENT_ID is not None:
        # Analyze specific agent
        if AGENT_ID in available_agents:
            print(f"\n🔍 Analyzing Agent {AGENT_ID}...")
            analyzer.analyze_agent(AGENT_ID, EPISODE_ID, SAVE_PLOTS, SHOW_PLOTS)
        else:
            print(f"❌ Agent {AGENT_ID} not found. Available agents: {available_agents}")
    else:
        # Analyze all agents individually
        print(f"\n🔍 Analyzing all agents individually...")
        for agent in available_agents:
            analyzer.analyze_agent(agent, EPISODE_ID, SAVE_PLOTS, SHOW_PLOTS)


if __name__ == "__main__":
    # ==========================================================================
    # CHOOSE YOUR MODE:
    # ==========================================================================
    # Option 1: Quick run using configuration variables only (recommended)
    # Uncomment the line below and comment out main()
    quick_run()

    # Option 2: Use command line arguments (with config variables as defaults)
    # Uncomment the line below and comment out quick_run()
    # main()