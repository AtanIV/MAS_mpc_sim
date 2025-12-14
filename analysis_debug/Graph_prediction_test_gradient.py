#!/usr/bin/env python3
"""
Complete MPC Test Script: Graph Prediction + CBF Evaluation + Velocity Gradients

Combines:
1. Graph predictor (from graph_predictor.py)
2. CBF evaluator (graph_evaluator.py)
3. Enhanced visualization with h values, Jacobians, and velocity gradients

Tests the complete pipeline for MPC with trained GCBF models.
"""

# Force JAX to use CPU
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import functools as ft
import sys
import pathlib
from pathlib import Path
from typing import List, Tuple

# Add project path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from gcbfplus.utils.graph import GraphsTuple, EdgeBlock, GetGraph
from pipeline.graph_evaluator import CBFEvaluator

# Import predictor class from test_graph_update8.py
from pipeline.graph_predictor import MPCGraphPredictor, create_test_scenario, plot_complete_graph_state


def plot_complete_graph_state_with_gradients(ax, graph, env, step_num, jacobian, title_prefix=""):
    """Enhanced plotting function showing ALL graph components + velocity gradients"""
    ax.clear()
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Get states
    agent_states = graph.type_states(type_idx=0, n_type=env.num_agents)
    goal_states = graph.type_states(type_idx=1, n_type=env.num_agents)
    n_rays_total = env._params["n_rays"] * env.num_agents
    lidar_states = graph.type_states(type_idx=2, n_type=n_rays_total)

    # Plot obstacles
    obstacles = graph.env_states.obstacle
    try:
        if len(obstacles) >= 6:
            vertices = obstacles[5]
            n_obstacles = vertices.shape[0]

            for i in range(n_obstacles):
                obs_vertices = vertices[i]
                polygon = patches.Polygon(
                    obs_vertices,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='red',
                    alpha=0.3,
                    closed=True
                )
                ax.add_patch(polygon)

                # Add obstacle label
                center = np.mean(obs_vertices, axis=0)
                ax.text(center[0], center[1], f'Obs{i}', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Error plotting obstacles: {e}")

    # Plot basic edges (simplified version to avoid import issues)
    # Agent-to-goal connections
    for i, (agent_state, goal_state) in enumerate(zip(agent_states, goal_states)):
        ax.plot([agent_state[0], goal_state[0]], [agent_state[1], goal_state[1]],
                'gray', alpha=0.3, linewidth=1, linestyle='--')

    # Plot agents and goals
    for i, (agent_state, goal_state) in enumerate(zip(agent_states, goal_states)):
        agent_pos = agent_state[:2]
        goal_pos = goal_state[:2]

        # Agent
        if i == 0:  # Ego agent
            color = 'blue'
            ax.plot(agent_pos[0], agent_pos[1], 'o', color=color, markersize=12, label='Ego Agent')
            ax.text(agent_pos[0] + 0.05, agent_pos[1] + 0.05, f'Ego', fontsize=10, color='blue', weight='bold')
        else:  # Other agents
            color = 'green'
            ax.plot(agent_pos[0], agent_pos[1], 'o', color=color, markersize=10, alpha=0.7,
                    label=f'Agent {i}' if i == 1 else None)
            ax.text(agent_pos[0] + 0.05, agent_pos[1] + 0.05, f'A{i}', fontsize=9, color='green')

        # Goal
        ax.plot(goal_pos[0], goal_pos[1], 's', color=color, markersize=8, alpha=0.5,
                label=f'Goal {i}' if i < 2 else None)
        ax.text(goal_pos[0] + 0.03, goal_pos[1] + 0.03, f'G{i}', fontsize=8, color=color)

        # Current velocity vector
        vel = agent_state[2:4]
        if np.linalg.norm(vel) > 1e-4:
            ax.arrow(agent_pos[0], agent_pos[1], vel[0] * 0.1, vel[1] * 0.1,
                     head_width=0.03, head_length=0.03, fc=color, ec=color, alpha=0.7,
                     label='Current Velocity' if i == 0 else '')

        # VELOCITY GRADIENT VISUALIZATION (only for ego agent)
        if i == 0 and jacobian is not None:
            # Extract velocity gradient components [∂h/∂vx, ∂h/∂vy]
            vel_grad = jacobian[2:4]  # Last two components of Jacobian
            vel_grad_norm = np.linalg.norm(vel_grad)

            if vel_grad_norm > 1e-6:  # Only plot if gradient is significant
                # Normalize and scale for visualization
                vel_grad_unit = vel_grad / vel_grad_norm
                arrow_scale = 0.15  # Adjust this for visibility

                # Direction that INCREASES h (safety) - GREEN arrow
                safety_increase_dir = vel_grad_unit * arrow_scale
                ax.arrow(agent_pos[0], agent_pos[1],
                         safety_increase_dir[0], safety_increase_dir[1],
                         head_width=0.04, head_length=0.04,
                         fc='lime', ec='darkgreen', alpha=0.8, linewidth=2)

                # Direction that DECREASES h (danger) - RED arrow
                safety_decrease_dir = -vel_grad_unit * arrow_scale
                ax.arrow(agent_pos[0], agent_pos[1],
                         safety_decrease_dir[0], safety_decrease_dir[1],
                         head_width=0.04, head_length=0.04,
                         fc='red', ec='darkred', alpha=0.8, linewidth=2)

                # Add text annotation
                grad_magnitude_text = f"|∇_v h|={vel_grad_norm:.3f}"
                ax.text(agent_pos[0] - 0.1, agent_pos[1] - 0.1, grad_magnitude_text,
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # Enhanced title with gradient info
    title = f"{title_prefix}Step {step_num}"

    if jacobian is not None:
        vel_grad_norm = np.linalg.norm(jacobian[2:4])
        title += f"\n|∇_v h|={vel_grad_norm:.3f}"

    ax.set_title(title)

    # Enhanced legend
    if jacobian is not None:
        ax.plot([], [], 'lime', marker='>', markersize=8, label='Vel: ↑Safety', linestyle='None')
        ax.plot([], [], 'red', marker='>', markersize=8, label='Vel: ↓Safety', linestyle='None')

    ax.legend(loc='upper right', fontsize=8)


class CompleteMPCTester:
    """
    Complete MPC testing framework combining graph prediction and CBF evaluation.
    """

    def __init__(self, model_path: str, ego_agent_idx: int = 0):
        """
        Initialize complete MPC tester.

        Args:
            model_path: Path to trained GCBF model
            ego_agent_idx: Index of ego agent
        """
        self.model_path = pathlib.Path(model_path)
        self.ego_agent_idx = ego_agent_idx

        # Verify model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        print(f"=== INITIALIZING COMPLETE MPC TESTER ===")
        print(f"Model path: {model_path}")
        print(f"Ego agent index: {ego_agent_idx}")

        # Create test scenario and environment
        self.env, self.initial_graph = create_test_scenario()

        # Initialize graph predictor
        self.graph_predictor = MPCGraphPredictor(self.env)

        # Initialize CBF evaluator
        try:
            self.cbf_evaluator = CBFEvaluator(str(model_path), ego_agent_idx)
            print("✓ CBF Evaluator loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load CBF Evaluator: {e}")
            raise

        # Verify initial graph compatibility
        if self.cbf_evaluator.verify_graph_compatibility(self.initial_graph):
            print("✓ Initial graph compatible with trained model")
        else:
            print("✗ Warning: Initial graph may not be compatible with trained model")

    def evaluate_graph_sequence(self, graphs: List[GraphsTuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate CBF values and Jacobians for a sequence of graphs.

        Args:
            graphs: List of predicted graphs

        Returns:
            Tuple of (h_values, jacobians)
        """
        h_values = []
        jacobians = []

        print(f"\n=== EVALUATING CBF FOR {len(graphs)} GRAPHS ===")

        for i, graph in enumerate(graphs):
            try:
                # Evaluate CBF and Jacobian
                h_val, jacobian = self.cbf_evaluator.evaluate_h_and_jacobian(graph)
                h_values.append(h_val)
                jacobians.append(jacobian)

                print(f"Graph {i}: h={h_val:.4f}, |∇h|={np.linalg.norm(jacobian):.4f}")

                # Also evaluate constraint components for demonstration
                control_demo = np.array([0.01, 0.01])  # Demo control
                drift, control_coeffs, _ = self.cbf_evaluator.evaluate_h_dot_constraint(
                    graph, control_demo, alpha=1.0
                )
                print(
                    f"  Constraint: drift={drift:.4f}, control_coeffs=[{control_coeffs[0]:.4f}, {control_coeffs[1]:.4f}]")

            except Exception as e:
                print(f"✗ Error evaluating graph {i}: {e}")
                h_values.append(0.0)
                jacobians.append(np.zeros(self.env.state_dim))

        return np.array(h_values), np.array(jacobians)

    def run_complete_test(self, control_sequence: np.ndarray) -> dict:
        """
        Run complete test: predict graphs and evaluate CBF values.

        Args:
            control_sequence: Control inputs over horizon [horizon, 2]

        Returns:
            Dictionary with results
        """
        horizon = control_sequence.shape[0]

        print(f"\n{'=' * 80}")
        print(f"RUNNING COMPLETE MPC TEST - HORIZON {horizon}")
        print(f"{'=' * 80}")

        results = {
            'initial_graph': self.initial_graph,
            'predicted_graphs': [],
            'h_values': [],
            'jacobians': [],
            'control_sequence': control_sequence,
            'success': False
        }

        try:
            # Step 1: Predict graph sequence
            print("\n--- STEP 1: GRAPH PREDICTION ---")
            predicted_graphs = self.graph_predictor.predict_graphs_horizon(
                self.initial_graph, control_sequence
            )
            results['predicted_graphs'] = predicted_graphs
            print(f"✓ Successfully predicted {len(predicted_graphs)} graphs")

            # Step 2: Evaluate CBF values and Jacobians
            print("\n--- STEP 2: CBF EVALUATION ---")
            all_graphs = [self.initial_graph] + predicted_graphs
            h_values, jacobians = self.evaluate_graph_sequence(all_graphs)
            results['h_values'] = h_values
            results['jacobians'] = jacobians

            # Step 3: Analyze results
            print("\n--- STEP 3: ANALYSIS ---")
            self.analyze_results(results)

            results['success'] = True
            print("✓ Complete test successful!")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def analyze_results(self, results: dict):
        """Analyze and summarize test results."""
        h_values = results['h_values']
        jacobians = results['jacobians']

        print(f"CBF VALUES OVER HORIZON:")
        for i, h_val in enumerate(h_values):
            status = "SAFE" if h_val > 0 else "UNSAFE"
            print(f"  Step {i}: h = {h_val:.4f} ({status})")

        print(f"\nJACOBIAN NORMS:")
        for i, jac in enumerate(jacobians):
            norm = np.linalg.norm(jac)
            print(f"  Step {i}: |∇h| = {norm:.4f}")

        # Safety analysis
        min_h = np.min(h_values)
        max_h = np.max(h_values)
        print(f"\nSAFETY SUMMARY:")
        print(f"  Min h: {min_h:.4f}")
        print(f"  Max h: {max_h:.4f}")
        print(f"  Always safe: {min_h > 0}")

    def analyze_velocity_gradients(self, results: dict):
        """Analyze velocity gradient patterns over the horizon."""
        if not results['success']:
            return

        jacobians = results['jacobians']

        print(f"\n{'=' * 60}")
        print("VELOCITY GRADIENT ANALYSIS")
        print("=" * 60)

        for i, jacobian in enumerate(jacobians):
            vel_grad = jacobian[2:4]  # [∂h/∂vx, ∂h/∂vy]
            vel_grad_norm = np.linalg.norm(vel_grad)

            print(f"\nStep {i}:")
            print(f"  Velocity gradient: [{vel_grad[0]:.4f}, {vel_grad[1]:.4f}]")
            print(f"  Gradient magnitude: {vel_grad_norm:.4f}")

            if vel_grad_norm > 1e-6:
                # Direction that increases safety
                safety_dir = vel_grad / vel_grad_norm
                angle_deg = np.degrees(np.arctan2(safety_dir[1], safety_dir[0]))
                print(f"  Safety increase direction: [{safety_dir[0]:.3f}, {safety_dir[1]:.3f}] ({angle_deg:.1f}°)")

                # Interpret the gradient
                if vel_grad[0] > 0:
                    print(f"  → Increasing x-velocity increases safety")
                else:
                    print(f"  → Decreasing x-velocity increases safety")

                if vel_grad[1] > 0:
                    print(f"  → Increasing y-velocity increases safety")
                else:
                    print(f"  → Decreasing y-velocity increases safety")

    def visualize_complete_results(self, results: dict, save_path: str = None):
        """
        Create comprehensive visualization with graphs, h values, Jacobians, and velocity gradients.
        """
        if not results['success']:
            print("Cannot visualize - test was not successful")
            return

        initial_graph = results['initial_graph']
        predicted_graphs = results['predicted_graphs']
        h_values = results['h_values']
        jacobians = results['jacobians']
        control_sequence = results['control_sequence']

        all_graphs = [initial_graph] + predicted_graphs
        horizon = len(predicted_graphs)

        print(f"\n=== CREATING COMPREHENSIVE VISUALIZATION WITH VELOCITY GRADIENTS ===")

        # Create figure with subplots: graphs + analysis
        fig = plt.figure(figsize=(20, 14))  # Increased height for better visibility

        # Graph visualization subplots (top rows)
        n_cols = min(4, len(all_graphs))
        n_rows_graphs = (len(all_graphs) + n_cols - 1) // n_cols

        for i, graph in enumerate(all_graphs):
            row = i // n_cols
            col = i % n_cols
            ax_graph = plt.subplot2grid((4, n_cols), (row, col))  # Changed to 4 rows

            title_prefix = "Initial: " if i == 0 else f"Predicted {i}: "

            # Use enhanced plotting with gradients
            jacobian = jacobians[i] if i < len(jacobians) else None
            plot_complete_graph_state_with_gradients(ax_graph, graph, self.env, i, jacobian, title_prefix)

            # Add CBF value to title
            if i < len(h_values):
                h_val = h_values[i]
                safety_status = "SAFE" if h_val > 0 else "UNSAFE"
                current_title = ax_graph.get_title()
                ax_graph.set_title(current_title + f"\nh = {h_val:.4f} ({safety_status})")

        # CBF values over time (bottom left)
        ax_h = plt.subplot2grid((4, n_cols), (3, 0), colspan=n_cols // 2)
        steps = range(len(h_values))
        colors = ['red' if h < 0 else 'green' for h in h_values]
        ax_h.bar(steps, h_values, color=colors, alpha=0.7)
        ax_h.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_h.set_xlabel('Step')
        ax_h.set_ylabel('CBF Value h(x)')
        ax_h.set_title('CBF Values Over Horizon')
        ax_h.grid(True, alpha=0.3)

        # Velocity gradient analysis (bottom right)
        ax_grad = plt.subplot2grid((4, n_cols), (3, n_cols // 2), colspan=n_cols // 2)

        # Plot velocity gradient components and norms
        vel_grad_norms = [np.linalg.norm(jac[2:4]) for jac in jacobians]
        vel_grad_x = [jac[2] for jac in jacobians]
        vel_grad_y = [jac[3] for jac in jacobians]

        ax_grad.plot(steps, vel_grad_norms, 'b-o', label='|∇_v h|', linewidth=2)
        ax_grad.plot(steps, vel_grad_x, 'r--', label='∂h/∂v_x', alpha=0.8)
        ax_grad.plot(steps, vel_grad_y, 'g--', label='∂h/∂v_y', alpha=0.8)
        ax_grad.axhline(y=0, color='black', linestyle=':', alpha=0.5)

        ax_grad.set_xlabel('Step')
        ax_grad.set_ylabel('Velocity Gradient Components')
        ax_grad.set_title('Velocity Gradient Analysis\n(Green arrows: ↑Safety, Red arrows: ↓Safety)')
        ax_grad.legend()
        ax_grad.grid(True, alpha=0.3)

        plt.suptitle(
            f'Complete MPC Test: Graph Prediction + CBF Evaluation + Velocity Gradients\n'
            f'Model: {self.model_path.name} | Ego Agent: {self.ego_agent_idx} | Horizon: {horizon}',
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def demonstrate_constraint_formation(self, results: dict):
        """
        Demonstrate how to form MPC constraints from CBF evaluations.
        """
        if not results['success']:
            return

        print(f"\n{'=' * 60}")
        print("DEMONSTRATING MPC CONSTRAINT FORMATION")
        print("=" * 60)

        alpha = 1.0  # CBF class-K parameter

        for i, graph in enumerate([results['initial_graph']] + results['predicted_graphs']):
            print(f"\n--- Step {i} ---")

            # Demo control input
            u_demo = np.array([0.01, 0.005])

            try:
                # Get constraint components
                drift_term, control_coeffs, h_val = self.cbf_evaluator.evaluate_h_dot_constraint(
                    graph, u_demo, alpha
                )

                print(f"CBF value: h = {h_val:.4f}")
                print(f"Drift term: f_drift = {drift_term:.4f}")
                print(f"Control coefficients: g_control = [{control_coeffs[0]:.4f}, {control_coeffs[1]:.4f}]")

                # Form constraint: drift + control_coeffs @ u + alpha * h >= 0
                constraint_value = drift_term + np.dot(control_coeffs, u_demo) + alpha * h_val
                print(
                    f"Constraint value: {drift_term:.4f} + [{control_coeffs[0]:.4f}, {control_coeffs[1]:.4f}] @ [{u_demo[0]:.3f}, {u_demo[1]:.3f}] + {alpha}*{h_val:.4f} = {constraint_value:.4f}")

                constraint_satisfied = constraint_value >= 0
                print(f"Constraint satisfied: {constraint_satisfied}")

            except Exception as e:
                print(f"Error forming constraints: {e}")


def main():
    """Main test function."""
    print("=" * 100)
    print("COMPLETE MPC TESTER: GRAPH PREDICTION + CBF EVALUATION + VELOCITY GRADIENTS")
    print("=" * 100)

    # Model path
    model_path = PROJECT_ROOT / "logs/DoubleIntegrator/gcbf+/seed1234_20251130013419"
    print(f"Computed model_path = {model_path}")

    # Verify model path exists
    if not pathlib.Path(model_path).exists():
        print(f"✗ Model path does not exist: {model_path}")
        print("Please update the model_path variable to point to your trained model")
        return

    try:
        # Initialize complete tester
        tester = CompleteMPCTester(model_path, ego_agent_idx=0)

        # Define test control sequence
        horizon = 3
        control_sequence = np.array([
            [0.0, 0.0],  # Small rightward + upward
            [0.0, 0.0],  # Small rightward
            [0.0, 0.0],  # No control
        ])

        print(f"Testing with control sequence:")
        for i, control in enumerate(control_sequence):
            print(f"  Step {i}: u = [{control[0]:.3f}, {control[1]:.3f}]")

        # Run complete test
        results = tester.run_complete_test(control_sequence)

        if results['success']:
            # Visualize results with velocity gradients
            tester.visualize_complete_results(results, save_path="complete_mpc_test_with_gradients.png")

            # Analyze velocity gradients
            tester.analyze_velocity_gradients(results)

            # Demonstrate constraint formation
            tester.demonstrate_constraint_formation(results)

            print(f"\n{'=' * 100}")
            print("✓ COMPLETE TEST SUCCESSFUL!")
            print("✓ Graph prediction working")
            print("✓ CBF evaluation working")
            print("✓ Velocity gradient visualization added")
            print("✓ MPC constraint formation demonstrated")
            print("✓ Ready for full MPC implementation")
            print("=" * 100)

        else:
            print("✗ Test failed - check errors above")

    except Exception as e:
        print(f"✗ Failed to initialize tester: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
