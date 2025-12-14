#!/usr/bin/env python3
"""
Complete MPC Test Script: Graph Prediction + CBF Evaluation

Combines:
1. Graph predictor (from test_graph_update8.py)
2. CBF evaluator (graph_evaluator.py)
3. Enhanced visualization with h values and Jacobians

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
from typing import List, Tuple

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcbfplus.utils.graph import GraphsTuple, EdgeBlock, GetGraph
from graph_evaluator import CBFEvaluator

# Import predictor class
from graph_predictor import MPCGraphPredictor, create_test_scenario, plot_complete_graph_state


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

    def print_detailed_graph(self, graph: GraphsTuple, step: int, graph_type: str = ""):
        """
        Print detailed information about a graph for debugging.
        Based on the actual DoubleIntegrator implementation structure.

        Args:
            graph: The GraphsTuple to analyze
            step: Step number
            graph_type: Type description (e.g., "Initial", "Predicted")
        """
        print(f"\n{'=' * 60}")
        print(f"DETAILED GRAPH DEBUG - STEP {step} ({graph_type})")
        print(f"{'=' * 60}")

        # Convert JAX arrays to numpy for easier inspection
        def safe_convert(arr):
            """Safely convert JAX array to numpy and handle scalars."""
            if hasattr(arr, 'shape'):
                if len(arr.shape) == 0:  # scalar
                    return np.asarray(arr).item()
                else:
                    return np.asarray(arr)
            return arr

        # Basic graph structure
        nodes_np = safe_convert(graph.nodes)
        edges_np = safe_convert(graph.edges)
        states_np = safe_convert(graph.states)

        print(f"Graph Structure:")
        print(f"  - Total nodes: {nodes_np.shape[0]}")
        print(f"  - Node features dimension: {nodes_np.shape[1]}")
        print(f"  - Total edges: {edges_np.shape[0]}")
        print(f"  - Edge features dimension: {edges_np.shape[1]}")
        print(f"  - States shape: {states_np.shape}")

        # Handle n_node and n_edge
        n_node_val = safe_convert(graph.n_node)
        n_edge_val = safe_convert(graph.n_edge)
        print(f"  - Nodes per graph: {n_node_val}")
        print(f"  - Edges per graph: {n_edge_val}")

        # Analyze node types based on actual implementation
        if hasattr(graph, 'node_type'):
            node_type_np = safe_convert(graph.node_type)
            agent_nodes = np.sum(node_type_np == 0)
            goal_nodes = np.sum(node_type_np == 1)
            lidar_nodes = np.sum(node_type_np == 2)
            other_nodes = len(node_type_np) - agent_nodes - goal_nodes - lidar_nodes

            print(f"\nNode Types:")
            print(f"  - Agent nodes (type 0): {agent_nodes}")
            print(f"  - Goal nodes (type 1): {goal_nodes}")
            print(f"  - LiDAR nodes (type 2): {lidar_nodes}")
            if other_nodes > 0:
                print(f"  - Other/padding nodes: {other_nodes}")

        # Extract different node categories using type_states
        try:
            # Agent states [x, y, vx, vy]
            agent_states = graph.type_states(type_idx=0, n_type=self.env.num_agents)
            agent_states_np = safe_convert(agent_states)
            print(f"\nAgent States:")
            for i in range(agent_states_np.shape[0]):
                x, y, vx, vy = agent_states_np[i]
                agent_type = "EGO" if i == self.ego_agent_idx else f"A{i}"
                print(f"  {agent_type}: pos=({x:.4f}, {y:.4f}), vel=({vx:.4f}, {vy:.4f})")

            # Goal states [x, y, vx, vy] (velocities should be zero)
            goal_states = graph.type_states(type_idx=1, n_type=self.env.num_agents)
            goal_states_np = safe_convert(goal_states)
            print(f"\nGoal States:")
            for i in range(goal_states_np.shape[0]):
                x, y, vx, vy = goal_states_np[i]
                print(f"  G{i}: pos=({x:.4f}, {y:.4f}), vel=({vx:.4f}, {vy:.4f})")

            # LiDAR states [x, y, vx, vy] (velocities should be zero for static obstacles)
            n_rays_total = self.env._params["n_rays"] * self.env.num_agents
            lidar_states = graph.type_states(type_idx=2, n_type=n_rays_total)
            lidar_states_np = safe_convert(lidar_states)
            print(f"\nLiDAR States (total of {lidar_states_np.shape[0]}):")

            n_rays = self.env._params["n_rays"]
            for agent_idx in range(self.env.num_agents):
                start_idx = agent_idx * n_rays
                end_idx = min(start_idx + 16, lidar_states_np.shape[0])  # Show first 5 rays per agent

                print(f"  Agent {agent_idx} LiDAR rays:")
                for i in range(start_idx, end_idx):
                    x, y, vx, vy = lidar_states_np[i]
                    ray_idx = i - start_idx
                    print(f"    Ray {ray_idx}: pos=({x:.4f}, {y:.4f}), vel=({vx:.4f}, {vy:.4f})")

                if agent_idx == 0:  # Show hit analysis for ego agent
                    ego_pos = agent_states_np[self.ego_agent_idx, :2]
                    hits = 0
                    for i in range(start_idx, start_idx + n_rays):
                        if i < lidar_states_np.shape[0]:
                            lidar_pos = lidar_states_np[i, :2]
                            distance = np.linalg.norm(lidar_pos - ego_pos)
                            if distance < self.env._params["comm_radius"] - 1e-3:
                                hits += 1
                    print(f"    Ego agent LiDAR hits: {hits}/{n_rays}")

        except Exception as e:
            print(f"Error extracting typed states: {e}")

        # Edge analysis
        if edges_np.shape[0] > 0:
            senders_np = safe_convert(graph.senders)
            receivers_np = safe_convert(graph.receivers)

            print(f"\nEdge Analysis:")
            print(f"  - Total edges: {len(senders_np)}")
            print(f"  - Sender range: [{np.min(senders_np)}, {np.max(senders_np)}]")
            print(f"  - Receiver range: [{np.min(receivers_np)}, {np.max(receivers_np)}]")

            # Sample edges
            max_edges_to_show = min(40, edges_np.shape[0])
            print(f"  - Sample edges (first {max_edges_to_show}):")
            for i in range(max_edges_to_show):
                sender = senders_np[i]
                receiver = receivers_np[i]
                edge_feat = edges_np[i]
                print(
                    f"    Edge {i}: {sender} -> {receiver}, feat=[{edge_feat[0]:.3f}, {edge_feat[1]:.3f}, {edge_feat[2]:.3f}, {edge_feat[3]:.3f}]")

        # Environment state info
        if hasattr(graph, 'env_states') and graph.env_states is not None:
            print(f"\nEnvironment State:")
            if hasattr(graph.env_states, 'obstacle'):
                print(f"  - Obstacles present: Yes")
            else:
                print(f"  - Obstacles present: No")

        print(f"{'=' * 60}")

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
            # Print detailed graph information for debugging
            graph_type = "Initial" if i == 0 else f"Predicted {i}"
            self.print_detailed_graph(graph, i, graph_type)

            try:
                # Evaluate CBF and Jacobian
                h_val, jacobian = self.cbf_evaluator.evaluate_h_and_jacobian(graph)
                h_values.append(h_val)
                jacobians.append(jacobian)

                print(f"\nCBF EVALUATION RESULTS - STEP {i}:")
                print(f"  CBF Value: h={h_val:.4f}")
                print(f"  Jacobian: {jacobian}")
                print(f"  Jacobian norm: |∇h|={np.linalg.norm(jacobian):.4f}")

                # Also evaluate constraint components for demonstration
                control_demo = np.array([0.01, 0.01])  # Demo control
                drift, control_coeffs, _ = self.cbf_evaluator.evaluate_h_dot_constraint(
                    graph, control_demo, alpha=1.0
                )
                print(f"  Constraint Analysis:")
                print(f"    Drift term: {drift:.4f}")
                print(f"    Control coeffs: [{control_coeffs[0]:.4f}, {control_coeffs[1]:.4f}]")

            except Exception as e:
                print(f"✗ Error evaluating graph {i}: {e}")
                import traceback
                traceback.print_exc()
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

            # Print initial graph first
            print("INITIAL GRAPH:")
            self.print_detailed_graph(self.initial_graph, 0, "Initial")

            predicted_graphs = self.graph_predictor.predict_graphs_horizon(
                self.initial_graph, control_sequence
            )
            results['predicted_graphs'] = predicted_graphs
            print(f"✓ Successfully predicted {len(predicted_graphs)} graphs")

            # Print each predicted graph
            print(f"\nPREDICTED GRAPHS:")
            for i, pred_graph in enumerate(predicted_graphs):
                print(f"\nControl applied: u_{i} = [{control_sequence[i][0]:.4f}, {control_sequence[i][1]:.4f}]")
                self.print_detailed_graph(pred_graph, i + 1, f"Predicted {i + 1}")

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

        print(f"\n{'=' * 60}")
        print("RESULTS ANALYSIS")
        print(f"{'=' * 60}")

        print(f"CBF VALUES OVER HORIZON:")
        for i, h_val in enumerate(h_values):
            status = "SAFE" if h_val > 0 else "UNSAFE"
            graph_type = "Initial" if i == 0 else f"Predicted {i}"
            print(f"  Step {i} ({graph_type}): h = {h_val:.6f} ({status})")

        print(f"\nJACOBIAN ANALYSIS:")
        for i, jac in enumerate(jacobians):
            norm = np.linalg.norm(jac)
            graph_type = "Initial" if i == 0 else f"Predicted {i}"
            print(f"  Step {i} ({graph_type}): |∇h| = {norm:.6f}")
            print(f"    Jacobian components: [{jac[0]:.6f}, {jac[1]:.6f}, {jac[2]:.6f}, {jac[3]:.6f}]")

        # Safety analysis
        min_h = np.min(h_values)
        max_h = np.max(h_values)
        mean_h = np.mean(h_values)

        print(f"\nSAFETY SUMMARY:")
        print(f"  Min h: {min_h:.6f}")
        print(f"  Max h: {max_h:.6f}")
        print(f"  Mean h: {mean_h:.6f}")
        print(f"  Always safe: {min_h > 0}")
        print(f"  Safety margin: {min_h:.6f}")

        # Find most critical step
        min_idx = np.argmin(h_values)
        print(f"  Most critical step: {min_idx} (h = {h_values[min_idx]:.6f})")

    def visualize_complete_results(self, results: dict, save_path: str = None):
        """
        Create comprehensive visualization with graphs, h values, and Jacobians.
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

        print(f"\n=== CREATING COMPREHENSIVE VISUALIZATION ===")

        # Create figure with subplots: graphs + analysis
        fig = plt.figure(figsize=(20, 12))

        # Graph visualization subplots (top row)
        n_cols = min(4, len(all_graphs))
        n_rows_graphs = (len(all_graphs) + n_cols - 1) // n_cols

        for i, graph in enumerate(all_graphs):
            row = i // n_cols
            col = i % n_cols
            ax_graph = plt.subplot2grid((5, n_cols), (row, col))

            title_prefix = "Initial: " if i == 0 else f"Predicted {i}: "
            plot_complete_graph_state(ax_graph, graph, self.env, i, title_prefix)

            # Add CBF value to title
            h_val = h_values[i]
            safety_status = "SAFE" if h_val > 0 else "UNSAFE"
            ax_graph.set_title(ax_graph.get_title() + f"\nh = {h_val:.4f} ({safety_status})")

        plt.suptitle(
            f'Complete MPC Test: Graph Prediction + CBF Evaluation\n'
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
    print("COMPLETE MPC TESTER: GRAPH PREDICTION + CBF EVALUATION")
    print("=" * 100)

    # Model path (update this to your actual path)
    model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorMPC/gcbf+/seed0_20250626023916"

    # Verify model path exists
    if not pathlib.Path(model_path).exists():
        print(f"✗ Model path does not exist: {model_path}")
        print("Please update the model_path variable to point to your trained model")
        return

    try:
        # Initialize complete tester
        tester = CompleteMPCTester(model_path, ego_agent_idx=0)

        # Define test control sequence
        horizon = 5
        control_sequence = np.array([
            [2.0000, 2.0000],
            [2.0000, 2.0000],
            [2.0000, 2.0000],
            [2.0000, 2.0000],
            [2.0000, 2.0000],
            [1.9247, 1.0646],
            [0.3396, -0.0853],
            [-0.7171, -0.8292],
            [-1.3518, -1.2526],
            [-1.6626, -1.4380],
            [-1.7369, -1.4404],
            [-1.6452, -1.3383],
            [-1.4547, -1.1573],
            [-1.2102, -0.9495],
            [-0.9468, -0.7337],
            [-0.6910, -0.5232],
            [-0.4601, -0.3471],
            [-0.2692, -0.2004],
            [-0.1244, -0.0998],
        ])

        print(f"Testing with control sequence:")
        for i, control in enumerate(control_sequence):
            print(f"  Step {i}: u = [{control[0]:.3f}, {control[1]:.3f}]")

        # Run complete test
        results = tester.run_complete_test(control_sequence)

        if results['success']:
            # Visualize results
            tester.visualize_complete_results(results, save_path="complete_mpc_test.png")

            # Demonstrate constraint formation
            tester.demonstrate_constraint_formation(results)

            print(f"\n{'=' * 100}")
            print("✓ COMPLETE TEST SUCCESSFUL!")
            print("✓ Graph prediction working")
            print("✓ CBF evaluation working")
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