#!/usr/bin/env python3
"""
SQP-based MPC Controller with Dynamic CBF Constraints

Implements a Sequential Quadratic Programming approach for MPC optimization
that dynamically updates CBF constraints based on predicted graph evolution.

Dependencies:
- scipy.optimize for SQP solver
- jax for automatic differentiation and fast computation
- Your existing graph predictor and CBF evaluator
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
from scipy.optimize import minimize, NonlinearConstraint
import time
import sys
import pathlib
from typing import List, Tuple, Dict, Any

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcbfplus.utils.graph import GraphsTuple
from test_graph_update8 import MPCGraphPredictor, create_test_scenario
from graph_evaluator import CBFEvaluator


class SQPMPCController:
    """
    SQP-based MPC Controller with Dynamic CBF Constraints

    Optimizes control inputs over a horizon while respecting CBF safety constraints
    that are dynamically updated based on predicted graph evolution.
    """

    def __init__(self,
                 model_path: str,
                 env,
                 initial_graph: GraphsTuple,
                 ego_agent_idx: int = 0,
                 horizon: int = 4,
                 dt: float = 0.03,
                 alpha: float = 1.0,
                 control_bounds: Tuple[float, float] = (-1.0, 1.0),
                 reference_tracking_weight: float = 1.0,
                 control_effort_weight: float = 0.1):
        """
        Initialize SQP MPC Controller.

        Args:
            model_path: Path to trained GCBF model
            env: Environment instance
            initial_graph: Initial graph state
            ego_agent_idx: Index of ego agent
            horizon: MPC prediction horizon
            dt: Time step
            alpha: CBF class-K parameter
            control_bounds: Control input bounds (min, max)
            reference_tracking_weight: Weight for reference tracking cost
            control_effort_weight: Weight for control effort cost
        """
        self.env = env
        self.initial_graph = initial_graph
        self.ego_agent_idx = ego_agent_idx
        self.horizon = horizon
        self.dt = dt
        self.alpha = alpha
        self.control_bounds = control_bounds
        self.ref_weight = reference_tracking_weight
        self.control_weight = control_effort_weight

        # Initialize components
        print(f"Initializing SQP MPC Controller...")
        print(f"  Horizon: {horizon} steps")
        print(f"  Control bounds: {control_bounds}")
        print(f"  CBF alpha: {alpha}")

        # Initialize graph predictor
        self.graph_predictor = MPCGraphPredictor(env)
        print("  ✓ Graph predictor initialized")

        # Initialize CBF evaluator
        try:
            self.cbf_evaluator = CBFEvaluator(model_path, ego_agent_idx)
            print("  ✓ CBF evaluator initialized")
        except Exception as e:
            print(f"  ✗ Failed to initialize CBF evaluator: {e}")
            raise

        # Get ego agent's goal for reference tracking
        goal_states = initial_graph.type_states(type_idx=1, n_type=env.num_agents)
        self.ego_goal = goal_states[ego_agent_idx, :2]  # [x, y] position only

        # Initialize optimization variables
        self.control_dim = 2  # [ax, ay] for double integrator
        self.decision_vars = horizon * self.control_dim

        print(f"  ✓ Decision variables: {self.decision_vars} ({horizon} × {self.control_dim})")
        print(f"  ✓ Ego goal: ({self.ego_goal[0]:.3f}, {self.ego_goal[1]:.3f})")

        # Statistics
        self.optimization_stats = {
            'iterations': [],
            'solve_times': [],
            'objective_values': [],
            'constraint_violations': []
        }

    def extract_control_sequence(self, decision_vector: np.ndarray) -> np.ndarray:
        """Extract control sequence from decision vector."""
        return decision_vector.reshape((self.horizon, self.control_dim))

    def predict_ego_trajectory(self, control_sequence: np.ndarray) -> np.ndarray:
        """
        Predict ego agent trajectory given control sequence.

        Args:
            control_sequence: Control inputs [horizon, 2]

        Returns:
            trajectory: Ego states [horizon+1, 4] including initial state
        """
        # Get initial ego state
        agent_states = self.initial_graph.type_states(type_idx=0, n_type=self.env.num_agents)
        ego_state = agent_states[self.ego_agent_idx]

        trajectory = [ego_state]
        current_state = ego_state

        for control in control_sequence:
            # Double integrator dynamics
            pos = current_state[:2]
            vel = current_state[2:]
            accel = control / self.env._params["m"]

            new_pos = pos + vel * self.dt + 0.5 * accel * self.dt ** 2
            new_vel = vel + accel * self.dt
            new_state = jnp.concatenate([new_pos, new_vel])

            trajectory.append(new_state)
            current_state = new_state

        return jnp.array(trajectory)

    def objective_function(self, decision_vector: np.ndarray) -> float:
        """
        MPC objective function: reference tracking + control effort.

        Args:
            decision_vector: Flattened control sequence

        Returns:
            objective_value: Scalar cost
        """
        control_sequence = self.extract_control_sequence(decision_vector)

        # Predict ego trajectory
        trajectory = self.predict_ego_trajectory(control_sequence)

        # Reference tracking cost (distance to goal)
        tracking_cost = 0.0
        for state in trajectory[1:]:  # Skip initial state
            pos = state[:2]
            tracking_cost += jnp.sum((pos - self.ego_goal) ** 2)

        # Control effort cost
        control_cost = jnp.sum(control_sequence ** 2)

        total_cost = (self.ref_weight * tracking_cost +
                      self.control_weight * control_cost)

        return float(total_cost)

    def constraint_function(self, decision_vector: np.ndarray) -> np.ndarray:
        """
        CBF safety constraints over the horizon.

        Args:
            decision_vector: Flattened control sequence

        Returns:
            constraint_values: Array of constraint values (≥ 0 for feasibility)
        """
        control_sequence = self.extract_control_sequence(decision_vector)

        try:
            # Predict graph sequence
            predicted_graphs = self.graph_predictor.predict_graphs_horizon(
                self.initial_graph, control_sequence
            )

            # Evaluate CBF constraints at each step
            constraint_values = []
            all_graphs = [self.initial_graph] + predicted_graphs

            for i, graph in enumerate(all_graphs[:-1]):  # Skip last graph
                control = control_sequence[i] if i < len(control_sequence) else np.zeros(2)

                # Get CBF constraint components: ḣ + α*h ≥ 0
                drift_term, control_coeffs, h_value = self.cbf_evaluator.evaluate_h_dot_constraint(
                    graph, control, self.alpha
                )

                # Constraint value: drift + control_coeffs @ control + alpha * h
                constraint_val = (drift_term +
                                  np.dot(control_coeffs, control) +
                                  self.alpha * h_value)

                constraint_values.append(constraint_val)

            return np.array(constraint_values)

        except Exception as e:
            print(f"Error in constraint evaluation: {e}")
            # Return large negative values to indicate constraint violation
            return np.full(self.horizon, -1000.0)

    def jacobian_objective(self, decision_vector: np.ndarray) -> np.ndarray:
        """Compute Jacobian of objective function using JAX."""

        # Convert to JAX-compatible function
        def objective_jax(u_flat):
            control_seq = u_flat.reshape((self.horizon, self.control_dim))

            # Initial state
            agent_states = self.initial_graph.type_states(type_idx=0, n_type=self.env.num_agents)
            ego_state = agent_states[self.ego_agent_idx]

            # Predict trajectory
            trajectory = [ego_state]
            current_state = ego_state

            for control in control_seq:
                pos = current_state[:2]
                vel = current_state[2:]
                accel = control / self.env._params["m"]

                new_pos = pos + vel * self.dt + 0.5 * accel * self.dt ** 2
                new_vel = vel + accel * self.dt
                new_state = jnp.concatenate([new_pos, new_vel])

                trajectory.append(new_state)
                current_state = new_state

            trajectory = jnp.array(trajectory)

            # Costs
            tracking_cost = jnp.sum((trajectory[1:, :2] - self.ego_goal) ** 2)
            control_cost = jnp.sum(control_seq ** 2)

            return self.ref_weight * tracking_cost + self.control_weight * control_cost

        # Compute gradient
        grad_fn = jax.grad(objective_jax)
        gradient = grad_fn(jnp.array(decision_vector))

        return np.array(gradient)

    def solve_single_step(self,
                          initial_guess: np.ndarray = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Solve single MPC optimization step.

        Args:
            initial_guess: Initial control sequence guess
            max_iterations: Maximum SQP iterations
            tolerance: Convergence tolerance

        Returns:
            result_dict: Optimization results and statistics
        """
        print(f"\n{'=' * 60}")
        print("SOLVING SINGLE MPC STEP")
        print("=" * 60)

        # Initialize decision variables
        if initial_guess is None:
            initial_guess = np.zeros(self.decision_vars)  # Zero control initial guess

        print(f"Initial guess shape: {initial_guess.shape}")
        print(f"Decision variables: {self.decision_vars}")

        # Set up bounds
        bounds = [self.control_bounds] * self.decision_vars

        # Set up nonlinear constraints
        constraints = NonlinearConstraint(
            fun=self.constraint_function,
            lb=0.0,  # CBF constraints: g(x) ≥ 0
            ub=np.inf,
            jac='2-point'  # Numerical Jacobian for constraints
        )

        # Solve optimization
        start_time = time.time()

        try:
            result = minimize(
                fun=self.objective_function,
                x0=initial_guess,
                method='SLSQP',  # Sequential Least Squares Programming
                jac=self.jacobian_objective,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': max_iterations,
                    'ftol': tolerance,
                    'disp': True
                }
            )

            solve_time = time.time() - start_time

            # Extract results
            optimal_control = self.extract_control_sequence(result.x)

            # Evaluate constraints at solution
            constraint_vals = self.constraint_function(result.x)
            min_constraint = np.min(constraint_vals)

            # Statistics
            self.optimization_stats['iterations'].append(result.nit)
            self.optimization_stats['solve_times'].append(solve_time)
            self.optimization_stats['objective_values'].append(result.fun)
            self.optimization_stats['constraint_violations'].append(min_constraint)

            print(f"\n--- OPTIMIZATION RESULTS ---")
            print(f"Success: {result.success}")
            print(f"Status: {result.message}")
            print(f"Iterations: {result.nit}")
            print(f"Solve time: {solve_time:.3f}s")
            print(f"Objective value: {result.fun:.6f}")
            print(f"Min constraint value: {min_constraint:.6f}")
            print(f"Constraint violation: {'NO' if min_constraint >= -1e-6 else 'YES'}")

            if result.success:
                print(f"\n--- OPTIMAL CONTROL SEQUENCE ---")
                for i, control in enumerate(optimal_control):
                    print(f"Step {i}: u = [{control[0]:.4f}, {control[1]:.4f}]")

            return {
                'success': result.success,
                'optimal_control': optimal_control,
                'objective_value': result.fun,
                'constraint_values': constraint_vals,
                'solve_time': solve_time,
                'iterations': result.nit,
                'raw_result': result
            }

        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'success': False,
                'error': str(e),
                'solve_time': time.time() - start_time
            }

    def solve_multiple_steps(self,
                             num_steps: int = 4,
                             max_iterations: int = 50) -> List[Dict[str, Any]]:
        """
        Solve multiple MPC steps in sequence.

        Args:
            num_steps: Number of MPC steps to solve
            max_iterations: Max iterations per step

        Returns:
            results_list: List of optimization results for each step
        """
        print(f"\n{'=' * 80}")
        print(f"SOLVING {num_steps} MPC STEPS IN SEQUENCE")
        print("=" * 80)

        results = []
        current_graph = self.initial_graph

        for step in range(num_steps):
            print(f"\n{'-' * 40} STEP {step + 1}/{num_steps} {'-' * 40}")

            # Update controller with current graph
            self.initial_graph = current_graph

            # Solve current step
            step_result = self.solve_single_step(max_iterations=max_iterations)
            results.append(step_result)

            if not step_result['success']:
                print(f"Step {step + 1} failed, stopping sequence")
                break

            # Apply first control input and update graph
            optimal_control = step_result['optimal_control']
            first_control = optimal_control[0]

            print(f"Applying control: [{first_control[0]:.4f}, {first_control[1]:.4f}]")

            try:
                # Predict next graph state
                next_graphs = self.graph_predictor.predict_graphs_horizon(
                    current_graph, first_control.reshape(1, -1)
                )
                current_graph = next_graphs[0]
                print(f"✓ Graph updated for next step")

            except Exception as e:
                print(f"✗ Failed to update graph: {e}")
                break

        # Summary statistics
        successful_steps = sum(1 for r in results if r['success'])
        total_time = sum(r.get('solve_time', 0) for r in results)
        avg_iterations = np.mean([r.get('iterations', 0) for r in results if r['success']])

        print(f"\n{'=' * 80}")
        print("SEQUENCE SUMMARY")
        print("=" * 80)
        print(f"Successful steps: {successful_steps}/{num_steps}")
        print(f"Total solve time: {total_time:.3f}s")
        print(f"Average iterations: {avg_iterations:.1f}")

        return results


def create_test_scenario_standalone():
    """Standalone test scenario creation function."""
    from gcbfplus.env.double_integrator import DoubleIntegrator

    # Environment parameters
    env_params = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.5],
        "n_obs": 1,
        "m": 0.1,
    }

    # Create environment
    env = DoubleIntegrator(
        num_agents=2,
        area_size=2.0,
        max_step=256,
        max_travel=None,
        dt=0.03,
        params=env_params
    )

    # Create obstacles
    obs_positions = jnp.array([[1.0, 1.2], [0.6, 1.2]])
    obs_lengths_x = jnp.array([0.6, 0.1])
    obs_lengths_y = jnp.array([0.1, 0.4])
    obs_thetas = jnp.array([0.0, -jnp.pi / 8])

    obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)

    # Create agents
    ego_state = jnp.array([0.8, 1.0, 0.2, -0.1])  # Slower initial velocity
    other_state = jnp.array([0.2, 0.2, 0, 0])
    agent_states = jnp.array([ego_state, other_state])

    # Goals
    goal_states = jnp.array([
        [1.8, 1.8, 0.0, 0.0],  # Ego goal - further away
        [1.5, 1.0, 0.0, 0.0]  # Other agent goal
    ])

    # Create graph
    env_state = env.EnvState(agent_states, goal_states, obstacles)
    initial_graph = env.get_graph(env_state)

    print(f"Test scenario created:")
    print(
        f"  Ego: pos=({ego_state[0]:.3f}, {ego_state[1]:.3f}), goal=({goal_states[0, 0]:.3f}, {goal_states[0, 1]:.3f})")
    print(f"  Distance to goal: {jnp.linalg.norm(ego_state[:2] - goal_states[0, :2]):.3f}")

    return env, initial_graph


def test_single_step_optimization():
    """Test single step MPC optimization."""
    print("TESTING SINGLE STEP MPC OPTIMIZATION")

    # Model path - update this to your trained model
    model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorMPC/gcbf+/seed0_20250626023916"

    if not pathlib.Path(model_path).exists():
        print(f"Model path does not exist: {model_path}")
        print("Please update the model_path variable")
        return

    # Create test scenario
    env, initial_graph = create_test_scenario_standalone()

    # Initialize controller
    controller = SQPMPCController(
        model_path=model_path,
        env=env,
        initial_graph=initial_graph,
        ego_agent_idx=0,
        horizon=4,
        alpha=1.0,
        control_bounds=(-2, 2),  # Reasonable control bounds
        reference_tracking_weight=10.0,
        control_effort_weight=0.1
    )

    # Test optimization
    result = controller.solve_single_step(max_iterations=50)

    if result['success']:
        print("\n✓ Single step optimization successful!")
        print("✓ Ready for multi-step testing")
    else:
        print("\n✗ Single step optimization failed")
        print("  Check constraints and bounds")

    return result


def test_multi_step_optimization():
    """Test multi-step MPC optimization."""
    print("\nTESTING MULTI-STEP MPC OPTIMIZATION")

    # Model path - update this to your trained model
    model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorMPC/gcbf+/seed0_20250626023916"

    if not pathlib.Path(model_path).exists():
        print(f"Model path does not exist: {model_path}")
        return

    # Create test scenario
    env, initial_graph = create_test_scenario_standalone()

    # Initialize controller
    controller = SQPMPCController(
        model_path=model_path,
        env=env,
        initial_graph=initial_graph,
        ego_agent_idx=0,
        horizon=3,  # Shorter horizon for faster testing
        alpha=1.0,
        control_bounds=(-2, 2),
        reference_tracking_weight=5.0,
        control_effort_weight=0.1
    )

    # Test multi-step optimization
    results = controller.solve_multiple_steps(num_steps=4, max_iterations=30)

    successful = sum(1 for r in results if r['success'])
    print(f"\n{'=' * 60}")
    print(f"MULTI-STEP TEST COMPLETED: {successful}/{len(results)} successful")

    if successful > 0:
        print("✓ Multi-step MPC working!")
        print("✓ Ready for deployment")
    else:
        print("✗ Multi-step optimization needs tuning")

    return results


def main():
    """Main test function."""
    print("=" * 100)
    print("SQP-BASED MPC CONTROLLER TEST")
    print("=" * 100)

    # Test single step first
    print("\n" + "=" * 50)
    print("PHASE 1: SINGLE STEP TEST")
    print("=" * 50)

    try:
        single_result = test_single_step_optimization()

        if single_result and single_result['success']:
            # Test multi-step if single step works
            print("\n" + "=" * 50)
            print("PHASE 2: MULTI-STEP TEST")
            print("=" * 50)

            # multi_results = test_multi_step_optimization()
            #
            # print("\n" + "=" * 100)
            # print("ALL TESTS COMPLETED")
            # print("=" * 100)

        else:
            print("\nSingle step test failed - skipping multi-step test")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()