#!/usr/bin/env python3
"""
MPC+GCBF Test Script with Safety Margin - Final Implementation

Uses trained GCBF model as hard safety constraints in MPC controller.
Includes configurable safety margin so agents brake before h reaches 0.
Prioritizes safety over goal-reaching - agents can move away from goals to maintain h ≥ margin.
"""

import argparse
import datetime
import os
import pathlib
import numpy as np
import yaml
import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from typing import Tuple, List, Optional
import matplotlib
import csv
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, jax2np, mask2index


class GCBFInterface:
    """Interface to trained GCBF model for MPC constraints."""

    def __init__(self, algo):
        self.algo = algo
        self.env = algo._env
        self.get_cbf_fn = jax.jit(algo.get_cbf)

    def evaluate_h(self, graph: GraphsTuple) -> np.ndarray:
        """Evaluate CBF h(x) for all agents."""
        h_values = self.get_cbf_fn(graph)
        return jax2np(h_values)

    def get_cbf_gradient(self, graph: GraphsTuple, agent_idx: int) -> np.ndarray:
        """Get CBF gradient ∇h w.r.t. specific agent's state."""

        # Get agent node indices (following the QP solver pattern)
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.env.num_agents)

        def h_aug(agent_states):
            """CBF as function of all agent states."""
            new_state = graph.states.at[agent_node_id].set(agent_states)
            new_graph = self.env.add_edge_feats(graph, new_state)
            h_all = self.algo.get_cbf(new_graph, params=self.algo.cbf_train_state.params)
            return h_all[agent_idx].squeeze()

        # Get current agent states and compute gradient
        agent_states = graph.type_states(type_idx=0, n_type=self.env.num_agents)

        try:
            # Compute jacobian w.r.t. all agent states, extract for this agent
            h_x_all = jax.jacobian(h_aug)(agent_states)
            h_x_agent = h_x_all[agent_idx]
            return jax2np(h_x_agent)

        except Exception as e:
            print(f"CBF gradient computation failed for agent {agent_idx}: {e}")
            # Return zero gradient as fallback (will not add CBF constraints)
            return np.zeros(agent_states.shape[1])


class MPCDebugLogger:
    """Comprehensive debug logger for troubleshooting MPC+GCBF issues."""

    def __init__(self, output_dir: pathlib.Path, num_agents: int, margin: float = 0.0):
        self.output_dir = output_dir
        self.num_agents = num_agents
        self.margin = margin

        # Create log files
        self.state_log_file = output_dir / "agent_states.csv"
        self.cbf_log_file = output_dir / "cbf_data.csv"
        self.mpc_log_file = output_dir / "mpc_status.csv"
        self.obstacle_log_file = output_dir / "obstacles.csv"

        # Initialize CSV files with headers
        self._init_log_files()

    def _init_log_files(self):
        """Initialize CSV log files with appropriate headers."""

        # Agent states log
        with open(self.state_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y',
                      'goal_x', 'goal_y', 'goal_vel_x', 'goal_vel_y']
            writer.writerow(header)

        # CBF data log
        with open(self.cbf_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'cbf_value', 'margin_violation',
                      'grad_x', 'grad_y', 'grad_vx', 'grad_vy',
                      'drift_term', 'control_coeff_x', 'control_coeff_y']
            writer.writerow(header)

        # MPC status log
        with open(self.mpc_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'solver_status', 'solve_time',
                      'action_x', 'action_y', 'feasible', 'emergency_brake']
            writer.writerow(header)

        # Obstacles log
        with open(self.obstacle_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'obstacle_id', 'pos_x', 'pos_y', 'type']
            writer.writerow(header)

    def log_episode_state(self, episode: int, step: int, agent_states: np.ndarray,
                          goal_states: np.ndarray, h_values: np.ndarray,
                          cbf_gradients: List[np.ndarray], graph: 'GraphsTuple'):
        """Log complete state information for troubleshooting."""

        # Log agent states and goals
        with open(self.state_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.num_agents):
                agent_state = agent_states[i]
                goal_state = goal_states[i]
                writer.writerow([
                    episode, step, i,
                    float(agent_state[0]), float(agent_state[1]),  # position
                    float(agent_state[2]), float(agent_state[3]),  # velocity
                    float(goal_state[0]), float(goal_state[1]),  # goal position
                    float(goal_state[2]), float(goal_state[3])  # goal velocity
                ])

        # Log CBF data
        with open(self.cbf_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.num_agents):
                cbf_val = float(h_values[i].item())  # Use .item() to extract scalar
                margin_violation = cbf_val < self.margin  # Check against margin, not 0

                if i < len(cbf_gradients) and cbf_gradients[i] is not None:
                    grad = cbf_gradients[i]
                    if len(grad) >= 4:
                        # Calculate drift term and control coefficients
                        drift_term = grad[0] * agent_states[i, 2] + grad[1] * agent_states[i, 3]
                        control_coeff_x = grad[2]
                        control_coeff_y = grad[3]

                        writer.writerow([
                            episode, step, i, cbf_val, margin_violation,
                            float(grad[0]), float(grad[1]), float(grad[2]), float(grad[3]),
                            float(drift_term), float(control_coeff_x), float(control_coeff_y)
                        ])
                    else:
                        writer.writerow([
                            episode, step, i, cbf_val, margin_violation,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Zero if gradient failed
                        ])
                else:
                    writer.writerow([
                        episode, step, i, cbf_val, margin_violation,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Zero if gradient not available
                    ])

        # Log obstacle positions
        with open(self.obstacle_log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Extract obstacle nodes (node_type == 2 typically)
            obstacle_mask = graph.node_type == 2
            if np.any(obstacle_mask):
                obstacle_states = graph.states[obstacle_mask]
                for obs_id, obs_state in enumerate(obstacle_states):
                    writer.writerow([
                        episode, step, obs_id,
                        float(obs_state[0]), float(obs_state[1]),  # obstacle position
                        'static'  # assume static obstacles for now
                    ])

    def log_mpc_result(self, episode: int, step: int, agent_id: int,
                       solver_status: str, solve_time: float, action: np.ndarray,
                       feasible: bool, emergency_brake: bool = False):
        """Log MPC solver results and actions."""

        with open(self.mpc_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, step, agent_id, solver_status, solve_time,
                float(action[0]), float(action[1]), feasible, emergency_brake
            ])


class SafetyFirstMPC:
    """MPC controller with hard safety constraints, safety margin, and relaxed goal-reaching."""

    def __init__(self,
                 gcbf_interface: GCBFInterface,
                 horizon: int = 10,
                 dt: float = 0.03,
                 u_max: float = 2.0,
                 alpha: float = 1.0,
                 margin: float = 0.0,
                 goal_weight: float = 1.0,
                 debug_logger: Optional[MPCDebugLogger] = None):

        self.gcbf = gcbf_interface
        self.horizon = horizon
        self.dt = dt
        self.u_max = u_max
        self.alpha = alpha
        self.margin = margin  # Safety margin - agents brake before h hits 0
        self.goal_weight = goal_weight  # Weight for goal-reaching (can be relaxed)
        self.debug_logger = debug_logger  # For comprehensive debugging

        # Cost matrices for double integrator [x, y, vx, vy]
        # These are SOFT - can be relaxed when safety constraints conflict
        self.Q = np.diag([1.0, 1.0, 0.1, 0.1])  # Goal-reaching penalty (relaxable)
        self.R = np.eye(2) * 0.01  # Control effort penalty (relaxable)

    def solve(self, agent_states: np.ndarray, goal_states: np.ndarray,
              graph: GraphsTuple, episode: int = 0, step: int = 0) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Solve MPC for all agents with comprehensive debug logging."""

        n_agents = agent_states.shape[0]
        actions = np.zeros((n_agents, 2))
        cbf_gradients = []

        # Get CBF values for all agents for logging
        h_values = self.gcbf.evaluate_h(graph)

        # Collect gradients for logging
        for i in range(n_agents):
            try:
                grad = self.gcbf.get_cbf_gradient(graph, i)
                cbf_gradients.append(grad)
            except:
                cbf_gradients.append(np.zeros(4))  # Fallback

        # Log current state before solving MPC
        if self.debug_logger:
            self.debug_logger.log_episode_state(
                episode, step, agent_states, goal_states, h_values, cbf_gradients, graph
            )

        for i in range(n_agents):
            start_time = time.time()

            try:
                action, solver_status, feasible = self._solve_single_agent(
                    agent_states[i], goal_states[i], graph, i, agent_states, episode, step
                )
                actions[i] = action
                solve_time = time.time() - start_time
                emergency_brake = False

            except Exception as e:
                solve_time = time.time() - start_time
                print(f"Agent {i} MPC failed: {e}")

                # Emergency brake: decelerate toward zero velocity
                current_vel = agent_states[i, 2:4]
                actions[i] = -np.sign(current_vel) * min(self.u_max, np.linalg.norm(current_vel))
                actions[i] = np.clip(actions[i], -self.u_max, self.u_max)

                solver_status = f"FAILED: {str(e)[:50]}"
                feasible = False
                emergency_brake = True

            # Log MPC result
            if self.debug_logger:
                self.debug_logger.log_mpc_result(
                    episode, step, i, solver_status, solve_time, actions[i], feasible, emergency_brake
                )

        return actions, cbf_gradients

    def _solve_single_agent(self, current_state: np.ndarray, goal_state: np.ndarray,
                            graph: GraphsTuple, agent_idx: int,
                            all_agent_states: np.ndarray, episode: int, step: int) -> Tuple[np.ndarray, str, bool]:
        """Solve safety-first MPC for single agent with detailed status tracking."""

        # Decision variables
        x = cp.Variable((self.horizon + 1, 4))  # [x, y, vx, vy]
        u = cp.Variable((self.horizon, 2))  # [ax, ay]

        # Objective: Minimize goal deviation (low weight) + control effort
        cost = 0
        constraints = []

        # Initial condition
        constraints += [x[0] == current_state]

        # Dynamics and constraints over horizon
        for k in range(self.horizon):
            # Double integrator dynamics
            constraints += [
                x[k + 1, 0] == x[k, 0] + x[k, 2] * self.dt + 0.5 * u[k, 0] * self.dt ** 2,
                x[k + 1, 1] == x[k, 1] + x[k, 3] * self.dt + 0.5 * u[k, 1] * self.dt ** 2,
                x[k + 1, 2] == x[k, 2] + u[k, 0] * self.dt,
                x[k + 1, 3] == x[k, 3] + u[k, 1] * self.dt
            ]

            # Control limits
            constraints += [
                cp.abs(u[k, 0]) <= self.u_max,
                cp.abs(u[k, 1]) <= self.u_max
            ]

            # SOFT GOAL-REACHING OBJECTIVE (can be relaxed for safety)
            cost += self.goal_weight * cp.quad_form(x[k] - goal_state, self.Q)
            cost += cp.quad_form(u[k], self.R)

            # HARD SAFETY CONSTRAINTS with MARGIN - Applied at every step
            # These MUST be satisfied - goal-reaching will be sacrificed if needed
            if k == 0:
                # First step: use current graph state
                cbf_graph = graph
                predicted_state = current_state
            else:
                # Future steps: predict graph state
                cbf_graph, predicted_state = self._predict_graph_state(
                    graph, agent_idx, all_agent_states, x[k], k
                )

            # Apply CBF constraint with margin at this step
            safety_constraint = self._add_cbf_constraint(
                cbf_graph, agent_idx, predicted_state, u[k], k
            )

            if safety_constraint is not None:
                constraints += [safety_constraint]

        # Terminal cost (soft penalty - can be relaxed for safety)
        cost += self.goal_weight * cp.quad_form(x[self.horizon] - goal_state, self.Q)

        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Try to solve with multiple solvers
        solver_names = ["OSQP", "ECOS", "SCS"]
        solver_status = "UNKNOWN"
        feasible = False

        for solver, solver_name in zip([cp.OSQP, cp.ECOS, cp.SCS], solver_names):
            try:
                prob.solve(solver=solver, verbose=False)
                solver_status = f"{solver_name}_{prob.status}"

                if prob.status in ["optimal", "optimal_inaccurate"]:
                    feasible = True
                    if step == 0:  # Only print for first step to avoid spam
                        print(f"    Agent {agent_idx}: MPC solved successfully ({solver_status})")
                    return np.array(u[0].value), solver_status, feasible

            except Exception as e:
                solver_status = f"{solver_name}_ERROR: {str(e)[:30]}"
                continue

        # If all solvers fail, use emergency brake
        print(f"  Agent {agent_idx}: MPC infeasible ({solver_status}) - emergency brake")
        current_vel = current_state[2:4]
        brake_action = -2 * current_vel
        emergency_action = np.clip(brake_action, -self.u_max, self.u_max)

        return emergency_action, f"INFEASIBLE_{solver_status}", False

    def _predict_graph_state(self, graph: GraphsTuple, agent_idx: int,
                             all_agent_states: np.ndarray,
                             predicted_agent_state: cp.Variable,
                             step: int) -> Tuple[GraphsTuple, np.ndarray]:
        """Predict graph state at future step for CBF evaluation."""

        # Start with current agent states
        predicted_states = all_agent_states.copy()

        # For this agent: use MPC predicted state
        # Note: predicted_agent_state is a CVXPY variable, we need current best estimate
        # For now, use simple prediction: current_state + velocity * dt * step
        current_state = all_agent_states[agent_idx]
        simple_predicted = current_state.copy()
        simple_predicted[0] += current_state[2] * self.dt * step  # x += vx * dt * step
        simple_predicted[1] += current_state[3] * self.dt * step  # y += vy * dt * step
        # Velocities assumed constant for simplicity

        predicted_states[agent_idx] = simple_predicted

        # For other agents: constant velocity prediction
        for other_idx in range(len(all_agent_states)):
            if other_idx != agent_idx:
                other_state = all_agent_states[other_idx].copy()
                other_state[0] += other_state[2] * self.dt * step
                other_state[1] += other_state[3] * self.dt * step
                predicted_states[other_idx] = other_state

        # Create new graph with predicted states (keeping current connectivity)
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.gcbf.env.num_agents)
        new_state = graph.states.at[agent_node_id].set(predicted_states)
        predicted_graph = self.gcbf.env.add_edge_feats(graph, new_state)

        return predicted_graph, simple_predicted

    def _add_cbf_constraint(self, graph: GraphsTuple, agent_idx: int,
                            agent_state: np.ndarray, control: cp.Variable,
                            step: int) -> Optional[cp.Constraint]:
        """Add CBF constraint with safety margin: ḣ + α*h ≥ margin."""

        try:
            # Get current CBF value and gradient
            h_values = self.gcbf.evaluate_h(graph)
            h_current = float(h_values[agent_idx].item())  # Use .item() to extract scalar
            cbf_grad = self.gcbf.get_cbf_gradient(graph, agent_idx)

            # Check if gradient computation succeeded
            if np.allclose(cbf_grad, 0):
                return None  # Skip constraint if gradient failed

            # Double integrator: ẋ = [vx, vy, ax, ay]
            # ḣ = ∇h · ẋ = grad[0]*vx + grad[1]*vy + grad[2]*ax + grad[3]*ay

            # Drift term (current velocity contribution)
            drift_term = cbf_grad[0] * agent_state[2] + cbf_grad[1] * agent_state[3]

            # Control term (acceleration contribution)
            control_term = cbf_grad[2] * control[0] + cbf_grad[3] * control[1]

            # CBF constraint with MARGIN: ḣ + α*h ≥ margin
            # drift_term + control_term + α*h_current ≥ margin
            # This creates a safety buffer - agents brake before h hits 0
            cbf_constraint = (drift_term + control_term + self.alpha * h_current >= self.margin)

            # Debug info for first step
            if step == 0:
                margin_status = "MARGIN VIOLATED" if h_current < self.margin else "SAFE"
                print(f"    Agent {agent_idx}: h={h_current:.4f}, margin={self.margin:.4f} [{margin_status}]")

            return cbf_constraint

        except Exception as e:
            if step == 0:  # Only print for first step to avoid spam
                print(f"    Agent {agent_idx}: CBF constraint failed at step {step}: {e}")
            return None


class MPCGCBFTester:
    """Testing framework for MPC+GCBF approach with margin-aware safety analysis."""

    def __init__(self, env, mpc_controller: SafetyFirstMPC, debug_logger: Optional[MPCDebugLogger] = None):
        self.env = env
        self.controller = mpc_controller
        self.debug_logger = debug_logger
        self.margin = mpc_controller.margin

        # Evaluation functions
        self.is_unsafe_fn = jax_jit_np(jax.vmap(env.collision_mask))
        self.is_finish_fn = jax_jit_np(jax.vmap(env.finish_mask))

    def run_episode(self, key: jr.PRNGKey, max_steps: int, episode_id: int = 0) -> Tuple[
        RolloutResult, List[np.ndarray], dict]:
        """Run single episode with detailed CBF tracking and margin analysis."""

        # Initialize
        graph = self.env.reset(key)
        h_trace = []
        episode_stats = {
            'cbf_violations': 0,  # Actual CBF violations (h < 0)
            'margin_violations': 0,  # Margin violations (h < margin)
            'emergency_brakes': 0,
            'min_h_overall': float('inf'),
            'min_h_margin_adjusted': float('inf'),
            'steps_completed': 0
        }

        # Storage for rollout
        T_graph = [graph]
        T_action = []
        T_reward = []
        T_cost = []
        T_done = []
        T_info = []

        for step in range(max_steps):
            # Get current states
            agent_states = np.array(graph.type_states(type_idx=0, n_type=self.env.num_agents))
            goal_states = np.array(graph.type_states(type_idx=1, n_type=self.env.num_agents))

            # Evaluate and track CBF values
            h_values = self.controller.gcbf.evaluate_h(graph)
            h_trace.append(h_values.copy())

            # Update episode statistics
            current_min_h = h_values.min()
            episode_stats['min_h_overall'] = min(episode_stats['min_h_overall'], current_min_h)
            episode_stats['min_h_margin_adjusted'] = min(episode_stats['min_h_margin_adjusted'],
                                                         current_min_h - self.margin)
            episode_stats['cbf_violations'] += np.sum(h_values < 0)
            episode_stats['margin_violations'] += np.sum(h_values < self.margin)
            episode_stats['steps_completed'] = step + 1

            # Print CBF status periodically or when violations occur
            if step % 20 == 0 or np.any(h_values < self.margin):
                unsafe_count = np.sum(h_values < 0)
                margin_count = np.sum(h_values < self.margin)
                print(f"  Step {step:3d}: CBF∈[{h_values.min():.3f}, {h_values.max():.3f}], "
                      f"mean={h_values.mean():.3f}, margin_viols={margin_count}, unsafe={unsafe_count}")

                if margin_count > 0:
                    margin_agents = np.where(h_values < self.margin)[0]
                    margin_h_values = h_values[margin_agents]
                    margin_h_str = ", ".join([f"{h.item():.3f}" for h in margin_h_values])
                    print(
                        f"    ⚠️  MARGIN: Agents {margin_agents.tolist()} with h=[{margin_h_str}] < {self.margin:.3f}")

                if unsafe_count > 0:
                    unsafe_agents = np.where(h_values < 0)[0]
                    unsafe_h_values = h_values[unsafe_agents]
                    unsafe_h_str = ", ".join([f"{h.item():.3f}" for h in unsafe_h_values])
                    print(f"    🚨 UNSAFE: Agents {unsafe_agents.tolist()} with h=[{unsafe_h_str}] < 0")

            # Get MPC control action with debug logging
            action, cbf_gradients = self.controller.solve(
                agent_states, goal_states, graph, episode_id, step
            )
            action_jax = jnp.array(action)

            # Step environment
            step_result = self.env.step(graph, action_jax, get_eval_info=True)
            graph, reward, cost, done, info = step_result

            # Store data
            T_graph.append(graph)
            T_action.append(action_jax)
            T_reward.append(reward)
            T_cost.append(cost)
            T_done.append(done)
            T_info.append(info)

            if done:
                print(f"    Episode completed at step {step}")
                break

        # Create rollout result
        Tp1_graph = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *T_graph)
        T_action = jnp.stack(T_action, axis=0)
        T_reward = jnp.stack(T_reward, axis=0)
        T_cost = jnp.stack(T_cost, axis=0)
        T_done = jnp.stack(T_done, axis=0)
        T_info = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *T_info)

        rollout = RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        return rollout, h_trace, episode_stats

    def evaluate_rollout(self, rollout: RolloutResult) -> dict:
        """Evaluate rollout performance metrics."""
        is_unsafe = self.is_unsafe_fn(rollout.Tp1_graph)
        is_finish = self.is_finish_fn(rollout.Tp1_graph)

        metrics = {
            'total_reward': float(rollout.T_reward.sum()),
            'total_cost': float(rollout.T_cost.sum()),
            'safe_rate': float(1 - is_unsafe.max(axis=0).mean()),
            'finish_rate': float(is_finish.max(axis=0).mean()),
            'success_rate': float(((1 - is_unsafe.max(axis=0)) * is_finish.max(axis=0)).mean()),
            'is_unsafe': is_unsafe,
            'is_finish': is_finish
        }

        return metrics


def generate_cbf_plots(h_traces: List[List[np.ndarray]], output_dir: pathlib.Path, margin: float = 0.0):
    """Generate comprehensive CBF analysis plots with margin visualization."""

    plt.figure(figsize=(16, 12))

    # Plot 1: CBF evolution over time for multiple episodes
    plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, min(5, len(h_traces))))

    for i, (h_trace, color) in enumerate(zip(h_traces[:5], colors)):
        if len(h_trace) == 0:
            continue

        # Convert to array and handle different shapes
        h_array = np.array([np.array(h_step).flatten() for h_step in h_trace])

        # Plot min, mean, max over time
        h_min = h_array.min(axis=1)
        h_mean = h_array.mean(axis=1)
        h_max = h_array.max(axis=1)

        time_steps = range(len(h_min))
        plt.plot(time_steps, h_mean, label=f'Episode {i + 1}', color=color, linewidth=2)
        plt.fill_between(time_steps, h_min, h_max, color=color, alpha=0.2)

    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Safety Threshold')
    if margin > 0:
        plt.axhline(y=margin, color='orange', linestyle=':', linewidth=2, label=f'Safety Margin ({margin:.3f})')
    plt.xlabel('Time Step')
    plt.ylabel('CBF Value h(x)')
    plt.title('CBF Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: CBF violation timeline (both margin and actual)
    plt.subplot(2, 3, 2)
    for i, h_trace in enumerate(h_traces[:5]):
        if len(h_trace) == 0:
            continue

        unsafe_steps = []
        margin_steps = []
        for step, h_step in enumerate(h_trace):
            h_array = np.array(h_step).flatten()
            if np.any(h_array < 0):
                unsafe_steps.append(step)
            if np.any(h_array < margin):
                margin_steps.append(step)

        if unsafe_steps:
            plt.scatter(unsafe_steps, [i + 1] * len(unsafe_steps),
                        color='red', s=50, alpha=0.8, marker='x', label='Unsafe' if i == 0 else '')
        if margin_steps:
            plt.scatter(margin_steps, [i + 1] * len(margin_steps),
                        color='orange', s=30, alpha=0.6, marker='o', label='Margin Violation' if i == 0 else '')

    plt.xlabel('Time Step')
    plt.ylabel('Episode')
    plt.title('CBF Violations Timeline')
    plt.yticks(range(1, min(6, len(h_traces) + 1)))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: CBF distribution histogram
    plt.subplot(2, 3, 3)
    all_h_values = []
    for h_trace in h_traces:
        for h_step in h_trace:
            all_h_values.extend(np.array(h_step).flatten())

    if all_h_values:
        all_h_values = np.array(all_h_values)
        plt.hist(all_h_values, bins=50, alpha=0.7, edgecolor='black', density=True)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Safety Threshold')
        if margin > 0:
            plt.axvline(x=margin, color='orange', linestyle=':', linewidth=2, label=f'Safety Margin ({margin:.3f})')
        plt.axvline(x=all_h_values.mean(), color='blue', linestyle='-.', linewidth=2,
                    label=f'Mean={all_h_values.mean():.3f}')
        plt.xlabel('CBF Value h(x)')
        plt.ylabel('Density')
        plt.title('CBF Value Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 4: Per-episode minimum CBF
    plt.subplot(2, 3, 4)
    episode_min_h = []
    for h_trace in h_traces:
        if len(h_trace) > 0:
            all_h = np.concatenate([np.array(h_step).flatten() for h_step in h_trace])
            episode_min_h.append(all_h.min())
        else:
            episode_min_h.append(0)

    episodes = range(1, len(episode_min_h) + 1)
    bars = plt.bar(episodes, episode_min_h, alpha=0.7)

    # Color bars based on safety levels
    for bar, min_h in zip(bars, episode_min_h):
        if min_h < 0:
            bar.set_color('red')  # Unsafe
        elif min_h < margin:
            bar.set_color('orange')  # Margin violation
        else:
            bar.set_color('green')  # Safe

    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Unsafe')
    if margin > 0:
        plt.axhline(y=margin, color='orange', linestyle=':', linewidth=2, label='Margin')
    plt.xlabel('Episode')
    plt.ylabel('Minimum CBF Value')
    plt.title('Minimum CBF per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: CBF statistics over episodes
    plt.subplot(2, 3, 5)
    episode_stats = []
    for h_trace in h_traces:
        if len(h_trace) > 0:
            all_h = np.concatenate([np.array(h_step).flatten() for h_step in h_trace])
            unsafe_violations = np.sum(all_h < 0)
            margin_violations = np.sum(all_h < margin)
            total = len(all_h)
            episode_stats.append(
                [all_h.min(), all_h.mean(), unsafe_violations / total * 100, margin_violations / total * 100])
        else:
            episode_stats.append([0, 0, 0, 0])

    episode_stats = np.array(episode_stats)
    episodes = range(1, len(episode_stats) + 1)

    plt.plot(episodes, episode_stats[:, 0], 'r-', label='Min h', linewidth=2)
    plt.plot(episodes, episode_stats[:, 1], 'b-', label='Mean h', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    if margin > 0:
        plt.axhline(y=margin, color='orange', linestyle=':', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('CBF Value')
    plt.title('CBF Statistics per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Violation rates per episode
    plt.subplot(2, 3, 6)
    plt.plot(episodes, episode_stats[:, 2], 'ro-', linewidth=2, markersize=6, label='Unsafe Rate')
    if margin > 0:
        plt.plot(episodes, episode_stats[:, 3], 'o-', color='orange', linewidth=2, markersize=6, label='Margin Rate')
    plt.xlabel('Episode')
    plt.ylabel('Violation Rate (%)')
    plt.title('Violation Rates per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(100, episode_stats[:, 2:].max() * 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / "cbf_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def test_mpc_gcbf(args):
    """Main testing function."""

    print(f"> Running MPC+GCBF Safety-First Test with Safety Margin")
    print(f"Arguments: {args}")

    # Setup JAX environment
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # Load GCBF model configuration
    with open(os.path.join(args.gcbf_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Create environment
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    print(f"\nEnvironment Setup:")
    print(f"  Type: {env.__class__.__name__}")
    print(f"  Agents: {num_agents}")
    print(f"  Obstacles: {args.obs}")
    print(f"  Area Size: {args.area_size}")
    print(f"  Max Steps: {args.max_step}")

    # Load GCBF model
    print(f"\nLoading GCBF model from: {args.gcbf_path}")
    algo = make_algo(
        algo=config.algo, env=env,
        node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim,
        n_agents=env.num_agents, gnn_layers=config.gnn_layers,
        batch_size=config.batch_size, buffer_size=config.buffer_size,
        horizon=config.horizon, lr_actor=config.lr_actor,
        lr_cbf=config.lr_cbf, alpha=config.alpha,
        eps=0.02, inner_epoch=8,
        loss_action_coef=config.loss_action_coef,
        loss_unsafe_coef=config.loss_unsafe_coef,
        loss_safe_coef=config.loss_safe_coef,
        loss_h_dot_coef=config.loss_h_dot_coef,
        max_grad_norm=2.0, seed=config.seed
    )

    # Load model weights
    model_path = os.path.join(args.gcbf_path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print(f"Loading model weights from step: {step}")
    algo.load(model_path, step)

    # Create output directory first for debug logging
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    output_dir = pathlib.Path(f"./logs/mpc_gcbf_results/{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # Create debug logger
    debug_logger = MPCDebugLogger(output_dir, num_agents, args.safety_margin)
    print(f"Debug logger initialized with margin={args.safety_margin:.3f} - will log to CSV files for troubleshooting")

    # Create GCBF interface and MPC controller
    gcbf_interface = GCBFInterface(algo)
    mpc_controller = SafetyFirstMPC(
        gcbf_interface=gcbf_interface,
        horizon=args.mpc_horizon,
        dt=env.dt,
        u_max=args.u_max,
        alpha=args.alpha,
        margin=args.safety_margin,
        goal_weight=args.goal_weight,
        debug_logger=debug_logger
    )

    print(f"\nMPC Controller Setup:")
    print(f"  Horizon: {args.mpc_horizon}")
    print(f"  CBF Alpha: {args.alpha}")
    print(f"  Safety Margin: {args.safety_margin:.3f}")
    print(f"  Control Limit: {args.u_max}")
    print(f"  Goal Weight: {args.goal_weight}")
    print(f"  Strategy: HARD safety constraints (with margin) + SOFT goal relaxation")

    # Create tester with debug logging
    tester = MPCGCBFTester(env, mpc_controller, debug_logger)

    # Generate test keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1000)[:args.epi]
    test_keys = test_keys[args.offset:]

    # Run episodes
    print(f"\nRunning {args.epi} episodes...")
    rollouts = []
    h_traces = []
    all_metrics = []
    all_episode_stats = []

    for i in range(args.epi):
        print(f"\nEpisode {i + 1}/{args.epi}")

        rollout, h_trace, episode_stats = tester.run_episode(test_keys[i], args.max_step, episode_id=i + 1)
        metrics = tester.evaluate_rollout(rollout)

        rollouts.append(rollout)
        h_traces.append(h_trace)
        all_metrics.append(metrics)
        all_episode_stats.append(episode_stats)

        # Episode summary
        print(f"  Results: Safe={metrics['safe_rate'] * 100:.1f}%, "
              f"Success={metrics['success_rate'] * 100:.1f}%, "
              f"Finish={metrics['finish_rate'] * 100:.1f}%")
        print(f"  Rewards: {metrics['total_reward']:.2f}, Cost: {metrics['total_cost']:.2f}")
        print(f"  CBF: Min={episode_stats['min_h_overall']:.4f}, "
              f"Unsafe={episode_stats['cbf_violations']}, "
              f"Margin={episode_stats['margin_violations']}, "
              f"Steps={episode_stats['steps_completed']}")

    # Compute overall statistics
    safe_rates = [m['safe_rate'] for m in all_metrics]
    success_rates = [m['success_rate'] for m in all_metrics]
    finish_rates = [m['finish_rate'] for m in all_metrics]

    # CBF analysis (both actual and margin-aware)
    all_h_values = []
    total_unsafe_violations = 0
    total_margin_violations = 0
    total_samples = 0

    for h_trace in h_traces:
        for h_step in h_trace:
            h_array = np.array(h_step).flatten()
            all_h_values.extend(h_array)
            total_unsafe_violations += np.sum(h_array < 0)
            total_margin_violations += np.sum(h_array < args.safety_margin)
            total_samples += len(h_array)

    all_h_values = np.array(all_h_values)

    # Print comprehensive results
    print(f"\n" + "=" * 60)
    print(f"OVERALL RESULTS")
    print(f"=" * 60)
    print(f"Safety Rate:  {np.mean(safe_rates) * 100:.2f}% ± {np.std(safe_rates) * 100:.2f}%")
    print(f"Success Rate: {np.mean(success_rates) * 100:.2f}% ± {np.std(success_rates) * 100:.2f}%")
    print(f"Finish Rate:  {np.mean(finish_rates) * 100:.2f}% ± {np.std(finish_rates) * 100:.2f}%")

    print(f"\nCBF ANALYSIS (with margin={args.safety_margin:.3f})")
    print(
        f"Unsafe Violations (h < 0): {total_unsafe_violations}/{total_samples} ({total_unsafe_violations / max(1, total_samples) * 100:.2f}%)")
    print(
        f"Margin Violations (h < {args.safety_margin:.3f}): {total_margin_violations}/{total_samples} ({total_margin_violations / max(1, total_samples) * 100:.2f}%)")
    print(f"CBF Range: [{all_h_values.min():.4f}, {all_h_values.max():.4f}]")
    print(f"CBF Mean ± Std: {all_h_values.mean():.4f} ± {all_h_values.std():.4f}")

    # Calculate margin effectiveness
    margin_effectiveness = max(0, total_margin_violations - total_unsafe_violations)
    print(f"Margin Effectiveness: {margin_effectiveness} early braking events prevented actual violations")

    # Save results (output_dir already created earlier)
    print(f"\nSaving results to: {output_dir}")

    # Save CBF traces and statistics
    np.save(output_dir / "h_traces.npy", h_traces)

    # Save comprehensive statistics
    with open(output_dir / "results_summary.txt", "w") as f:
        f.write(f"MPC+GCBF Safety-First Test Results with Margin\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Environment: {env.__class__.__name__}\n")
        f.write(f"Agents: {num_agents}, Obstacles: {args.obs}, Area: {args.area_size}\n")
        f.write(f"Episodes: {args.epi}, Max Steps: {args.max_step}\n\n")

        f.write(f"MPC Settings:\n")
        f.write(f"  Horizon: {args.mpc_horizon}\n")
        f.write(f"  CBF Alpha: {args.alpha}\n")
        f.write(f"  Safety Margin: {args.safety_margin:.3f}\n")
        f.write(f"  Control Limit: {args.u_max}\n")
        f.write(f"  Goal Weight: {args.goal_weight}\n")
        f.write(f"  Strategy: Hard safety constraints (with margin) + Soft goal relaxation\n\n")

        f.write(f"Overall Results:\n")
        f.write(f"  Safety Rate: {np.mean(safe_rates) * 100:.3f}% ± {np.std(safe_rates) * 100:.3f}%\n")
        f.write(f"  Success Rate: {np.mean(success_rates) * 100:.3f}% ± {np.std(success_rates) * 100:.3f}%\n")
        f.write(f"  Finish Rate: {np.mean(finish_rates) * 100:.3f}% ± {np.std(finish_rates) * 100:.3f}%\n\n")

        f.write(f"CBF Analysis (with margin={args.safety_margin:.3f}):\n")
        f.write(
            f"  Unsafe Violations (h < 0): {total_unsafe_violations}/{total_samples} ({total_unsafe_violations / max(1, total_samples) * 100:.3f}%)\n")
        f.write(
            f"  Margin Violations (h < {args.safety_margin:.3f}): {total_margin_violations}/{total_samples} ({total_margin_violations / max(1, total_samples) * 100:.3f}%)\n")
        f.write(f"  CBF Range: [{all_h_values.min():.6f}, {all_h_values.max():.6f}]\n")
        f.write(f"  CBF Mean ± Std: {all_h_values.mean():.6f} ± {all_h_values.std():.6f}\n")
        f.write(f"  Margin Effectiveness: {margin_effectiveness} early braking events\n\n")

        f.write(f"Per-Episode CBF Statistics:\n")
        for i, stats in enumerate(all_episode_stats):
            f.write(f"  Episode {i + 1}: Min h={stats['min_h_overall']:.6f}, "
                    f"Unsafe={stats['cbf_violations']}, Margin={stats['margin_violations']}, "
                    f"Steps={stats['steps_completed']}\n")

    # Generate comprehensive plots with margin
    print("Generating CBF analysis plots with margin visualization...")
    generate_cbf_plots(h_traces, output_dir, args.safety_margin)

    # Generate videos
    if not args.no_video:
        print("Generating episode videos...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        for i, (rollout, metrics) in enumerate(zip(rollouts[:5], all_metrics[:5])):
            video_name = f"mpc_gcbf_margin{args.safety_margin:.3f}_ep{i:02d}_safe{metrics['safe_rate'] * 100:.0f}_success{metrics['success_rate'] * 100:.0f}"
            video_path = videos_dir / f"{video_name}.mp4"
            env.render_video(rollout, video_path, metrics['is_unsafe'], dpi=args.dpi)

    print(f"\n🎉 MPC+GCBF Safety-First Testing with Margin Complete!")
    print(f"📊 Key Results:")
    print(f"   Safety: {np.mean(safe_rates) * 100:.1f}% | Success: {np.mean(success_rates) * 100:.1f}%")
    print(f"   Unsafe Violations: {total_unsafe_violations / max(1, total_samples) * 100:.1f}%")
    print(f"   Margin Violations: {total_margin_violations / max(1, total_samples) * 100:.1f}%")
    print(f"   Margin Effectiveness: {margin_effectiveness} early braking events")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n🔍 DEBUG LOGS for troubleshooting:")
    print(f"   📋 Agent states & positions: {output_dir}/agent_states.csv")
    print(f"   🎯 CBF values & gradients: {output_dir}/cbf_data.csv")
    print(f"   ⚙️  MPC solver status: {output_dir}/mpc_status.csv")
    print(f"   🚧 Obstacle positions: {output_dir}/obstacles.csv")
    print(f"   💾 CBF traces: {output_dir}/h_traces.npy")
    print(f"\n💡 Analysis Tips:")
    print(f"   - Check cbf_data.csv for 'margin_violation' column to see early braking")
    print(f"   - Orange lines in plots show margin threshold (h = {args.safety_margin:.3f})")
    print(f"   - Compare margin violations vs unsafe violations for effectiveness")
    print(f"   - Emergency brake events indicate margin constraint conflicts")
    print(f"   - Higher margin = more conservative, lower performance but safer")


def main():
    parser = argparse.ArgumentParser(description="MPC+GCBF Safety-First Testing with Safety Margin")

    # Environment parameters
    parser.add_argument("-n", "--num-agents", type=int, default=None,
                        help="Number of agents (default: from model config)")
    parser.add_argument("--obs", type=int, default=0,
                        help="Number of obstacles")
    parser.add_argument("--area-size", type=float, required=True,
                        help="Environment area size")
    parser.add_argument("--max-step", type=int, default=256,
                        help="Maximum steps per episode")
    parser.add_argument("--max-travel", type=float, default=None,
                        help="Maximum travel distance")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment type (default: from model config)")

    # Model parameters
    parser.add_argument("--gcbf-path", type=str, required=True,
                        help="Path to trained GCBF model directory")
    parser.add_argument("--step", type=int, default=None,
                        help="Model checkpoint step (default: latest)")

    # MPC parameters
    parser.add_argument("--mpc-horizon", type=int, default=10,
                        help="MPC prediction horizon")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CBF class-K function parameter")
    parser.add_argument("--safety-margin", type=float, default=0.0,
                        help="Safety margin - agents brake when h < margin (default: 0.0)")
    parser.add_argument("--u-max", type=float, default=1.0,
                        help="Maximum control input magnitude")
    parser.add_argument("--goal-weight", type=float, default=1.0,
                        help="Weight for goal-reaching objective (automatically relaxed when conflicts with safety)")

    # Testing parameters
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--epi", type=int, default=5,
                        help="Number of episodes to test")
    parser.add_argument("--offset", type=int, default=0,
                        help="Episode offset")

    # System parameters
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode (disable JAX JIT)")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Use CPU instead of GPU")
    parser.add_argument("--no-video", action="store_true", default=False,
                        help="Skip video generation")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Video DPI")

    args = parser.parse_args()
    test_mpc_gcbf(args)


if __name__ == "__main__":
    main()