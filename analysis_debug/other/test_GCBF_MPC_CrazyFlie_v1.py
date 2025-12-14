#!/usr/bin/env python3
"""
GCBF + MPC Test Script for CrazyFlie - v1

Hierarchical control structure:
- High-level MPC: Computes desired world-frame velocities [vx, vy, vz, r] with GCBF safety constraints
- Low-level LQR: Tracks velocity commands using motor controls (handled by CrazyFlie environment)

Based on the v9 script structure but adapted for 3D CrazyFlie dynamics.
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
import gc

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

        # Get agent node indices
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
            # Return zero gradient as fallback
            return np.zeros(agent_states.shape[1])


class CrazyFlieDebugLogger:
    """Debug logger for CrazyFlie MPC controller."""

    def __init__(self, output_dir: pathlib.Path, num_agents: int, margin: float = 0.0):
        self.output_dir = output_dir
        self.num_agents = num_agents
        self.margin = margin

        # Create log files
        self.state_log_file = output_dir / "agent_states.csv"
        self.cbf_log_file = output_dir / "cbf_data.csv"
        self.mpc_log_file = output_dir / "mpc_status.csv"
        self.obstacle_log_file = output_dir / "obstacles.csv"

        self._init_log_files()

    def _init_log_files(self):
        """Initialize CSV log files with appropriate headers."""

        # Agent states log - CrazyFlie has 12D state
        with open(self.state_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id',
                      'pos_x', 'pos_y', 'pos_z',  # Position
                      'yaw', 'pitch', 'roll',  # Orientation
                      'vel_u', 'vel_v', 'vel_w',  # Body frame velocities
                      'rate_r', 'rate_q', 'rate_p',  # Angular rates
                      'goal_x', 'goal_y', 'goal_z',  # Goal position
                      'goal_yaw', 'goal_pitch', 'goal_roll',  # Goal orientation
                      'goal_vel_u', 'goal_vel_v', 'goal_vel_w',  # Goal velocities
                      'goal_rate_r', 'goal_rate_q', 'goal_rate_p']  # Goal rates
            writer.writerow(header)
            f.flush()

        # CBF data log
        with open(self.cbf_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'cbf_value', 'margin_violation']
            # Add gradient components for 12D state
            for i in range(12):
                header.append(f'grad_{i}')
            writer.writerow(header)
            f.flush()

        # MPC status log
        with open(self.mpc_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'solver_status', 'solve_time',
                      'action_vx', 'action_vy', 'action_vz', 'action_r',  # 4D action
                      'feasible']
            writer.writerow(header)
            f.flush()

        # Obstacles log
        with open(self.obstacle_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'obstacle_id', 'pos_x', 'pos_y', 'pos_z', 'type']
            writer.writerow(header)
            f.flush()

    def log_episode_state(self, episode: int, step: int, agent_states: np.ndarray,
                          goal_states: np.ndarray, h_values: np.ndarray,
                          cbf_gradients: List[np.ndarray], graph: 'GraphsTuple'):
        """Log complete state information."""

        # Log agent states and goals
        with open(self.state_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.num_agents):
                agent_state = agent_states[i]  # 12D state
                goal_state = goal_states[i]  # 12D goal

                row = [episode, step, i]
                row.extend([float(x) for x in agent_state])  # 12 agent state values
                row.extend([float(x) for x in goal_state])  # 12 goal state values
                writer.writerow(row)
            f.flush()

        # Log CBF data
        with open(self.cbf_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.num_agents):
                cbf_val = float(h_values[i].item())
                margin_violation = cbf_val < self.margin

                row = [episode, step, i, cbf_val, margin_violation]

                if i < len(cbf_gradients) and cbf_gradients[i] is not None:
                    grad = cbf_gradients[i]
                    if len(grad) >= 12:
                        row.extend([float(x) for x in grad[:12]])
                    else:
                        row.extend([float(x) for x in grad] + [0.0] * (12 - len(grad)))
                else:
                    row.extend([0.0] * 12)

                writer.writerow(row)
            f.flush()

        # Log obstacle positions
        with open(self.obstacle_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            obstacle_mask = graph.node_type == 2
            if np.any(obstacle_mask):
                obstacle_states = graph.states[obstacle_mask]
                for obs_id, obs_state in enumerate(obstacle_states):
                    writer.writerow([
                        episode, step, obs_id,
                        float(obs_state[0]), float(obs_state[1]), float(obs_state[2]),
                        'static'
                    ])
            f.flush()

    def log_mpc_result(self, episode: int, step: int, agent_id: int,
                       solver_status: str, solve_time: float, action: np.ndarray,
                       feasible: bool):
        """Log MPC solver results."""

        with open(self.mpc_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, step, agent_id, solver_status, solve_time,
                float(action[0]), float(action[1]), float(action[2]), float(action[3]),
                feasible
            ])
            f.flush()


class CrazyflieSafetyMPC:
    """MPC controller for CrazyFlie with GCBF safety constraints."""

    def __init__(self,
                 gcbf_interface: GCBFInterface,
                 horizon: int = 10,
                 dt: float = 0.03,
                 v_max: float = 2.0,
                 r_max: float = 1.0,
                 alpha: float = 1.0,
                 margin: float = 0.0,
                 goal_weight: float = 1.0,
                 debug_logger: Optional[CrazyFlieDebugLogger] = None):

        self.gcbf = gcbf_interface
        self.horizon = horizon
        self.dt = dt
        self.v_max = v_max  # Max velocity magnitude
        self.r_max = r_max  # Max yaw rate
        self.alpha = alpha
        self.margin = margin
        self.goal_weight = goal_weight
        self.debug_logger = debug_logger

        # Cost matrices for CrazyFlie
        # State: [x, y, z, psi, theta, phi, u, v, w, r, q, p]
        # Penalize position and orientation more, velocities and rates less
        # self.Q = np.diag([10.0, 10.0, 10.0,  # Position (x,y,z)
        #                   2.0, 2.0, 2.0,  # Orientation (psi,theta,phi)
        #                   1.0, 1.0, 1.0,  # Body velocities (u,v,w)
        #                   1.0, 1.0, 1.0])  # Angular rates (r,q,p)

        self.Q = np.diag([10 , 10 , 10 ,  # Position (x,y,z) - high weight
                          2  , 0  , 0  ,  # Orientation - moderate
                          0  , 0  , 0  ,  # Body velocities - very low
                          0  , 0  , 0   ])  # Angular rates - very low

        # Control cost for [vx_cmd, vy_cmd, vz_cmd, r_cmd]
        self.R = np.eye(4) * 0.5

        print(f"CrazyFlie Safety MPC Controller initialized:")
        print(f"  Horizon: {horizon} steps")
        print(f"  Max velocity: {v_max} m/s")
        print(f"  Max yaw rate: {r_max} rad/s")
        print(f"  CBF alpha: {alpha}")
        print(f"  Safety margin: {margin}")
        print(f"  Goal weight: {goal_weight}")

    def solve(self, agent_states: np.ndarray, goal_states: np.ndarray,
              graph: GraphsTuple, episode: int = 0, step: int = 0) -> np.ndarray:
        """Solve MPC for all CrazyFlie agents with GCBF constraints."""

        n_agents = agent_states.shape[0]
        actions = np.zeros((n_agents, 4))  # [vx_cmd, vy_cmd, vz_cmd, r_cmd]

        # Get CBF values for logging
        h_values = self.gcbf.evaluate_h(graph)

        # Collect gradients for logging
        cbf_gradients = []
        for i in range(n_agents):
            try:
                grad = self.gcbf.get_cbf_gradient(graph, i)
                cbf_gradients.append(grad)
            except:
                cbf_gradients.append(np.zeros(12))

        # Log current state
        if self.debug_logger:
            self.debug_logger.log_episode_state(
                episode, step, agent_states, goal_states, h_values, cbf_gradients, graph
            )

        # Solve MPC for each agent
        for i in range(n_agents):
            start_time = time.time()

            try:
                action, solver_status, feasible = self._solve_single_agent(
                    agent_states[i], goal_states[i], graph, i, agent_states, episode, step
                )
                actions[i] = action
                solve_time = time.time() - start_time

                if step < 3:  # Print for first few steps
                    print(f"  Agent {i}: MPC solved ({solver_status}) - "
                          f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")

            except Exception as e:
                solve_time = time.time() - start_time
                print(f"Agent {i} MPC failed: {e}")

                # Emergency hover action
                actions[i] = np.zeros(4)  # Hover in place
                solver_status = f"FAILED: {str(e)[:50]}"
                feasible = False

            # Log MPC result
            if self.debug_logger:
                self.debug_logger.log_mpc_result(
                    episode, step, i, solver_status, solve_time, actions[i], feasible
                )

        return actions

    def _solve_single_agent(self, current_state: np.ndarray, goal_state: np.ndarray,
                            graph: GraphsTuple, agent_idx: int, all_agent_states: np.ndarray,
                            episode: int, step: int) -> Tuple[np.ndarray, str, bool]:
        """Solve MPC for single CrazyFlie agent."""

        # Decision variables over horizon
        # State: [x, y, z, psi, theta, phi, u, v, w, r, q, p] (12D)
        # Action: [vx_cmd, vy_cmd, vz_cmd, r_cmd] (4D - world frame velocities + yaw rate)
        x = cp.Variable((self.horizon + 1, 12))
        u = cp.Variable((self.horizon, 4))

        # Objective and constraints
        cost = 0
        constraints = []

        # Initial condition
        constraints += [x[0] == current_state]

        # Dynamics and constraints over horizon
        for k in range(self.horizon):
            # Simplified CrazyFlie dynamics for MPC prediction
            # This is a linearized approximation - in practice you might want
            # more sophisticated dynamics prediction

            # Position dynamics: ẋ = R * [u, v, w] (world frame velocity)
            # For simplicity, assume small angles so R ≈ I + skew(angles)
            constraints += [
                # Position update (simplified - assumes world frame velocities)
                x[k + 1, 0] == x[k, 0] + u[k, 0] * self.dt,  # x += vx_cmd * dt
                x[k + 1, 1] == x[k, 1] + u[k, 1] * self.dt,  # y += vy_cmd * dt
                x[k + 1, 2] == x[k, 2] + u[k, 2] * self.dt,  # z += vz_cmd * dt

                # Yaw update
                x[k + 1, 3] == x[k, 3] + u[k, 3] * self.dt,  # psi += r_cmd * dt

                # For simplicity, assume other states evolve slowly or are controlled by low-level
                x[k + 1, 4] == x[k, 4],  # theta (pitch)
                x[k + 1, 5] == x[k, 5],  # phi (roll)
                x[k + 1, 6] == x[k, 6],  # u (body vel)
                x[k + 1, 7] == x[k, 7],  # v (body vel)
                x[k + 1, 8] == x[k, 8],  # w (body vel)
                x[k + 1, 9] == x[k, 9],  # r (yaw rate)
                x[k + 1, 10] == x[k, 10],  # q (pitch rate)
                x[k + 1, 11] == x[k, 11],  # p (roll rate)
            ]

            # Control limits
            constraints += [
                cp.norm(u[k, :3], 2) <= self.v_max,  # Velocity magnitude limit
                cp.abs(u[k, 3]) <= self.r_max,  # Yaw rate limit
            ]

            # Goal-reaching objective (soft)
            cost += self.goal_weight * cp.quad_form(x[k] - goal_state, self.Q)
            cost += cp.quad_form(u[k], self.R)

            # CBF safety constraints (hard)
            safety_constraint = self._add_cbf_constraint(graph, agent_idx, x[k], u[k], k)
            if safety_constraint is not None:
                constraints += [safety_constraint]

        # Terminal cost
        cost += self.goal_weight * cp.quad_form(x[self.horizon] - goal_state, self.Q)

        # Solve optimization
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Try multiple solvers
        solver_names = ["OSQP", "ECOS", "SCS"]
        solver_status = "UNKNOWN"
        feasible = False

        for solver, solver_name in zip([cp.OSQP, cp.ECOS, cp.SCS], solver_names):
            try:
                prob.solve(solver=solver, verbose=False)
                solver_status = f"{solver_name}_{prob.status}"

                if prob.status in ["optimal", "optimal_inaccurate"]:
                    feasible = True
                    if step == 0:
                        print(f"    Agent {agent_idx}: MPC solved successfully ({solver_status})")

                    optimal_u = u[0].value
                    if optimal_u is None:
                        optimal_u = np.zeros(4)
                    return np.array(optimal_u).flatten(), solver_status, feasible

            except Exception as e:
                solver_status = f"{solver_name}_ERROR: {str(e)[:30]}"
                continue

        # If all solvers fail, hover in place
        print(f"  Agent {agent_idx}: MPC infeasible ({solver_status}) - hovering")
        hover_action = np.zeros(4)  # Zero velocity commands = hover
        return hover_action, f"INFEASIBLE_{solver_status}", False

    def _add_cbf_constraint(self, graph: GraphsTuple, agent_idx: int,
                            agent_state: cp.Variable, control: cp.Variable,
                            step: int) -> Optional[cp.Constraint]:
        """Add CBF constraint: ḣ + α*h ≥ margin."""

        try:
            # Get current CBF value and gradient
            h_values = self.gcbf.evaluate_h(graph)
            h_current = float(h_values[agent_idx].item())
            cbf_grad = self.gcbf.get_cbf_gradient(graph, agent_idx)

            # Check if gradient computation succeeded
            if np.allclose(cbf_grad, 0):
                return None

            # CrazyFlie dynamics: simplified linearization
            # State: [x, y, z, psi, theta, phi, u, v, w, r, q, p]
            # For the constraint, we need ∇h · ẋ where ẋ is state derivative

            # State derivative from our simplified dynamics:
            # ẋ = [vx_cmd, vy_cmd, vz_cmd, r_cmd, 0, 0, 0, 0, 0, 0, 0, 0]

            # So ḣ = ∇h · ẋ = grad[0]*vx_cmd + grad[1]*vy_cmd + grad[2]*vz_cmd + grad[3]*r_cmd
            # (assuming other derivatives are small)

            if len(cbf_grad) >= 4:
                # Control-affine form: ḣ = drift_term + control_term
                drift_term = 0  # No drift in simplified model
                control_term = (cbf_grad[0] * control[0] +  # vx contribution
                                cbf_grad[1] * control[1] +  # vy contribution
                                cbf_grad[2] * control[2] +  # vz contribution
                                cbf_grad[3] * control[3])  # yaw rate contribution

                # CBF constraint: ḣ + α*h ≥ margin
                cbf_constraint = (drift_term + control_term + self.alpha * h_current >= self.margin)

                # Debug info for first step
                if step == 0:
                    margin_status = "MARGIN VIOLATED" if h_current < self.margin else "SAFE"
                    print(f"    Agent {agent_idx}: h={h_current:.4f}, margin={self.margin:.4f} [{margin_status}]")

                return cbf_constraint
            else:
                return None

        except Exception as e:
            if step == 0:
                print(f"    Agent {agent_idx}: CBF constraint failed: {e}")
            return None


class CrazyFlieGCBFTester:
    """Testing framework for CrazyFlie GCBF+MPC approach."""

    def __init__(self, env, mpc_controller: CrazyflieSafetyMPC, debug_logger: Optional[CrazyFlieDebugLogger] = None):
        self.env = env
        self.controller = mpc_controller
        self.debug_logger = debug_logger
        self.margin = mpc_controller.margin

        # Evaluation functions
        self.is_unsafe_fn = jax_jit_np(jax.vmap(env.collision_mask))
        self.is_finish_fn = jax_jit_np(jax.vmap(env.finish_mask))

    def run_episode(self, key: jr.PRNGKey, max_steps: int, episode_id: int = 0) -> Tuple[
        RolloutResult, List[np.ndarray], dict]:
        """Run single episode with CrazyFlie GCBF+MPC."""

        # Initialize
        graph = self.env.reset(key)
        h_trace = []
        episode_stats = {
            'cbf_violations': 0,
            'margin_violations': 0,
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

            # Evaluate CBF values
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

            # Print CBF status periodically
            if step % 20 == 0 or np.any(h_values < self.margin):
                unsafe_count = np.sum(h_values < 0)
                margin_count = np.sum(h_values < self.margin)
                print(f"  Step {step:3d}: CBF∈[{h_values.min():.3f}, {h_values.max():.3f}], "
                      f"mean={h_values.mean():.3f}, margin_viols={margin_count}, unsafe={unsafe_count}")

                if margin_count > 0:
                    margin_agents = np.where(h_values < self.margin)[0]
                    print(f"    ⚠️  MARGIN: Agents {margin_agents.tolist()}")

                if unsafe_count > 0:
                    unsafe_agents = np.where(h_values < 0)[0]
                    print(f"    🚨 UNSAFE: Agents {unsafe_agents.tolist()}")

            # Get MPC control action (world frame velocities + yaw rate)
            action_vel_cmd = self.controller.solve(agent_states, goal_states, graph, episode_id, step)

            # Convert to JAX array for environment
            action_jax = jnp.array(action_vel_cmd)

            # Step environment (CrazyFlie will handle conversion to motor commands internally)
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


def generate_analysis_plots(h_traces: List[List[np.ndarray]], output_dir: pathlib.Path, margin: float = 0.0):
    """Generate CrazyFlie GCBF+MPC analysis plots."""

    plt.figure(figsize=(15, 10))

    # CBF evolution over time
    plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(h_traces), 10)))

    for i, (h_trace, color) in enumerate(zip(h_traces, colors)):
        if len(h_trace) == 0:
            continue

        h_array = np.array([np.array(h_step).flatten() for h_step in h_trace])
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
    plt.title('CrazyFlie GCBF+MPC: CBF Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # CBF distribution
    plt.subplot(2, 3, 2)
    all_h_values = []
    for h_trace in h_traces:
        for h_step in h_trace:
            all_h_values.extend(np.array(h_step).flatten())

    if all_h_values:
        all_h_values = np.array(all_h_values)
        plt.hist(all_h_values, bins=50, alpha=0.7, edgecolor='black', density=True)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Safety Threshold')
        if margin > 0:
            plt.axvline(x=margin, color='orange', linestyle=':', linewidth=2, label=f'Safety Margin')
        plt.axvline(x=all_h_values.mean(), color='blue', linestyle='-.', linewidth=2,
                    label=f'Mean={all_h_values.mean():.3f}')
        plt.xlabel('CBF Value h(x)')
        plt.ylabel('Density')
        plt.title('CBF Value Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Additional analysis plots
    plt.subplot(2, 3, 3)
    plt.text(0.5, 0.5, 'CrazyFlie\n3D Trajectories\n(To be implemented)',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('3D Flight Paths')

    plt.subplot(2, 3, 4)
    plt.text(0.5, 0.5, 'Control Effort\nAnalysis\n(Velocity Commands)',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Control Analysis')

    plt.subplot(2, 3, 5)
    plt.text(0.5, 0.5, 'Altitude\nMaintenance\nPerformance',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Altitude Control')

    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, 'Goal Reaching\nTrajectories\n3D Analysis',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Goal Achievement')

    plt.tight_layout()
    plt.savefig(output_dir / "crazyflie_gcbf_mpc_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    gc.collect()


def test_crazyflie_gcbf_mpc(args):
    """Main testing function for CrazyFlie GCBF+MPC integration."""

    print(f"> Running CrazyFlie GCBF+MPC Test (v1)")
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

    # Create CrazyFlie environment
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id="CrazyFlie",  # Force CrazyFlie environment
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    print(f"\nCrazyFlie Environment Setup:")
    print(f"  Type: {env.__class__.__name__}")
    print(f"  Agents: {num_agents}")
    print(f"  Obstacles: {args.obs}")
    print(f"  Area Size: {args.area_size}")
    print(f"  Max Steps: {args.max_step}")
    print(f"  State Dimension: {env.state_dim}")
    print(f"  Action Dimension: {env.action_dim}")

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

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    output_dir = pathlib.Path(f"./logs/crazyflie_gcbf_mpc_results/{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # Create debug logger
    debug_logger = CrazyFlieDebugLogger(output_dir, num_agents, args.safety_margin)
    print(f"CrazyFlie debug logger initialized")

    # Create GCBF interface and MPC controller
    gcbf_interface = GCBFInterface(algo)
    mpc_controller = CrazyflieSafetyMPC(
        gcbf_interface=gcbf_interface,
        horizon=args.mpc_horizon,
        dt=env.dt,
        v_max=args.v_max,
        r_max=args.r_max,
        alpha=args.alpha,
        margin=args.safety_margin,
        goal_weight=args.goal_weight,
        debug_logger=debug_logger
    )

    print(f"\nCrazyFlie GCBF+MPC Controller Setup:")
    print(f"  MPC Horizon: {args.mpc_horizon}")
    print(f"  CBF Alpha: {args.alpha}")
    print(f"  Safety Margin: {args.safety_margin:.3f}")
    print(f"  Max Velocity: {args.v_max} m/s")
    print(f"  Max Yaw Rate: {args.r_max} rad/s")
    print(f"  Goal Weight: {args.goal_weight}")

    # Create tester
    tester = CrazyFlieGCBFTester(env, mpc_controller, debug_logger)

    # Generate test keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1000)[:args.epi]
    test_keys = test_keys[args.offset:]

    # Run episodes
    print(f"\nRunning {args.epi} episodes with CrazyFlie GCBF+MPC...")

    h_traces = []
    all_metrics = []
    all_episode_stats = []
    rollouts = []

    for i in range(args.epi):
        print(f"\nEpisode {i + 1}/{args.epi}")

        rollout, h_trace, episode_stats = tester.run_episode(test_keys[i], args.max_step, episode_id=i + 1)
        metrics = tester.evaluate_rollout(rollout)

        # Store data
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

        gc.collect()

    # Compute overall statistics
    safe_rates = [m['safe_rate'] for m in all_metrics]
    success_rates = [m['success_rate'] for m in all_metrics]
    finish_rates = [m['finish_rate'] for m in all_metrics]

    # CBF analysis
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
    print(f"\n" + "=" * 80)
    print(f"CRAZYFLIE GCBF+MPC INTEGRATION RESULTS (v1)")
    print(f"=" * 80)
    print(f"Safety Rate:  {np.mean(safe_rates) * 100:.2f}% ± {np.std(safe_rates) * 100:.2f}%")
    print(f"Success Rate: {np.mean(success_rates) * 100:.2f}% ± {np.std(success_rates) * 100:.2f}%")
    print(f"Finish Rate:  {np.mean(finish_rates) * 100:.2f}% ± {np.std(finish_rates) * 100:.2f}%")

    print(f"\nCBF SAFETY ANALYSIS (with margin={args.safety_margin:.3f})")
    print(
        f"Unsafe Violations (h < 0): {total_unsafe_violations}/{total_samples} ({total_unsafe_violations / max(1, total_samples) * 100:.2f}%)")
    print(
        f"Margin Violations (h < {args.safety_margin:.3f}): {total_margin_violations}/{total_samples} ({total_margin_violations / max(1, total_samples) * 100:.2f}%)")
    print(f"CBF Range: [{all_h_values.min():.4f}, {all_h_values.max():.4f}]")
    print(f"CBF Mean ± Std: {all_h_values.mean():.4f} ± {all_h_values.std():.4f}")

    # Save results
    print(f"\nSaving CrazyFlie GCBF+MPC results to: {output_dir}")

    # Save summary
    with open(output_dir / "crazyflie_gcbf_mpc_summary.txt", "w") as f:
        f.write(f"CrazyFlie GCBF+MPC Integration Test Results (v1)\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Environment: {env.__class__.__name__}\n")
        f.write(f"Agents: {num_agents}, Obstacles: {args.obs}, Area: {args.area_size}\n")
        f.write(f"Episodes: {args.epi}, Max Steps: {args.max_step}\n\n")

        f.write(f"MPC Settings:\n")
        f.write(f"  Horizon: {args.mpc_horizon}\n")
        f.write(f"  CBF Alpha: {args.alpha}\n")
        f.write(f"  Safety Margin: {args.safety_margin:.3f}\n")
        f.write(f"  Max Velocity: {args.v_max} m/s\n")
        f.write(f"  Max Yaw Rate: {args.r_max} rad/s\n")
        f.write(f"  Goal Weight: {args.goal_weight}\n\n")

        f.write(f"Overall Results:\n")
        f.write(f"  Safety Rate: {np.mean(safe_rates) * 100:.3f}% ± {np.std(safe_rates) * 100:.3f}%\n")
        f.write(f"  Success Rate: {np.mean(success_rates) * 100:.3f}% ± {np.std(success_rates) * 100:.3f}%\n")
        f.write(f"  Finish Rate: {np.mean(finish_rates) * 100:.3f}% ± {np.std(finish_rates) * 100:.3f}%\n\n")

        f.write(f"CBF Analysis:\n")
        f.write(
            f"  Unsafe Violations: {total_unsafe_violations}/{total_samples} ({total_unsafe_violations / max(1, total_samples) * 100:.3f}%)\n")
        f.write(
            f"  Margin Violations: {total_margin_violations}/{total_samples} ({total_margin_violations / max(1, total_samples) * 100:.3f}%)\n")
        f.write(f"  CBF Range: [{all_h_values.min():.6f}, {all_h_values.max():.6f}]\n")
        f.write(f"  CBF Mean ± Std: {all_h_values.mean():.6f} ± {all_h_values.std():.6f}\n")

    # Generate analysis plots
    print("Generating CrazyFlie GCBF+MPC analysis plots...")
    generate_analysis_plots(h_traces, output_dir, args.safety_margin)

    # Generate videos if requested
    if not args.no_video:
        print("Generating episode videos...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        for i, (rollout, metrics) in enumerate(zip(rollouts, all_metrics)):
            video_name = f"crazyflie_gcbf_mpc_ep{i:02d}_safe{metrics['safe_rate'] * 100:.0f}_success{metrics['success_rate'] * 100:.0f}"
            video_path = videos_dir / f"{video_name}.mp4"
            env.render_video(rollout, video_path, metrics['is_unsafe'], dpi=args.dpi)
            print(f"  Generated video {i + 1}/{len(rollouts)}: {video_path}")
            gc.collect()

    print(f"\n🎉 CrazyFlie GCBF+MPC Integration Testing Complete! (v1)")
    print(f"📊 Key Results:")
    print(f"   Safety: {np.mean(safe_rates) * 100:.1f}% | Success: {np.mean(success_rates) * 100:.1f}%")
    print(f"   3D Flight Control: Hierarchical MPC + Low-level LQR")
    print(f"   Safety Violations: {total_unsafe_violations / max(1, total_samples) * 100:.1f}%")
    print(f"📁 Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CrazyFlie GCBF+MPC Integration Testing (v1)")

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
                        help="Safety margin - agents brake when h < margin")
    parser.add_argument("--v-max", type=float, default=2.0,
                        help="Maximum velocity magnitude (m/s)")
    parser.add_argument("--r-max", type=float, default=1.0,
                        help="Maximum yaw rate (rad/s)")
    parser.add_argument("--goal-weight", type=float, default=1.0,
                        help="Weight for goal-reaching objective")

    # Testing parameters
    parser.add_argument("--seed", type=int, default=1111,
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
    test_crazyflie_gcbf_mpc(args)


if __name__ == "__main__":
    main()