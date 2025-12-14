#!/usr/bin/env python3
"""
ESO + GCBF MPC Test Script - SELECTIVE MEMORY OPTIMIZED VERSION with GRAPH PREDICTION (v9-graph-prediction)

Based on v8 with graph prediction over MPC horizon to solve infeasibility issues.
Ensures consistent CBF constraints by predicting agent trajectories and obstacle motion.
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
import gc  # For garbage collection

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, jax2np, mask2index

# CONTROL FLAGS FOR EXPERIMENTAL FEATURES
USE_ESO_FEEDBACK = True  # Use ESO state estimates for MPC feedback (vs true states)
ENABLE_DISTURBANCE_COMPENSATION = True  # Apply disturbance compensation to ALL control actions


def fal(e, alpha, delta):
    """
    Nonlinear function from Han '09 IEEE TIE eq. (18)
    fal(e, α, δ) = { e/δ^(1-α),     |e| ≤ δ
                   { |e|^α sign(e), |e| > δ
    """
    abs_e = np.abs(e)
    if abs_e <= delta:
        return e / (delta ** (1 - alpha))
    else:
        return (abs_e ** alpha) * np.sign(e)


def nonlinear_eso(y, u_accel, z_prev, h, beta01, beta02, beta03, alpha1, alpha2, delta, b0):
    """
    FIXED: Nonlinear Extended State Observer with correct discrete equations

    Args:
        y: current measurement (position)
        u_accel: current control input (acceleration)
        z_prev: previous ESO states [z1, z2, z3] = [pos_est, vel_est, dist_est]
        h: sampling time
        beta01, beta02, beta03: ESO gains
        alpha1, alpha2: nonlinear function parameters
        delta: nonlinear function parameter
        b0: control effectiveness parameter (1.0 for acceleration input)

    Returns:
        z_new: updated ESO states [z1, z2, z3]
        e: estimation error (z1 - y)
    """
    # SAFETY CHECK: Ensure z_prev has correct dimension
    if len(z_prev) != 3:
        print(f"ERROR in nonlinear_eso: z_prev has wrong dimension: {len(z_prev)}")
        print(f"  z_prev: {z_prev}")
        # Initialize with safe values
        z_prev = np.array([float(y), 0.0, 0.0])

    # Estimation error
    e = z_prev[0] - y

    # Nonlinear functions per Han eq. (21)
    fe = fal(e, alpha1, delta)  # fal(e, α1, δ)
    fe1 = fal(e, alpha2, delta)  # fal(e, α2, δ)

    # FIXED: Proper discrete ESO equations with h scaling
    z1_new = z_prev[0] + h * z_prev[1] - h * beta01 * e
    z2_new = z_prev[1] + h * (z_prev[2] + b0 * u_accel) - h * beta02 * fe
    z3_new = z_prev[2] - h * beta03 * fe1

    z_new = np.array([z1_new, z2_new, z3_new])

    # SAFETY CHECK: Ensure output has correct dimension
    if len(z_new) != 3:
        print(f"ERROR in nonlinear_eso: z_new has wrong dimension: {len(z_new)}")
        print(f"  z_new: {z_new}")
        z_new = np.array([float(y), 0.0, 0.0])  # Fallback

    return z_new, e


def disturbance_compensation(u_controller_accel, disturbance_estimate, b0):
    """
    Apply disturbance compensation to controller output

    Args:
        u_controller_accel: control effort from controller (MPC) - scalar acceleration
        disturbance_estimate: estimated disturbance from ESO - scalar
        b0: plant gain estimate

    Returns:
        u_compensated_accel: final control input with disturbance rejection - scalar acceleration
    """
    # Ensure inputs are scalars
    if isinstance(u_controller_accel, (list, np.ndarray)):
        u_controller_accel = float(u_controller_accel)
    if isinstance(disturbance_estimate, (list, np.ndarray)):
        disturbance_estimate = float(disturbance_estimate)

    # Compensate for disturbance: u = (u_controller - disturbance) / b0
    u_compensated_accel = (u_controller_accel - disturbance_estimate) / b0

    # Final control saturation
    u_max_accel = 2.0
    u_compensated_accel = np.clip(u_compensated_accel, -u_max_accel, u_max_accel)

    return float(u_compensated_accel)


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


class ESODebugLogger:
    """MEMORY OPT: ESO debug logger with immediate disk writes (FULL LOGGING RESTORED)."""

    def __init__(self, output_dir: pathlib.Path, num_agents: int, margin: float = 0.0):
        self.output_dir = output_dir
        self.num_agents = num_agents
        self.margin = margin

        # Create log files
        self.state_log_file = output_dir / "agent_states.csv"
        self.cbf_log_file = output_dir / "cbf_data.csv"
        self.mpc_log_file = output_dir / "mpc_status.csv"
        self.obstacle_log_file = output_dir / "obstacles.csv"
        self.eso_log_file = output_dir / "eso_data.csv"
        self.disturbance_log_file = output_dir / "disturbances.csv"

        # MEMORY OPT: Initialize files with headers and keep file handles closed
        self._init_log_files()

    def _init_log_files(self):
        """Initialize CSV log files with appropriate headers."""

        # Agent states log
        with open(self.state_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y',
                      'goal_x', 'goal_y', 'goal_vel_x', 'goal_vel_y']
            writer.writerow(header)
            f.flush()  # MEMORY OPT: Immediate flush

        # CBF data log
        with open(self.cbf_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'cbf_value', 'margin_violation',
                      'grad_x', 'grad_y', 'grad_vx', 'grad_vy',
                      'drift_term', 'control_coeff_x', 'control_coeff_y']
            writer.writerow(header)
            f.flush()  # MEMORY OPT: Immediate flush

        # MPC status log
        with open(self.mpc_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'solver_status', 'solve_time',
                      'action_x', 'action_y', 'feasible', 'emergency_brake',
                      'eso_action_x', 'eso_action_y', 'compensated_x', 'compensated_y']
            writer.writerow(header)
            f.flush()  # MEMORY OPT: Immediate flush

        # Obstacles log
        with open(self.obstacle_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'obstacle_id', 'pos_x', 'pos_y', 'type']
            writer.writerow(header)
            f.flush()  # MEMORY OPT: Immediate flush

        # ESO data log
        with open(self.eso_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'axis',
                      'true_pos', 'eso_pos', 'pos_error',
                      'true_vel', 'eso_vel', 'vel_error',
                      'true_dist', 'eso_dist', 'dist_error',
                      'measurement_error']
            writer.writerow(header)
            f.flush()  # MEMORY OPT: Immediate flush

        # Disturbance log
        with open(self.disturbance_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'true_dist_x', 'true_dist_y',
                      'eso_dist_x', 'eso_dist_y']
            writer.writerow(header)
            f.flush()  # MEMORY OPT: Immediate flush

    def log_episode_state(self, episode: int, step: int, agent_states: np.ndarray,
                          goal_states: np.ndarray, h_values: np.ndarray,
                          cbf_gradients: List[np.ndarray], graph: 'GraphsTuple'):
        """RESTORED: Log complete state information for troubleshooting (EVERY STEP)."""

        # Log agent states and goals with immediate file flush
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
            f.flush()  # MEMORY OPT: Force write to disk immediately

        # Log CBF data with immediate file flush
        with open(self.cbf_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.num_agents):
                cbf_val = float(h_values[i].item())
                margin_violation = cbf_val < self.margin

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
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        ])
                else:
                    writer.writerow([
                        episode, step, i, cbf_val, margin_violation,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])
            f.flush()  # MEMORY OPT: Force write to disk immediately

        # Log obstacle positions with immediate file flush
        with open(self.obstacle_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            obstacle_mask = graph.node_type == 2
            if np.any(obstacle_mask):
                obstacle_states = graph.states[obstacle_mask]
                for obs_id, obs_state in enumerate(obstacle_states):
                    writer.writerow([
                        episode, step, obs_id,
                        float(obs_state[0]), float(obs_state[1]),
                        'static'
                    ])
            f.flush()  # MEMORY OPT: Force write to disk immediately

    def log_eso_data(self, episode: int, step: int, agent_id: int,
                     true_states_x: np.ndarray, true_states_y: np.ndarray,
                     eso_states_x: np.ndarray, eso_states_y: np.ndarray,
                     true_dist_x: float, true_dist_y: float):
        """RESTORED: Log ESO estimation performance (EVERY STEP)."""

        with open(self.eso_log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # SAFETY CHECK: Ensure ESO states have correct dimensions
            if len(eso_states_x) < 3 or len(eso_states_y) < 3:
                print(f"WARNING: ESO states have insufficient dimensions for logging")
                print(f"  eso_states_x length: {len(eso_states_x)}")
                print(f"  eso_states_y length: {len(eso_states_y)}")
                return

            # X-axis ESO data - eso_states_x should be [pos_est, vel_est, dist_est]
            writer.writerow([
                episode, step, agent_id, 'x',
                float(true_states_x[0]), float(eso_states_x[0]), float(eso_states_x[0] - true_states_x[0]),  # position
                float(true_states_x[1]), float(eso_states_x[1]), float(eso_states_x[1] - true_states_x[1]),  # velocity
                float(true_dist_x), float(eso_states_x[2]), float(eso_states_x[2] - true_dist_x),  # disturbance
                float(eso_states_x[0] - true_states_x[0])  # measurement error
            ])

            # Y-axis ESO data - eso_states_y should be [pos_est, vel_est, dist_est]
            writer.writerow([
                episode, step, agent_id, 'y',
                float(true_states_y[0]), float(eso_states_y[0]), float(eso_states_y[0] - true_states_y[0]),  # position
                float(true_states_y[1]), float(eso_states_y[1]), float(eso_states_y[1] - true_states_y[1]),  # velocity
                float(true_dist_y), float(eso_states_y[2]), float(eso_states_y[2] - true_dist_y),  # disturbance
                float(eso_states_y[0] - true_states_y[0])  # measurement error
            ])
            f.flush()  # MEMORY OPT: Force write to disk immediately

    def log_disturbances(self, episode: int, step: int, agent_id: int,
                         true_dist_x: float, true_dist_y: float,
                         eso_dist_x: float, eso_dist_y: float):
        """RESTORED: Log disturbance estimates vs true values (EVERY STEP)."""

        with open(self.disturbance_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, step, agent_id,
                float(true_dist_x), float(true_dist_y),
                float(eso_dist_x), float(eso_dist_y)
            ])
            f.flush()  # MEMORY OPT: Force write to disk immediately

    def log_mpc_result(self, episode: int, step: int, agent_id: int,
                       solver_status: str, solve_time: float, action: np.ndarray,
                       feasible: bool, emergency_brake: bool = False,
                       eso_action: np.ndarray = None, compensated_action: np.ndarray = None):
        """RESTORED: Log MPC solver results and actions with ESO data (EVERY STEP)."""

        with open(self.mpc_log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            eso_x = float(eso_action[0]) if eso_action is not None else 0.0
            eso_y = float(eso_action[1]) if eso_action is not None else 0.0
            comp_x = float(compensated_action[0]) if compensated_action is not None else 0.0
            comp_y = float(compensated_action[1]) if compensated_action is not None else 0.0

            writer.writerow([
                episode, step, agent_id, solver_status, solve_time,
                float(action[0]), float(action[1]), feasible, emergency_brake,
                eso_x, eso_y, comp_x, comp_y
            ])
            f.flush()  # MEMORY OPT: Force write to disk immediately


class ESOSafetyMPC:
    """UNIVERSAL COMPENSATION + GRAPH PREDICTION: MPC controller with ESO feedback, disturbance compensation, and graph prediction over horizon."""

    def __init__(self,
                 gcbf_interface: GCBFInterface,
                 horizon: int = 20,
                 dt: float = 0.03,
                 u_max: float = 2.0,
                 alpha: float = 1.0,
                 margin: float = 0.0,
                 goal_weight: float = 1.0,
                 debug_logger: Optional[ESODebugLogger] = None,
                 # ESO parameters
                 eso_dt: float = 0.01,
                 mass: float = 0.1,
                 b0: float = 1.0):

        self.gcbf = gcbf_interface
        self.horizon = horizon
        self.dt = dt
        self.u_max = u_max
        self.alpha = alpha
        self.margin = margin
        self.goal_weight = goal_weight
        self.debug_logger = debug_logger

        # ESO parameters
        self.eso_dt = eso_dt
        self.mass = mass
        self.b0 = b0

        # Keep v6 performance optimization
        self.eso_per_plant = max(1, int(dt / eso_dt))  # Now 3 instead of 30

        # ESO gains from Han 2009 paper equation (23) for nonlinear ESO
        self.beta01 = 1.0 * 10
        self.beta02 = 1.0 / (2.0 * self.dt ** 0.5) * 0.7
        self.beta03 = 2.0 / (25.0 * self.dt ** 1.2) * 0.01

        # Nonlinear function parameters
        self.delta = eso_dt
        self.alpha1 = 0.5
        self.alpha2 = 0.25

        # Initialize ESO states for all agents [position, velocity, disturbance]
        num_agents = gcbf_interface.env.num_agents
        self.eso_states_x = np.zeros((num_agents, 3))  # X-axis ESO states
        self.eso_states_y = np.zeros((num_agents, 3))  # Y-axis ESO states
        self.prev_actions = np.zeros((num_agents, 2))

        # RESTORED v3 DEBUGGING: Verify initialization
        print(f"ESO Initialization Debug:")
        print(f"  Number of agents: {num_agents}")
        print(f"  ESO states X shape: {self.eso_states_x.shape}")
        print(f"  ESO states Y shape: {self.eso_states_y.shape}")
        print(f"  Sample ESO state X[0]: {self.eso_states_x[0]}")
        print(f"  Sample ESO state Y[0]: {self.eso_states_y[0]}")

        # Cost matrices for double integrator [x, y, vx, vy]
        self.Q = np.diag([1.0, 1.0, 0.1, 0.1])  # Goal-reaching penalty (relaxable)
        self.R = np.eye(2) * 0.01  # Control effort penalty

        print(f"UNIVERSAL COMPENSATION ESO-MPC Controller with Graph Prediction initialized:")
        print(f"  🔮 GRAPH PREDICTION: Enabled over MPC horizon")
        print(f"     ✅ Agent trajectory prediction: Nominal goal-seeking")
        print(f"     ✅ Obstacle motion: Constant velocity extrapolation")
        print(f"     ✅ Edge features: Recomputed at each horizon step")
        print(f"     ✅ CBF consistency: Maintained across prediction horizon")
        print(f"  ESO sampling: {eso_dt}s ({1 / eso_dt:.0f} Hz)")
        print(f"  Plant sampling: {dt}s ({1 / dt:.0f} Hz)")
        print(f"  ESO updates per plant step: {self.eso_per_plant}")
        print(f"  Agent mass: {mass} kg")
        print(f"  ESO gains: β01={self.beta01:.3f}, β02={self.beta02:.3f}, β03={self.beta03:.3f}")
        print(f"  🆕 UNIVERSAL DISTURBANCE COMPENSATION: Available (controlled by external flag)")
        print(f"     ✅ All control actions: Compensation applied uniformly")
        print(f"     ✅ No action type differentiation: Simple universal approach")

    def generate_disturbances(self, t: float, agent_idx: int) -> Tuple[float, float]:
        """Generate disturbances for agent at time t (acceleration units)."""
        # Keep v6 reduced frequencies for stability
        freq_x = 2 * np.pi * 0.05
        freq_y = 2 * np.pi * 0.05

        gamma_x = 0
        gamma_y = 0

        dx = gamma_x * np.sign(np.sin(freq_x * 0.03 * t))
        dy = gamma_y * np.sign(np.sin(freq_y * 0.03 * t))

        return dx, dy

    def _apply_universal_compensation(self, raw_action: np.ndarray, agent_idx: int, step: int = 0,
                                      is_emergency: bool = False) -> np.ndarray:
        """
        UNIVERSAL COMPENSATION: Apply disturbance compensation to ANY control action if enabled.

        Args:
            raw_action: Raw control action [ax, ay] in acceleration units
            agent_idx: Agent index
            step: Current step for debugging
            is_emergency: Whether this is an emergency action

        Returns:
            compensated_action: Action with disturbance compensation applied
        """
        if not ENABLE_DISTURBANCE_COMPENSATION:
            return raw_action.copy()

        # Ensure raw_action is a proper numpy array
        if not isinstance(raw_action, np.ndarray):
            raw_action = np.array(raw_action)
        if raw_action.size != 2:
            print(f"Warning: Action size {raw_action.size} != 2 for agent {agent_idx}, using zeros")
            raw_action = np.zeros(2)

        # Check if ESO states are available and valid
        if (len(self.eso_states_x[agent_idx]) >= 3 and
                len(self.eso_states_y[agent_idx]) >= 3):

            try:
                # Apply compensation to both axes
                compensated_x = disturbance_compensation(
                    raw_action[0], self.eso_states_x[agent_idx][2], self.b0
                )
                compensated_y = disturbance_compensation(
                    raw_action[1], self.eso_states_y[agent_idx][2], self.b0
                )

                # Debug output for first few steps or emergency situations
                if step < 3 or is_emergency:
                    action_type = "EMERGENCY" if is_emergency else "NORMAL"
                    print(
                        f"  🔧 Agent {agent_idx}: {action_type} compensation applied - Raw:[{raw_action[0]:.3f},{raw_action[1]:.3f}] → Comp:[{compensated_x:.3f},{compensated_y:.3f}]")

                return np.array([compensated_x, compensated_y])

            except Exception as e:
                print(f"Warning: Compensation failed for agent {agent_idx}: {e}")
                return raw_action.copy()
        else:
            return raw_action.copy()

    def update_eso(self, agent_idx: int, measurements: np.ndarray,
                   control_input_accel: np.ndarray, step: int) -> None:
        """Update ESO observers for both axes of an agent (with RESTORED v3 debugging)."""

        pos_x, pos_y = measurements[0], measurements[1]
        u_accel_x, u_accel_y = control_input_accel[0], control_input_accel[1]

        # DEBUGGING: Check ESO state dimensions before update
        if step == 0 and agent_idx == 0:
            print(f"DEBUG update_eso: ESO states shapes before update:")
            print(f"  eso_states_x[{agent_idx}] shape: {self.eso_states_x[agent_idx].shape}")
            print(f"  eso_states_y[{agent_idx}] shape: {self.eso_states_y[agent_idx].shape}")

        # SAFETY CHECK: Ensure ESO states have correct dimensions
        if self.eso_states_x[agent_idx].shape[0] != 3:
            print(f"ERROR: ESO state X for agent {agent_idx} has wrong shape: {self.eso_states_x[agent_idx].shape}")
            self.eso_states_x[agent_idx] = np.array([pos_x, 0.0, 0.0])  # Reinitialize

        if self.eso_states_y[agent_idx].shape[0] != 3:
            print(f"ERROR: ESO state Y for agent {agent_idx} has wrong shape: {self.eso_states_y[agent_idx].shape}")
            self.eso_states_y[agent_idx] = np.array([pos_y, 0.0, 0.0])  # Reinitialize

        try:
            # Update X-axis ESO
            eso_result_x, error_x = nonlinear_eso(
                pos_x, u_accel_x, self.eso_states_x[agent_idx],
                self.eso_dt, self.beta01, self.beta02, self.beta03,
                self.alpha1, self.alpha2, self.delta, self.b0
            )

            # SAFETY CHECK: Ensure result has correct dimension
            if len(eso_result_x) != 3:
                print(f"ERROR: ESO update returned wrong dimension for agent {agent_idx} X-axis: {len(eso_result_x)}")
                print(f"  Result: {eso_result_x}")
                # Keep previous state if update failed
            else:
                self.eso_states_x[agent_idx] = eso_result_x

            # Update Y-axis ESO
            eso_result_y, error_y = nonlinear_eso(
                pos_y, u_accel_y, self.eso_states_y[agent_idx],
                self.eso_dt, self.beta01, self.beta02, self.beta03,
                self.alpha1, self.alpha2, self.delta, self.b0
            )

            # SAFETY CHECK: Ensure result has correct dimension
            if len(eso_result_y) != 3:
                print(f"ERROR: ESO update returned wrong dimension for agent {agent_idx} Y-axis: {len(eso_result_y)}")
                print(f"  Result: {eso_result_y}")
                # Keep previous state if update failed
            else:
                self.eso_states_y[agent_idx] = eso_result_y

        except Exception as e:
            print(f"ERROR in update_eso for agent {agent_idx}: {e}")
            import traceback
            traceback.print_exc()

        # DEBUGGING: Check ESO state dimensions after update
        if step == 0 and agent_idx == 0:
            print(f"DEBUG update_eso: ESO states shapes after update:")
            print(f"  eso_states_x[{agent_idx}] shape: {self.eso_states_x[agent_idx].shape}")
            print(f"  eso_states_y[{agent_idx}] shape: {self.eso_states_y[agent_idx].shape}")
            print(f"  eso_states_x[{agent_idx}] content: {self.eso_states_x[agent_idx]}")
            print(f"  eso_states_y[{agent_idx}] content: {self.eso_states_y[agent_idx]}")

    def _predict_nominal_trajectory(self, current_state: np.ndarray, goal_state: np.ndarray,
                                    current_graph: GraphsTuple, agent_idx: int) -> np.ndarray:
        """
        Generate nominal trajectory using CBF-constrained PID for first step, then propagate.

        Uses current graph to compute ONE CBF-filtered control action, then applies that
        action constantly over the horizon to generate nominal states for graph prediction.

        Args:
            current_state: [x, y, vx, vy] current agent state
            goal_state: [x, y, vx, vy] goal state
            current_graph: Current graph from environment (only current CBF available)
            agent_idx: Index of agent for CBF constraint lookup

        Returns:
            x_nom: (horizon+1, 4) nominal state trajectory using CBF-filtered first step
        """
        x_nom = np.zeros((self.horizon + 1, 4))
        x_nom[0] = current_state.copy()

        # Step 1: Compute PID tracking control for current state
        pos_error = goal_state[:2] - current_state[:2]
        vel_error = goal_state[2:4] - current_state[2:4]

        # PID controller gains (increased from simple P controller)
        Kp = 0.8
        Kd = 0.4
        u_pid = Kp * pos_error + Kd * vel_error
        u_pid = np.clip(u_pid, -self.u_max, self.u_max)

        # Step 2: Get current CBF constraint from current graph
        try:
            h_values = self.gcbf.evaluate_h(current_graph)
            h_current = float(h_values[agent_idx].item())
            grad_h_current = self.gcbf.get_cbf_gradient(current_graph, agent_idx)

            # Step 3: Solve CBF filter QP using current constraint only
            if not np.allclose(grad_h_current, 0) and len(grad_h_current) >= 4:
                # CBF constraint: h_dot + alpha * h >= margin
                # h_dot = grad_h[0:2] @ vel + grad_h[2:4] @ accel
                drift_term = grad_h_current[0] * current_state[2] + grad_h_current[1] * current_state[3]
                b_current = self.margin - self.alpha * h_current - drift_term
                a_current = grad_h_current[2:4]  # Control coefficients

                # Solve small CBF filter QP
                u_var = cp.Variable(2)
                s_var = cp.Variable(1, nonneg=True)  # Slack variable

                # Objective: minimize deviation from PID + control effort + slack penalty
                cost = (0.5 * cp.sum_squares(u_var - u_pid) +
                        0.01 * cp.sum_squares(u_var) +
                        1000.0 * s_var)

                # Constraints
                constraints = [
                    a_current @ u_var >= b_current - s_var,  # CBF constraint with slack
                    u_var >= -self.u_max,  # Input bounds
                    u_var <= self.u_max
                ]

                # Solve QP
                prob = cp.Problem(cp.Minimize(cost), constraints)
                try:
                    prob.solve(solver=cp.OSQP, verbose=False)
                    if prob.status in ["optimal", "optimal_inaccurate"]:
                        u_nom = u_var.value
                        if u_nom is not None:
                            u_nom = np.array(u_nom).flatten()
                        else:
                            u_nom = u_pid  # Fallback
                    else:
                        u_nom = u_pid  # Fallback if infeasible
                except:
                    u_nom = u_pid  # Fallback if solver failed
            else:
                u_nom = u_pid  # Fallback if gradient invalid

        except Exception as e:
            u_nom = u_pid  # Fallback if CBF evaluation failed

        # Step 4: Propagate forward using the single CBF-filtered control action
        # Apply constant control over entire horizon
        for k in range(self.horizon):
            # Double integrator dynamics with constant u_nom
            x_nom[k + 1, 0] = x_nom[k, 0] + x_nom[k, 2] * self.dt + 0.5 * u_nom[0] * self.dt ** 2
            x_nom[k + 1, 1] = x_nom[k, 1] + x_nom[k, 3] * self.dt + 0.5 * u_nom[1] * self.dt ** 2
            x_nom[k + 1, 2] = x_nom[k, 2] + u_nom[0] * self.dt
            x_nom[k + 1, 3] = x_nom[k, 3] + u_nom[1] * self.dt

        return x_nom

    def _predict_graph_at_step(self, base_graph: GraphsTuple, all_agent_states: np.ndarray,
                               x_nom_all: np.ndarray, step_k: int) -> GraphsTuple:
        """
        Predict graph state at MPC step k using nominal trajectories and obstacle extrapolation.

        Args:
            base_graph: Current graph from environment
            all_agent_states: Current states of all agents (n_agents, 4)
            x_nom_all: Nominal trajectories for all agents (n_agents, horizon+1, 4)
            step_k: MPC step index (0 to horizon)

        Returns:
            predicted_graph: Graph with predicted states at step k
        """
        # Create a copy of the base graph
        predicted_graph = base_graph

        # Get node masks for different entity types
        agent_mask = base_graph.node_type == 0
        goal_mask = base_graph.node_type == 1
        obstacle_mask = base_graph.node_type == 2

        # Update agent states with nominal trajectory predictions
        if np.any(agent_mask):
            agent_indices = np.where(agent_mask)[0]
            current_agent_states = predicted_graph.states[agent_mask]

            for i, agent_idx in enumerate(agent_indices):
                if i < len(x_nom_all):
                    # Use nominal trajectory prediction
                    predicted_state = current_agent_states[i]

                    # Update position and velocity from nominal trajectory
                    predicted_state = predicted_state.at[0].set(x_nom_all[i, step_k, 0])  # x pos
                    predicted_state = predicted_state.at[1].set(x_nom_all[i, step_k, 1])  # y pos
                    predicted_state = predicted_state.at[2].set(x_nom_all[i, step_k, 2])  # x vel
                    predicted_state = predicted_state.at[3].set(x_nom_all[i, step_k, 3])  # y vel

                    # Keep other state dimensions unchanged (z, angles, etc.)
                    current_agent_states = current_agent_states.at[i].set(predicted_state)

            # Update the graph with predicted agent states
            new_states = predicted_graph.states.at[agent_mask].set(current_agent_states)
            predicted_graph = predicted_graph._replace(states=new_states)

        # Extrapolate obstacle positions with constant velocity
        if np.any(obstacle_mask):
            obstacle_states = predicted_graph.states[obstacle_mask]
            dt_pred = step_k * self.dt  # Time into the future

            predicted_obstacle_states = obstacle_states

            # For each obstacle, extrapolate position based on current velocity
            for i in range(len(obstacle_states)):
                current_obs_state = obstacle_states[i]

                # Extract current position and velocity (assuming similar state structure)
                if len(current_obs_state) >= 4:  # Has velocity info
                    pos_x = current_obs_state[0] + current_obs_state[2] * dt_pred
                    pos_y = current_obs_state[1] + current_obs_state[3] * dt_pred

                    predicted_obs_state = current_obs_state.at[0].set(pos_x)
                    predicted_obs_state = predicted_obs_state.at[1].set(pos_y)
                    predicted_obstacle_states = predicted_obstacle_states.at[i].set(predicted_obs_state)

            # Update graph with predicted obstacle states
            new_states = predicted_graph.states.at[obstacle_mask].set(predicted_obstacle_states)
            predicted_graph = predicted_graph._replace(states=new_states)

        # Goals typically don't move, so keep them unchanged

        # Recompute edge features with updated states
        predicted_graph = self.gcbf.env.add_edge_feats(predicted_graph, predicted_graph.states)

        return predicted_graph

    def _add_cbf_constraint_with_predicted_graph(self, predicted_graph: GraphsTuple, agent_idx: int,
                                                 predicted_state: cp.Variable, control: cp.Variable,
                                                 step: int) -> Optional[cp.Constraint]:
        """Add CBF constraint using predicted graph state: ḣ + α*h ≥ margin."""

        try:
            # Get CBF value and gradient from predicted graph
            h_values = self.gcbf.evaluate_h(predicted_graph)
            h_current = float(h_values[agent_idx].item())
            cbf_grad = self.gcbf.get_cbf_gradient(predicted_graph, agent_idx)

            # Check if gradient computation succeeded
            if np.allclose(cbf_grad, 0):
                return None

            # Double integrator: ẋ = [vx, vy, ax, ay]
            # ḣ = ∇h · ẋ = grad[0]*vx + grad[1]*vy + grad[2]*ax + grad[3]*ay

            # Drift term (predicted velocity contribution)
            drift_term = cbf_grad[0] * predicted_state[2] + cbf_grad[1] * predicted_state[3]

            # Control term (acceleration contribution)
            control_term = cbf_grad[2] * control[0] + cbf_grad[3] * control[1]

            # CBF constraint with MARGIN: ḣ + α*h ≥ margin
            cbf_constraint = (drift_term + control_term + self.alpha * h_current >= self.margin)

            # DEBUG info for first step
            if step == 0:
                margin_status = "MARGIN VIOLATED" if h_current < self.margin else "SAFE"
                print(
                    f"    Agent {agent_idx}: h={h_current:.4f} (predicted), margin={self.margin:.4f} [{margin_status}]")

            return cbf_constraint

        except Exception as e:
            if step == 0:
                print(f"    Agent {agent_idx}: Predicted CBF constraint failed at step {step}: {e}")
            return None

    def solve(self, agent_states: np.ndarray, goal_states: np.ndarray,
              graph: GraphsTuple, episode: int = 0, step: int = 0,
              true_disturbances: List[Tuple[float, float]] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Solve ESO-enhanced MPC for all agents with UNIVERSAL disturbance compensation and GRAPH PREDICTION."""

        n_agents = agent_states.shape[0]
        actions = np.zeros((n_agents, 2))
        compensated_actions = np.zeros((n_agents, 2))
        cbf_gradients = []

        # Get CBF values for all agents for logging
        h_values = self.gcbf.evaluate_h(graph)

        # Collect gradients for logging
        for i in range(n_agents):
            try:
                grad = self.gcbf.get_cbf_gradient(graph, i)
                cbf_gradients.append(grad)
            except:
                cbf_gradients.append(np.zeros(4))

        # LOGGING: Log current state before solving MPC
        if self.debug_logger:
            self.debug_logger.log_episode_state(
                episode, step, agent_states, goal_states, h_values, cbf_gradients, graph
            )

        # Initialize ESO with actual agent states on first step
        if step == 0:
            for i in range(n_agents):
                self.eso_states_x[i] = np.array([agent_states[i, 0], agent_states[i, 2], 0.0])  # [x_pos, x_vel, x_dist]
                self.eso_states_y[i] = np.array([agent_states[i, 1], agent_states[i, 3], 0.0])  # [y_pos, y_vel, y_dist]
                self.prev_actions[i] = np.zeros(2)
                print(f"Initialized ESO for agent {i}:")
                print(
                    f"  True state: pos=({agent_states[i, 0]:.3f}, {agent_states[i, 1]:.3f}), vel=({agent_states[i, 2]:.3f}, {agent_states[i, 3]:.3f})")
                print(f"  ESO X: {self.eso_states_x[i]}")
                print(f"  ESO Y: {self.eso_states_y[i]}")

        # Update ESO for all agents based on current measurements
        for i in range(n_agents):
            current_pos = agent_states[i, :2]
            previous_accel = self.prev_actions[i]

            # PERFORMANCE OPTIMIZED: Only do a few ESO updates instead of 30
            for sub_step in range(self.eso_per_plant):
                self.update_eso(i, current_pos, previous_accel, step * self.eso_per_plant + sub_step)

        # Solve MPC for each agent using ESO feedback
        for i in range(n_agents):
            start_time = time.time()

            try:
                # DEBUGGING: Check ESO state dimensions
                if step == 0 and i == 0:
                    print(f"DEBUG: ESO states shapes: x={self.eso_states_x.shape}, y={self.eso_states_y.shape}")
                    print(
                        f"DEBUG: ESO state x[{i}] shape: {self.eso_states_x[i].shape}, content: {self.eso_states_x[i]}")
                    print(
                        f"DEBUG: ESO state y[{i}] shape: {self.eso_states_y[i].shape}, content: {self.eso_states_y[i]}")

                # Get ESO estimates for feedback with safety checks
                eso_state_x = self.eso_states_x[i]
                eso_state_y = self.eso_states_y[i]

                # SAFETY CHECK: Ensure ESO states have correct dimension
                if len(eso_state_x) != 3 or len(eso_state_y) != 3:
                    print(f"ERROR: ESO state dimension mismatch for agent {i}")
                    print(f"  eso_state_x length: {len(eso_state_x)}, expected: 3")
                    print(f"  eso_state_y length: {len(eso_state_y)}, expected: 3")
                    print(f"  eso_state_x: {eso_state_x}")
                    print(f"  eso_state_y: {eso_state_y}")

                    # Reinitialize ESO states if corrupted
                    print(f"  Reinitializing ESO states for agent {i}")
                    self.eso_states_x[i] = np.array([agent_states[i, 0], agent_states[i, 2], 0.0])  # [pos, vel, dist]
                    self.eso_states_y[i] = np.array([agent_states[i, 1], agent_states[i, 3], 0.0])  # [pos, vel, dist]
                    eso_state_x = self.eso_states_x[i]
                    eso_state_y = self.eso_states_y[i]

                # DEBUG: Compare ESO estimates vs true states
                true_state = agent_states[i]  # [x, y, vx, vy]
                eso_pos_error_x = abs(eso_state_x[0] - true_state[0])
                eso_pos_error_y = abs(eso_state_y[0] - true_state[1])
                eso_vel_error_x = abs(eso_state_x[1] - true_state[2])
                eso_vel_error_y = abs(eso_state_y[1] - true_state[3])

                if step % 20 == 0 or step < 3:
                    print(f"Agent {i} State Comparison (Step {step}):")
                    print(
                        f"  True:  pos=({true_state[0]:.3f}, {true_state[1]:.3f}), vel=({true_state[2]:.3f}, {true_state[3]:.3f})")
                    print(
                        f"  ESO:   pos=({eso_state_x[0]:.3f}, {eso_state_y[0]:.3f}), vel=({eso_state_x[1]:.3f}, {eso_state_y[1]:.3f})")
                    print(
                        f"  Error: pos=({eso_pos_error_x:.3f}, {eso_pos_error_y:.3f}), vel=({eso_vel_error_x:.3f}, {eso_vel_error_y:.3f})")
                    print(f"  Dist:  est=({eso_state_x[2]:.3f}, {eso_state_y[2]:.3f})")

                # Use ESO feedback
                if USE_ESO_FEEDBACK:
                    mpc_feedback_state = np.array([
                        eso_state_x[0],  # x position estimate
                        eso_state_y[0],  # y position estimate
                        eso_state_x[1],  # x velocity estimate
                        eso_state_y[1]  # y velocity estimate
                    ])
                    if step < 3:
                        print(f"  🔧 Using ESO feedback for MPC")
                else:
                    print(f"  🔧 DEBUGGING: Using TRUE states for MPC (bypassing ESO)")
                    mpc_feedback_state = agent_states[i]  # Use true states

                raw_action, solver_status, feasible = self._solve_single_agent(
                    mpc_feedback_state, goal_states[i], graph, i, agent_states, episode, step
                )

                # UNIVERSAL COMPENSATION: Apply compensation to ALL actions
                compensated_action = self._apply_universal_compensation(raw_action, i, step)

                actions[i] = raw_action
                compensated_actions[i] = compensated_action
                solve_time = time.time() - start_time
                emergency_brake = False

                if step < 3:  # Print actions for first few steps
                    print(
                        f"  Action: raw=({raw_action[0]:.3f}, {raw_action[1]:.3f}), compensated=({compensated_action[0]:.3f}, {compensated_action[1]:.3f})")

                # LOGGING: Log ESO performance
                if self.debug_logger and true_disturbances and len(eso_state_x) >= 3 and len(eso_state_y) >= 3:
                    true_dist_x, true_dist_y = true_disturbances[i]
                    true_state_x = agent_states[i, [0, 2]]  # [pos_x, vel_x]
                    true_state_y = agent_states[i, [1, 3]]  # [pos_y, vel_y]

                    # Pass full ESO states (all 3 elements)
                    self.debug_logger.log_eso_data(
                        episode, step, i,
                        true_state_x, true_state_y,
                        eso_state_x, eso_state_y,  # ← Pass full states (3 elements each)
                        true_dist_x, true_dist_y
                    )

                    self.debug_logger.log_disturbances(
                        episode, step, i,
                        true_dist_x, true_dist_y,
                        eso_state_x[2], eso_state_y[2]
                    )

            except Exception as e:
                solve_time = time.time() - start_time
                print(f"Agent {i} ESO-MPC failed: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace for debugging

                # UNIVERSAL COMPENSATION: Apply compensation to emergency brake actions
                current_vel = agent_states[i, 2:4]
                emergency_raw_action = -np.sign(current_vel) * min(self.u_max, np.linalg.norm(current_vel))
                emergency_raw_action = np.clip(emergency_raw_action, -self.u_max, self.u_max)

                # Apply compensation to emergency action
                emergency_compensated_action = self._apply_universal_compensation(emergency_raw_action, i, step,
                                                                                  is_emergency=True)

                actions[i] = emergency_raw_action
                compensated_actions[i] = emergency_compensated_action

                solver_status = f"FAILED: {str(e)[:50]}"
                feasible = False
                emergency_brake = True

            # LOGGING: Log MPC result
            if self.debug_logger:
                self.debug_logger.log_mpc_result(
                    episode, step, i, solver_status, solve_time, actions[i], feasible, emergency_brake,
                    actions[i], compensated_actions[i]
                )

        # Store current actions for next iteration (v6 optimization)
        self.prev_actions = compensated_actions.copy()
        return compensated_actions, cbf_gradients

    def _solve_single_agent(self, current_state: np.ndarray, goal_state: np.ndarray,
                            graph: GraphsTuple, agent_idx: int,
                            all_agent_states: np.ndarray, episode: int, step: int) -> Tuple[np.ndarray, str, bool]:
        """Solve safety-first MPC for single agent using ESO feedback with PREDICTED GRAPHS."""

        # Decision variables
        x = cp.Variable((self.horizon + 1, 4))  # [x, y, vx, vy]
        u = cp.Variable((self.horizon, 2))  # [ax, ay]

        # Objective: Minimize goal deviation + control effort
        cost = 0
        constraints = []

        # Initial condition (using ESO estimates)
        constraints += [x[0] == current_state]

        # STEP 1: Generate nominal trajectories for all agents for graph prediction
        x_nom_all = []
        for i in range(len(all_agent_states)):
            if i == agent_idx:
                # For current agent, use CBF-constrained PID (filter first step, then propagate)
                x_nom_agent = self._predict_nominal_trajectory(
                    current_state, goal_state, graph, agent_idx
                )
            else:
                # For other agents, predict with their current velocity extrapolation
                other_current = all_agent_states[i]
                other_goal = other_current.copy()
                other_goal[:2] += other_current[2:4] * self.dt * self.horizon  # Extrapolate position
                # Use simple nominal trajectory for other agents (no CBF filtering needed)
                # x_nom_agent = self._predict_simple_nominal_trajectory(other_current, other_goal)
                x_nom_agent = self._predict_nominal_trajectory(
                    other_current, other_goal, graph, i
                )
            x_nom_all.append(x_nom_agent)

        x_nom_all = np.array(x_nom_all)  # (n_agents, horizon+1, 4)

        # STEP 2: Build predicted graphs and constraints over horizon
        for k in range(self.horizon):
            # Double integrator dynamics constraints
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

            # STEP 3: HARD SAFETY CONSTRAINTS with PREDICTED GRAPH
            # Create predicted graph for step k+1 (future state)
            try:
                # Update the nominal trajectory with current MPC prediction for this agent
                # (Use nominal for other agents, but incorporate MPC variables for current agent)
                x_nom_current = x_nom_all.copy()

                predicted_graph_k = self._predict_graph_at_step(graph, all_agent_states, x_nom_current, k + 1)

                # Use predicted graph for CBF constraint
                safety_constraint = self._add_cbf_constraint_with_predicted_graph(
                    predicted_graph_k, agent_idx, x[k + 1], u[k], k
                )
                if safety_constraint is not None:
                    constraints += [safety_constraint]

            except Exception as e:
                if step < 3:  # Only print for early steps to avoid spam
                    print(f"    Warning: Graph prediction failed at step {k}: {e}")
                # Fallback to original graph if prediction fails
                safety_constraint = self._add_cbf_constraint(graph, agent_idx, current_state, u[k], k)
                if safety_constraint is not None:
                    constraints += [safety_constraint]

        # Terminal cost (soft penalty)
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
                    if step == 0:
                        print(
                            f"    Agent {agent_idx}: ESO-MPC solved successfully with predicted graphs ({solver_status})")
                    # FIXED: Ensure we return a proper numpy array
                    optimal_u = u[0].value
                    if optimal_u is None:
                        optimal_u = np.zeros(2)
                    return np.array(optimal_u).flatten(), solver_status, feasible

            except Exception as e:
                solver_status = f"{solver_name}_ERROR: {str(e)[:30]}"
                continue

        # UNIVERSAL COMPENSATION: If all solvers fail, create emergency brake and apply compensation
        print(f"  Agent {agent_idx}: ESO-MPC with predicted graphs infeasible ({solver_status}) - emergency brake")
        current_vel = current_state[2:4]
        brake_action = -2 * current_vel
        emergency_raw_action = np.clip(brake_action, -self.u_max, self.u_max)

        # Note: Compensation will be applied at the higher level in solve()
        return emergency_raw_action, f"INFEASIBLE_{solver_status}", False

    def _add_cbf_constraint(self, graph: GraphsTuple, agent_idx: int,
                            agent_state: np.ndarray, control: cp.Variable,
                            step: int) -> Optional[cp.Constraint]:
        """Add CBF constraint with safety margin: ḣ + α*h ≥ margin (FALLBACK VERSION)."""

        try:
            # Get current CBF value and gradient
            h_values = self.gcbf.evaluate_h(graph)
            h_current = float(h_values[agent_idx].item())
            cbf_grad = self.gcbf.get_cbf_gradient(graph, agent_idx)

            # Check if gradient computation succeeded
            if np.allclose(cbf_grad, 0):
                return None

            # Double integrator: ẋ = [vx, vy, ax, ay]
            # ḣ = ∇h · ẋ = grad[0]*vx + grad[1]*vy + grad[2]*ax + grad[3]*ay

            # Drift term (current velocity contribution)
            drift_term = cbf_grad[0] * agent_state[2] + cbf_grad[1] * agent_state[3]

            # Control term (acceleration contribution)
            control_term = cbf_grad[2] * control[0] + cbf_grad[3] * control[1]

            # CBF constraint with MARGIN: ḣ + α*h ≥ margin
            cbf_constraint = (drift_term + control_term + self.alpha * h_current >= self.margin)

            # DEBUG info for first step
            if step == 0:
                margin_status = "MARGIN VIOLATED" if h_current < self.margin else "SAFE"
                print(
                    f"    Agent {agent_idx}: h={h_current:.4f} (fallback), margin={self.margin:.4f} [{margin_status}]")

            return cbf_constraint

        except Exception as e:
            if step == 0:
                print(f"    Agent {agent_idx}: CBF constraint failed at step {step}: {e}")
            return None


class ESOGCBFTester:
    """MEMORY OPT: Testing framework for ESO+GCBF approach with selective memory management."""

    def __init__(self, env, mpc_controller: ESOSafetyMPC, debug_logger: Optional[ESODebugLogger] = None):
        self.env = env
        self.controller = mpc_controller
        self.debug_logger = debug_logger
        self.margin = mpc_controller.margin

        # Evaluation functions
        self.is_unsafe_fn = jax_jit_np(jax.vmap(env.collision_mask))
        self.is_finish_fn = jax_jit_np(jax.vmap(env.finish_mask))

    def run_episode(self, key: jr.PRNGKey, max_steps: int, episode_id: int = 0) -> Tuple[
        RolloutResult, List[np.ndarray], dict]:
        """Run single episode with ESO disturbance estimation and GCBF safety (with RESTORED v3 logging)."""

        # Initialize
        graph = self.env.reset(key)
        h_trace = []
        episode_stats = {
            'cbf_violations': 0,
            'margin_violations': 0,
            'emergency_brakes': 0,
            'min_h_overall': float('inf'),
            'min_h_margin_adjusted': float('inf'),
            'steps_completed': 0,
            'total_disturbance_error_x': 0.0,
            'total_disturbance_error_y': 0.0,
            'eso_convergence_steps': 0
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

            # Generate disturbances for each agent
            true_disturbances = []
            for i in range(self.env.num_agents):
                dx_accel, dy_accel = self.controller.generate_disturbances(step, i)
                true_disturbances.append((dx_accel, dy_accel))

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

            # RESTORED v3 CBF STATUS PRINTING (but with v6 performance balance)
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

            # Get ESO-enhanced MPC control action with graph prediction
            action_accel, cbf_gradients = self.controller.solve(
                agent_states, goal_states, graph, episode_id, step, true_disturbances
            )

            # Convert to forces and step environment
            action_forces = action_accel * self.controller.mass
            action_jax = jnp.array(action_forces)

            # Step environment
            step_result = self.env.step(graph, action_jax, get_eval_info=True)
            graph, reward, cost, done, info = step_result

            # Apply disturbances to plant states (simulate real disturbances)
            if hasattr(graph, 'states'):
                agent_node_mask = graph.node_type == 0
                if np.any(agent_node_mask):
                    agent_states_current = graph.states[agent_node_mask]

                    for i in range(self.env.num_agents):
                        dx_accel, dy_accel = true_disturbances[i]
                        current_state = agent_states_current[i]

                        # Apply disturbance as velocity change
                        updated_state = current_state.at[2].set(
                            current_state[2] + dx_accel * self.env.dt
                        )
                        updated_state = updated_state.at[3].set(
                            updated_state[3] + dy_accel * self.env.dt
                        )
                        agent_states_current = agent_states_current.at[i].set(updated_state)

                    new_states = graph.states.at[agent_node_mask].set(agent_states_current)
                    graph = graph._replace(states=new_states)

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


def generate_eso_analysis_plots(h_traces: List[List[np.ndarray]], output_dir: pathlib.Path, margin: float = 0.0):
    """Generate comprehensive ESO+CBF analysis plots (RESTORED from v3 with MEMORY OPT: immediate cleanup)."""

    plt.figure(figsize=(20, 16))

    # Plot 1: CBF evolution over time
    plt.subplot(3, 4, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(h_traces), 10)))  # Limit to avoid memory issues

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
    plt.title('ESO+GCBF+Graph Prediction: CBF Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: CBF violation timeline
    plt.subplot(3, 4, 2)
    for i, h_trace in enumerate(h_traces):
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
    plt.title('ESO+GCBF+Graph Prediction: CBF Violations Timeline')
    plt.yticks(range(1, min(len(h_traces) + 1, 11)))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: CBF distribution histogram
    plt.subplot(3, 4, 3)
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
        plt.title('ESO+GCBF+Graph Prediction: CBF Value Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Add placeholder plots for ESO-specific analysis
    plt.subplot(3, 4, 4)
    plt.text(0.5, 0.5, 'Graph Prediction\nConsistency\nAnalysis',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Graph Prediction Performance')

    plt.subplot(3, 4, 5)
    plt.text(0.5, 0.5, 'MPC Feasibility\nImprovement\n(With/Without Prediction)',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Feasibility Analysis')

    plt.subplot(3, 4, 6)
    plt.text(0.5, 0.5, 'Safety Constraint\nConsistency\nOver Horizon',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Constraint Consistency')

    plt.tight_layout()
    plt.savefig(output_dir / "eso_gcbf_graph_prediction_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()  # MEMORY OPT: Close figure immediately

    # MEMORY OPT: Force garbage collection after plotting
    gc.collect()


def test_eso_gcbf(args):
    """Main testing function for UNIVERSAL COMPENSATION ESO+GCBF integration with GRAPH PREDICTION (v9-graph-prediction)."""

    print(f"> Running UNIVERSAL COMPENSATION ESO+GCBF Test with GRAPH PREDICTION (v9-graph-prediction)")
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

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    output_dir = pathlib.Path(f"./logs/eso_gcbf_results_graph_prediction/{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # RESTORED: Create enhanced debug logger with comprehensive monitoring (EVERY STEP)
    debug_logger = ESODebugLogger(output_dir, num_agents, args.safety_margin)
    print(f"UNIVERSAL COMPENSATION ESO debug logger with GRAPH PREDICTION initialized - FULL LOGGING RESTORED")

    # Create GCBF interface and ESO-enhanced MPC controller
    gcbf_interface = GCBFInterface(algo)
    mpc_controller = ESOSafetyMPC(
        gcbf_interface=gcbf_interface,
        horizon=args.mpc_horizon,
        dt=env.dt,
        u_max=args.u_max,
        alpha=args.alpha,
        margin=args.safety_margin,
        goal_weight=args.goal_weight,
        debug_logger=debug_logger,
        # ESO parameters (v6 performance optimized)
        eso_dt=0.01,  # Keep v6 performance optimization
        mass=args.mass,
        b0=1.0
    )

    print(f"\nUNIVERSAL COMPENSATION ESO+GCBF Controller with GRAPH PREDICTION Setup (v9-graph-prediction):")
    print(f"  🔮 GRAPH PREDICTION: ENABLED over MPC horizon")
    print(f"     ✅ Agent trajectory prediction: Nominal goal-seeking")
    print(f"     ✅ Obstacle motion: Constant velocity extrapolation")
    print(f"     ✅ Edge features: Recomputed at each horizon step")
    print(f"     ✅ CBF consistency: Maintained across prediction horizon")
    print(f"     ✅ MPC infeasibility: Expected significant reduction")
    print(f"  🆕 UNIVERSAL DISTURBANCE COMPENSATION: {'ENABLED' if ENABLE_DISTURBANCE_COMPENSATION else 'DISABLED'}")
    print(
        f"     ✅ All control actions: {'Compensation applied uniformly' if ENABLE_DISTURBANCE_COMPENSATION else 'No compensation'}")
    print(
        f"     ✅ Simple universal approach: {'No action type differentiation' if ENABLE_DISTURBANCE_COMPENSATION else 'Raw actions used'}")
    print(f"  📊 FULL DEBUG FEATURES (RESTORED):")
    print(f"     ✅ Comprehensive state logging (EVERY STEP)")
    print(f"     ✅ CBF gradient analysis (EVERY STEP)")
    print(f"     ✅ ESO estimation tracking (EVERY STEP)")
    print(f"     ✅ Disturbance monitoring (EVERY STEP)")
    print(f"     ✅ MPC solver diagnostics (EVERY STEP)")
    print(f"     ✅ Safety violation detection (EVERY STEP)")
    print(f"     ✅ All debug prints restored")
    print(f"  ⚙️  CONTROLLER SETTINGS:")
    print(f"     MPC Horizon: {args.mpc_horizon}")
    print(f"     CBF Alpha: {args.alpha}")
    print(f"     Safety Margin: {args.safety_margin:.3f}")
    print(f"     Control Limit: {args.u_max}")
    print(f"     Goal Weight: {args.goal_weight}")
    print(f"     Agent Mass: {args.mass} kg")
    print(f"     ESO Sampling: 100 Hz (performance optimized)")

    # Create tester
    tester = ESOGCBFTester(env, mpc_controller, debug_logger)

    # Generate test keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1000)[:args.epi]
    test_keys = test_keys[args.offset:]

    # Run episodes
    print(f"\nRunning {args.epi} episodes with GRAPH PREDICTION ESO disturbance rejection...")

    # MEMORY OPT: Process episodes one at a time and clean up immediately
    h_traces = []
    all_metrics = []
    all_episode_stats = []
    rollouts = []  # Store for video generation

    for i in range(args.epi):
        print(f"\nEpisode {i + 1}/{args.epi}")

        rollout, h_trace, episode_stats = tester.run_episode(test_keys[i], args.max_step, episode_id=i + 1)
        metrics = tester.evaluate_rollout(rollout)

        # Store essential data
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

        # MEMORY OPT: Force garbage collection after each episode
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

    # RESTORED v3: Print comprehensive results
    print(f"\n" + "=" * 80)
    print(f"UNIVERSAL COMPENSATION ESO+GCBF+GRAPH PREDICTION INTEGRATION RESULTS (v9-graph-prediction)")
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

    # ESO effectiveness
    margin_effectiveness = max(0, total_margin_violations - total_unsafe_violations)
    print(f"\nUNIVERSAL COMPENSATION ESO+GRAPH PREDICTION EFFECTIVENESS:")
    print(f"Early Safety Interventions: {margin_effectiveness} events")
    print(f"ESO Sampling Rate: 100 Hz (performance optimized)")
    print(f"Graph Prediction: ENABLED over {args.mpc_horizon}-step horizon")
    print(
        f"Universal Disturbance Compensation: {'ALL control actions compensated uniformly' if ENABLE_DISTURBANCE_COMPENSATION else 'Compensation DISABLED'}")
    if ENABLE_DISTURBANCE_COMPENSATION:
        print(f"  ✅ All control actions: Compensation applied uniformly")
        print(f"  ✅ Simple universal approach: No action type differentiation")
    else:
        print(f"  ❌ All actions: Raw (no compensation)")

    # Save results
    print(f"\nSaving UNIVERSAL COMPENSATION ESO+GCBF+GRAPH PREDICTION results to: {output_dir}")

    # Save CBF traces and statistics
    np.save(output_dir / "h_traces.npy", h_traces)

    # RESTORED v3: Save comprehensive statistics
    with open(output_dir / "eso_gcbf_graph_prediction_summary.txt", "w") as f:
        f.write(f"Universal Compensation ESO+GCBF+Graph Prediction Integration Test Results (v9-graph-prediction)\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Environment: {env.__class__.__name__}\n")
        f.write(f"Agents: {num_agents}, Obstacles: {args.obs}, Area: {args.area_size}\n")
        f.write(f"Episodes: {args.epi}, Max Steps: {args.max_step}\n\n")

        f.write(f"GRAPH PREDICTION: ENABLED over {args.mpc_horizon}-step horizon\n")
        f.write(f"  Agent trajectory prediction: Nominal goal-seeking\n")
        f.write(f"  Obstacle motion: Constant velocity extrapolation\n")
        f.write(f"  Edge features: Recomputed at each horizon step\n")
        f.write(f"  CBF consistency: Maintained across prediction horizon\n\n")

        f.write(f"UNIVERSAL DISTURBANCE COMPENSATION: {'ENABLED' if ENABLE_DISTURBANCE_COMPENSATION else 'DISABLED'}\n")
        if ENABLE_DISTURBANCE_COMPENSATION:
            f.write(f"  All control actions: Compensation applied uniformly\n")
            f.write(f"  Simple universal approach: No action type differentiation\n")
            f.write(f"  ALL control actions benefit from disturbance rejection\n")
        else:
            f.write(f"  All actions: Raw (no disturbance compensation)\n")
        f.write(f"  Full debug logging: RESTORED (every step)\n")
        f.write(f"  Full debug prints: RESTORED\n\n")

        f.write(f"ESO+MPC Settings:\n")
        f.write(f"  MPC Horizon: {args.mpc_horizon}\n")
        f.write(f"  CBF Alpha: {args.alpha}\n")
        f.write(f"  Safety Margin: {args.safety_margin:.3f}\n")
        f.write(f"  Control Limit: {args.u_max}\n")
        f.write(f"  Goal Weight: {args.goal_weight}\n")
        f.write(f"  ESO Sampling: 100 Hz (performance optimized)\n")
        f.write(f"  Agent Mass: {args.mass} kg\n")
        f.write(f"  ESO Gain b0: 1.0\n\n")

        f.write(f"Overall Results:\n")
        f.write(f"  Safety Rate: {np.mean(safe_rates) * 100:.3f}% ± {np.std(safe_rates) * 100:.3f}%\n")
        f.write(f"  Success Rate: {np.mean(success_rates) * 100:.3f}% ± {np.std(success_rates) * 100:.3f}%\n")
        f.write(f"  Finish Rate: {np.mean(finish_rates) * 100:.3f}% ± {np.std(finish_rates) * 100:.3f}%\n\n")

        f.write(f"CBF+ESO+Graph Prediction Analysis:\n")
        f.write(
            f"  Unsafe Violations: {total_unsafe_violations}/{total_samples} ({total_unsafe_violations / max(1, total_samples) * 100:.3f}%)\n")
        f.write(
            f"  Margin Violations: {total_margin_violations}/{total_samples} ({total_margin_violations / max(1, total_samples) * 100:.3f}%)\n")
        f.write(f"  CBF Range: [{all_h_values.min():.6f}, {all_h_values.max():.6f}]\n")
        f.write(f"  CBF Mean ± Std: {all_h_values.mean():.6f} ± {all_h_values.std():.6f}\n")
        f.write(
            f"  Universal Compensation + Graph Prediction Effectiveness: {margin_effectiveness} early safety interventions {'(if enabled)' if not ENABLE_DISTURBANCE_COMPENSATION else ''}\n")

    # RESTORED v3: Generate comprehensive plots
    print("Generating UNIVERSAL COMPENSATION ESO+GCBF+GRAPH PREDICTION analysis plots...")
    generate_eso_analysis_plots(h_traces, output_dir, args.safety_margin)

    # Generate videos if requested (RESTORED - all episodes)
    if not args.no_video:
        print("Generating episode videos...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        for i, (rollout, metrics) in enumerate(zip(rollouts, all_metrics)):
            video_name = f"eso_gcbf_graph_pred_ep{i:02d}_safe{metrics['safe_rate'] * 100:.0f}_success{metrics['success_rate'] * 100:.0f}"
            video_path = videos_dir / f"{video_name}.mp4"
            env.render_video(rollout, video_path, metrics['is_unsafe'], dpi=args.dpi)
            print(f"  Generated video {i + 1}/{len(rollouts)}: {video_path}")

            # MEMORY OPT: Force garbage collection after each video
            gc.collect()

    print(f"\n🎉 UNIVERSAL COMPENSATION ESO+GCBF+GRAPH PREDICTION Integration Testing Complete! (v9-graph-prediction)")
    print(f"📊 Key Results:")
    print(f"   Safety: {np.mean(safe_rates) * 100:.1f}% | Success: {np.mean(success_rates) * 100:.1f}%")
    print(f"   Graph Prediction: Active over {args.mpc_horizon}-step horizon")
    print(f"   Universal ESO Disturbance Rejection: Active at 100 Hz")
    print(f"   Safety Violations: {total_unsafe_violations / max(1, total_samples) * 100:.1f}%")
    print(f"   Early Interventions: {margin_effectiveness} events")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n🔮 GRAPH PREDICTION BENEFITS:")
    print(f"   ✅ Consistent CBF constraints across MPC horizon")
    print(f"   ✅ Predicted agent trajectories for safety evaluation")
    print(f"   ✅ Obstacle motion extrapolation with constant velocity")
    print(f"   ✅ Dynamic edge feature recomputation at each step")
    print(f"   ✅ Expected significant reduction in MPC infeasibility")
    print(f"   ✅ Smoother control actions with better horizon consistency")
    print(f"\n🆕 UNIVERSAL DISTURBANCE COMPENSATION BENEFITS:")
    if ENABLE_DISTURBANCE_COMPENSATION:
        print(f"   ✅ All control actions: Disturbance rejection applied uniformly")
        print(f"   ✅ Simple universal approach: No action type differentiation")
        print(f"   ✅ Consistent safety performance across ALL control modes")
        print(f"   ✅ Enhanced robustness during system failures")
    else:
        print(f"   ❌ Disturbance compensation: DISABLED")
        print(f"   ❌ All actions use raw control (no disturbance rejection)")
        print(f"   ❌ Reduced robustness to external disturbances")
    print(f"\n🔍 FULL DEBUG CAPABILITIES (RESTORED):")
    print(f"   📋 Agent states: {output_dir}/agent_states.csv (EVERY STEP)")
    print(f"   🎯 CBF analysis: {output_dir}/cbf_data.csv (EVERY STEP)")
    print(f"   🔬 ESO performance: {output_dir}/eso_data.csv (EVERY STEP)")
    print(f"   🌊 Disturbance tracking: {output_dir}/disturbances.csv (EVERY STEP)")
    print(f"   🚧 Obstacle positions: {output_dir}/obstacles.csv (EVERY STEP)")
    print(f"   ⚙️  MPC solver status: {output_dir}/mpc_status.csv (EVERY STEP)")
    print(f"   📈 Analysis plots: {output_dir}/eso_gcbf_graph_prediction_analysis.png")
    print(f"   🎬 Episode videos: {output_dir}/videos/ (ALL EPISODES)")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Compensation ESO+GCBF Integration Testing with Graph Prediction (v9-graph-prediction)")

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
                        help="Safety margin - agents brake when h < margin")
    parser.add_argument("--u-max", type=float, default=1.0,
                        help="Maximum control input magnitude")
    parser.add_argument("--goal-weight", type=float, default=1.0,
                        help="Weight for goal-reaching objective")

    # ESO parameters (performance optimized from v6)
    parser.add_argument("--mass", type=float, default=0.1,
                        help="Agent mass for dynamics (default: 0.1 kg)")

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
    test_eso_gcbf(args)


if __name__ == "__main__":
    main()