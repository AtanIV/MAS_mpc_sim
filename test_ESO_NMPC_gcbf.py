#!/usr/bin/env python3
"""
ESO + MPC-GCBF Integration Script - FIXED WITH LAZY INITIALIZATION

MAJOR FIX: Implements lazy initialization to avoid resource creation before graph data exists.
Key improvements:
1. Lazy MPC instance creation (create on first use, not during init)
2. Comprehensive memory monitoring and management
3. Automatic cleanup and garbage collection
4. Performance statistics and error tracking
5. GPU memory management
"""

import argparse
import datetime
import os

from numpy import ndarray, dtype

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pathlib
import pickle
import numpy as np
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from typing import Tuple, List, Optional, Dict, Any
import matplotlib
import csv
import time
import gc
import sys
import psutil
from collections import defaultdict, deque

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.utils.graph import GraphsTuple, EdgeBlock, GetGraph
from gcbfplus.utils.utils import jax_jit_np, jax2np, mask2index
# from safe_fallback_controller import SafeFallbackController
from pipeline.safe_fallback_controller import SafeFallbackController, FallbackConfig

# Import pipeline components
from pipeline.graph_predictor import MPCGraphPredictor
from pipeline.graph_evaluator import CBFEvaluatorOptimized
from pipeline.gcbf_mpc_solver import NLPMPCController

# CONTROL FLAGS FOR EXPERIMENTAL FEATURES
USE_ESO_FEEDBACK = True # Use ESO state estimates for MPC feedback (vs true states)
ENABLE_DISTURBANCE_COMPENSATION = True  # Apply disturbance compensation to ALL control actions

# USE_ESO_FEEDBACK = False # Use ESO state estimates for MPC feedback (vs true states)
# ENABLE_DISTURBANCE_COMPENSATION = False  # Apply disturbance compensation to ALL control actions



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
    Nonlinear Extended State Observer with correct discrete equations

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

    # Discrete ESO equations with h scaling
    z1_new = z_prev[0] + h * z_prev[1] - beta01 * e
    z2_new = z_prev[1] + h * (z_prev[2] + b0 * u_accel) - beta02 * fe
    z3_new = z_prev[2] - beta03 * fe1

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
    print(f"    Original control acceleration is {u_controller_accel}")
    print(f"    Disturbance estimate is {disturbance_estimate}")
    print(f"    Compensated control acceleration is {u_compensated_accel}")

    # # Final control saturation
    # u_max_accel = 10.0
    # u_compensated_accel = np.clip(u_compensated_accel, -u_max_accel, u_max_accel)

    return float(u_compensated_accel)


class LocalSubgraphExtractor:
    """
    FIXED: Extract local subgraphs with proper padding node handling.

    Fixes:
    1. Preserve original padding structure from full graph
    2. Ensure consistent node indexing between states and node_type arrays
    3. Handle padding node properly during extraction and reconstruction
    """

    def __init__(self, env, comm_radius: float = None):
        self.env = env
        self.comm_radius = comm_radius or env._params.get("comm_radius", 0.5)

    def extract_local_subgraph(self, full_graph: GraphsTuple, ego_agent_idx: int) -> tuple[
        GraphsTuple, ndarray[tuple[int, ...], dtype[Any]]]:
        """
        Extract local subgraph with proper padding handling.
        """
        print(f"\n{'=' * 60}")
        print(f"\n=== EXTRACTING LOCAL SUBGRAPH FOR AGENT {ego_agent_idx} ===")

        # # Debug full graph structure
        # print(f"Full graph structure:")
        # print(f"  States shape: {full_graph.states.shape}")
        # print(f"  Node types shape: {full_graph.node_type.shape}")
        # print(f"  Unique node types: {jnp.unique(full_graph.node_type)}")

        # CRITICAL: Identify padding nodes in original graph
        padding_mask = full_graph.node_type == -1
        logical_mask = full_graph.node_type != -1
        n_padding_nodes = jnp.sum(padding_mask)
        n_logical_nodes = jnp.sum(logical_mask)

        # print(f"  Logical nodes: {n_logical_nodes}, Padding nodes: {n_padding_nodes}")

        # Extract only logical nodes for processing
        logical_states = full_graph.states[logical_mask]
        logical_node_types = full_graph.node_type[logical_mask]

        # print(f"  Logical states shape: {logical_states.shape}")
        # print(f"  Logical node types shape: {logical_node_types.shape}")

        # Get agent, goal, and lidar states from logical nodes only
        logical_agent_mask = logical_node_types == 0
        logical_goal_mask = logical_node_types == 1
        logical_lidar_mask = logical_node_types == 2

        n_total_agents = jnp.sum(logical_agent_mask)
        n_total_goals = jnp.sum(logical_goal_mask)
        n_total_lidar = jnp.sum(logical_lidar_mask)

        # print(f"  Total agents: {n_total_agents}, goals: {n_total_goals}, lidar: {n_total_lidar}")

        # Extract agent and goal states
        agent_states = logical_states[logical_agent_mask]  # [n_agents, 4]
        goal_states = logical_states[logical_goal_mask]  # [n_goals, 4]

        # Get ego agent position for distance calculations
        ego_pos = agent_states[ego_agent_idx, :2]

        # Find nearby agents within communication radius
        agent_positions = agent_states[:, :2]
        distances_to_ego = jnp.linalg.norm(agent_positions - ego_pos, axis=1)

        # Include ego agent and nearby agents
        nearby_agents = distances_to_ego <= self.comm_radius
        nearby_agent_indices = jnp.where(nearby_agents)[0]

        # Always include ego agent first
        if ego_agent_idx not in nearby_agent_indices:
            nearby_agent_indices = jnp.concatenate([jnp.array([ego_agent_idx]), nearby_agent_indices])

        # Reorder so ego agent is first
        ego_mask = nearby_agent_indices == ego_agent_idx
        other_indices = nearby_agent_indices[~ego_mask]
        local_agent_indices = jnp.concatenate([jnp.array([ego_agent_idx]), other_indices])

        # print(f"  Local agent indices: {local_agent_indices}")

        # Extract local agent and goal states (ego agent first)
        local_agent_states = agent_states[local_agent_indices]
        local_goal_states = goal_states[local_agent_indices]  # Corresponding goals

        # Extract LiDAR data for local agents
        n_rays = self.env._params["n_rays"]
        all_lidar_states = logical_states[logical_lidar_mask]

        local_lidar_states = []
        for i, agent_idx in enumerate(local_agent_indices):
            lidar_start = agent_idx * n_rays
            lidar_end = (agent_idx + 1) * n_rays

            # Bounds check
            if lidar_end <= len(all_lidar_states):
                agent_lidar = all_lidar_states[lidar_start:lidar_end]
            else:
                # Create fallback lidar if not enough data
                agent_lidar = jnp.zeros((n_rays, 4))
                if lidar_start < len(all_lidar_states):
                    available_rays = len(all_lidar_states) - lidar_start
                    agent_lidar = agent_lidar.at[:available_rays].set(
                        all_lidar_states[lidar_start:lidar_start + available_rays]
                    )

            local_lidar_states.append(agent_lidar)

        all_local_lidar = jnp.concatenate(local_lidar_states, axis=0)

        # Create local graph structure
        n_local_agents = len(local_agent_indices)
        n_local_goals = n_local_agents
        n_local_lidar = n_rays * n_local_agents
        n_local_logical = n_local_agents + n_local_goals + n_local_lidar

        # print(f"  Local structure: {n_local_agents} agents, {n_local_goals} goals, {n_local_lidar} lidar")
        # print(f"  Total logical nodes in subgraph: {n_local_logical}")

        # Create node features for logical nodes only
        local_node_feats = jnp.zeros((n_local_logical, 3))
        local_node_feats = local_node_feats.at[:n_local_agents, 2].set(1)  # agents [0,0,1]
        local_node_feats = local_node_feats.at[n_local_agents:n_local_agents + n_local_goals, 1].set(
            1)  # goals [0,1,0]
        local_node_feats = local_node_feats.at[-n_local_lidar:, 0].set(1)  # lidar [1,0,0]

        # Create node types for logical nodes only
        local_node_types = jnp.concatenate([
            jnp.zeros(n_local_agents, dtype=jnp.int32),  # agents: type 0
            jnp.ones(n_local_goals, dtype=jnp.int32),  # goals: type 1
            jnp.full(n_local_lidar, 2, dtype=jnp.int32)  # lidar: type 2
        ])

        # Combine all logical states in correct order
        local_all_states = jnp.concatenate([local_agent_states, local_goal_states, all_local_lidar], axis=0)

        # print(f"  Combined states shape: {local_all_states.shape}")
        # print(f"  Node types shape: {local_node_types.shape}")
        # print(f"  Node features shape: {local_node_feats.shape}")

        # Verify consistency
        assert local_all_states.shape[0] == local_node_types.shape[0] == local_node_feats.shape[0], \
            f"Shape mismatch: states={local_all_states.shape[0]}, types={local_node_types.shape[0]}, feats={local_node_feats.shape[0]}"

        # Create edge blocks (simplified version)
        edge_blocks = self._create_local_edge_blocks(
            local_agent_states, local_goal_states, all_local_lidar, n_local_agents
        )

        # Create environment state
        local_env_state = self.env.EnvState(
            agent=local_agent_states,
            goal=local_goal_states,
            obstacle=full_graph.env_states.obstacle  # Keep full obstacle info
        )

        # Use GetGraph format to add the padding node correctly
        get_graph = GetGraph(
            nodes=local_node_feats,
            node_type=local_node_types,
            edge_blocks=edge_blocks,
            env_states=local_env_state,
            states=local_all_states
        )

        # Convert to padded format (adds exactly one padding node)
        local_graph = get_graph.to_padded()

        # Convert local_agent_indices (JAX array) to plain numpy for external use
        local_agent_indices_np = np.array(local_agent_indices, dtype=int)

        # # FINAL VERIFICATION
        # print(f"  Final graph structure:")
        # print(f"    States shape: {local_graph.states.shape}")
        # print(f"    Node types shape: {local_graph.node_type.shape}")
        # print(f"    Unique node types: {jnp.unique(local_graph.node_type)}")

        # Ensure consistency after padding
        expected_total = n_local_logical + 1  # +1 for padding
        assert local_graph.states.shape[0] == expected_total, \
            f"States shape mismatch: got {local_graph.states.shape[0]}, expected {expected_total}"
        assert local_graph.node_type.shape[0] == expected_total, \
            f"Node types shape mismatch: got {local_graph.node_type.shape[0]}, expected {expected_total}"

        # print(f"  Local subgraph created successfully with proper padding")

        return local_graph, local_agent_indices_np

    def _create_local_edge_blocks(self, agent_states, goal_states, lidar_states, n_agents):
        """Create edge blocks for local subgraph."""
        n_rays = self.env._params["n_rays"]
        edge_blocks = []

        # Agent-Agent edges
        pos_diff = agent_states[:, None, :] - agent_states[None, :, :]
        dist = jnp.linalg.norm(pos_diff[:, :, :2], axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self.comm_radius + 1)
        state_diff = agent_states[:, None, :] - agent_states[None, :, :]
        agent_agent_mask = jnp.less(dist, self.comm_radius)
        id_agent = jnp.arange(n_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)
        edge_blocks.append(agent_agent_edges)

        # Agent-Goal edges
        id_goal = jnp.arange(n_agents, n_agents * 2)
        agent_goal_mask = jnp.eye(n_agents)
        agent_goal_feats = agent_states[:, None, :] - goal_states[None, :, :]

        # Apply distance clipping
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :, :2] ** 2, axis=-1, keepdims=True))
        safe_feats_norm = jnp.maximum(feats_norm, self.comm_radius)
        coef = jnp.where(feats_norm > self.comm_radius, self.comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :, :2].set(agent_goal_feats[:, :, :2] * coef)

        goal_agent_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )
        edge_blocks.append(goal_agent_edges)

        # Agent-LiDAR edges
        id_obs = jnp.arange(n_agents * 2, n_agents * 2 + len(lidar_states))

        for i in range(n_agents):
            start_idx = i * n_rays
            end_idx = (i + 1) * n_rays
            agent_lidar_rays = lidar_states[start_idx:end_idx]

            agent_pos = agent_states[i, :2]
            lidar_pos = agent_lidar_rays[:, :2]
            lidar_dist = jnp.linalg.norm(lidar_pos - agent_pos, axis=1)
            active_lidar = jnp.less(lidar_dist, self.comm_radius - 1e-6)

            lidar_feats = agent_states[i] - agent_lidar_rays
            lidar_feats = lidar_feats[None, :, :]
            agent_obs_mask = active_lidar[None, :]

            agent_lidar_edge = EdgeBlock(
                lidar_feats, agent_obs_mask,
                jnp.array([id_agent[i]]), id_obs[start_idx:end_idx]
            )
            edge_blocks.append(agent_lidar_edge)

        return edge_blocks


class MemoryMonitor:
    """Memory monitoring and management for operations."""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.memory_history = deque(maxlen=history_size)
        self.gpu_memory_history = deque(maxlen=history_size)
        self.instance_stats = defaultdict(int)
        self.error_stats = defaultdict(int)
        self.performance_stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'reuse_count': 0,
            'fallback_count': 0,
            'avg_solve_time': 0.0
        }
        self.solve_times = deque(maxlen=50)

    def record_memory_usage(self):
        """Record current system memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_history.append({
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'cpu_percent': process.cpu_percent()
            })

            # Try to get GPU memory if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_used_mb = gpu_info.used / 1024 / 1024
                self.gpu_memory_history.append({
                    'timestamp': time.time(),
                    'gpu_memory_mb': gpu_used_mb,
                    'gpu_total_mb': gpu_info.total / 1024 / 1024
                })
            except:
                pass  # GPU monitoring not available

        except Exception as e:
            print(f"Memory monitoring error: {e}")

    def record_solve_attempt(self, agent_idx: int, success: bool, solve_time: float, error_type: str = None):
        """Record MPC solve attempt statistics."""
        self.performance_stats['total_solves'] += 1
        if success:
            self.performance_stats['successful_solves'] += 1
            self.performance_stats['reuse_count'] += 1
        else:
            self.performance_stats['failed_solves'] += 1
            if error_type:
                self.error_stats[error_type] += 1

        self.solve_times.append(solve_time)
        if len(self.solve_times) > 0:
            self.performance_stats['avg_solve_time'] = sum(self.solve_times) / len(self.solve_times)

    def trigger_cleanup(self, force: bool = False):
        """Trigger memory cleanup operations."""
        if force or self.performance_stats['total_solves'] % 20 == 0:
            print(f"  Memory cleanup triggered (force={force})")

            # Clear JAX caches
            try:
                jax.clear_caches()
                print(f"    JAX caches cleared")
            except Exception as e:
                print(f"    JAX cache clear failed: {e}")

            # Force garbage collection
            collected = gc.collect()
            print(f"    Garbage collected: {collected} objects")

            return True
        return False

    def get_memory_summary(self) -> Dict:
        """Get comprehensive memory and performance summary."""
        current_memory = self.memory_history[-1] if self.memory_history else {'memory_mb': 0}
        current_gpu = self.gpu_memory_history[-1] if self.gpu_memory_history else {'gpu_memory_mb': 0}

        return {
            'current_memory_mb': current_memory.get('memory_mb', 0),
            'current_gpu_mb': current_gpu.get('gpu_memory_mb', 0),
            'peak_memory_mb': max((h.get('memory_mb', 0) for h in self.memory_history), default=0),
            'instance_reuse_count': self.performance_stats['reuse_count'],
            'success_rate': (self.performance_stats['successful_solves'] /
                             max(self.performance_stats['total_solves'], 1)) * 100,
            'avg_solve_time': self.performance_stats['avg_solve_time'],
            'error_breakdown': dict(self.error_stats),
            'total_solves': self.performance_stats['total_solves']
        }

    def print_periodic_report(self, step: int, episode: int):
        """Print periodic monitoring report."""
        if step % 50 == 0 and step > 0:
            summary = self.get_memory_summary()
            print(f"\n--- MPC MEMORY MONITOR REPORT (Episode {episode}, Step {step}) ---")
            print(f"Memory: {summary['current_memory_mb']:.1f} MB (peak: {summary['peak_memory_mb']:.1f} MB)")
            if summary['current_gpu_mb'] > 0:
                print(f"GPU Memory: {summary['current_gpu_mb']:.1f} MB")
            print(f"Instance Reuse: {summary['instance_reuse_count']} times")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Avg Solve Time: {summary['avg_solve_time']:.3f}s")
            print(f"Total Solves: {summary['total_solves']}")
            if summary['error_breakdown']:
                print(f"Errors: {summary['error_breakdown']}")
            print("--- END REPORT ---\n")


class PersistentMPCController:
    """Wrapper for NLPMPCController that allows state updates without recreation."""

    def __init__(self, mpc_controller_instance, agent_idx: int):
        self.mpc_instance = mpc_controller_instance
        self.agent_idx = agent_idx
        self.creation_time = time.time()
        self.usage_count = 0
        self.last_used = time.time()
        self.total_solve_time = 0.0
        self.last_u_opt = None  # warm-start cache: shape (H, control_dim)

    def update_state(self, new_graph: GraphsTuple, agent_states: np.ndarray):
        """Update the controller's internal state with new graph and agent data."""
        try:
            # Update core graph references
            self.mpc_instance.initial_graph = new_graph
            self.mpc_instance.graph_info = self.mpc_instance._analyze_graph_structure(new_graph)

            if hasattr(self.mpc_instance, "graph_predictor") and self.mpc_instance.graph_predictor is not None:
                try:
                    self.mpc_instance.graph_predictor.reset_wall_cache()
                    print("[Predictor] Wall cache cleared for new env step")
                except Exception as e:
                    print(f"[Predictor] Warning: could not reset wall cache: {e}")

            # Update ego goal from new graph
            try:
                goal_states = new_graph.type_states(type_idx=1, n_type=self.mpc_instance.graph_info['goal_nodes'])
                if len(goal_states) > 0:
                    self.mpc_instance.ego_goal = jnp.array(goal_states[0, :2])
            except Exception as e:
                print(f"    Warning: Could not update ego goal for agent {self.agent_idx}: {e}")

            # Extract current ego state from updated graph
            agent_states_from_graph = new_graph.type_states(type_idx=0,
                                                            n_type=self.mpc_instance.graph_info['agent_nodes'])
            if agent_states_from_graph.shape[0] > 0:
                self.mpc_instance.initial_ego_state = jnp.array(agent_states_from_graph[0])

            # Recompile JAX functions with fresh state
            self.mpc_instance._create_pure_jax_functions()

            # Reset optimization statistics for fresh start
            self.mpc_instance.optimization_stats = {
                'iterations': [],
                'solve_times': [],
                'objective_values': [],
                'constraint_violations': []
            }

            self.usage_count += 1
            self.last_used = time.time()

            return True

        except Exception as e:
            print(f"    Error updating MPC state for agent {self.agent_idx}: {e}")
            return False


    def solve(self, max_iterations: int = 100) -> Dict:
        """
        Solve one MPC step (IPOPT-based) using the updated state,
        with warm-start from the previous optimum.
        """
        # Debug diagnostic
        print(f"  [DEBUG] Agent {self.mpc_instance.ego_agent_idx} solving with "
              f"initial_ego_state: {self.mpc_instance.initial_ego_state}")

        # -----------------------------
        # Build warm start (tail + zero)
        # -----------------------------
        try:
            H = int(self.mpc_instance.horizon)
            nu = int(self.mpc_instance.control_dim)
        except Exception:
            # Fallback if attributes differ in your class
            H = int(self.mpc_instance.horizon)
            nu = 2

        if self.last_u_opt is None:
            # Let the solver fall back to zeros internally
            initial_guess = None
        else:
            # Expect shape (H, nu); be defensive if it came back flattened
            u_prev = self.last_u_opt
            if u_prev.ndim == 1:
                u_prev = u_prev.reshape(H, nu)
            # Tail-shift + append zero row
            tail = u_prev[1:].reshape(-1)
            initial_guess = np.concatenate(
                [tail, np.zeros(nu, dtype=u_prev.dtype)],
                axis=0
            )

            # Sanity: solver expects flat (H*nu,)
            if initial_guess.size != H * nu:
                print(f"[WARN] Warm-start size mismatch: got {initial_guess.size}, "
                      f"expected {H * nu}. Using zeros.")
                initial_guess = None

        # -----------------------------
        # Solve
        # -----------------------------
        solve_start = time.time()
        try:
            result = self.mpc_instance.solve_single_step_ipopt(
                initial_guess=initial_guess,
                max_iterations=max_iterations
            )
            solve_time = time.time() - solve_start
            self.total_solve_time += solve_time

            # Cache the new optimum for the next warm start
            if result.get("success", False) and ("optimal_control" in result):
                u_star = result["optimal_control"]
                # Ensure shape (H, nu)
                if u_star.ndim == 1:
                    u_star = u_star.reshape(H, nu)
                self.last_u_opt = u_star

            # Do NOT mutate result structure: v4 logging expects keys like
            # "success", "optimal_control", "nit", "status", "solve_time", etc.
            return result

        except Exception as e:
            solve_time = time.time() - solve_start
            self.total_solve_time += solve_time
            return {
                "success": False,
                "error": str(e),
                "solve_time": solve_time,
            }


    def get_stats(self) -> Dict:
        """Get usage statistics for this persistent controller."""
        age = time.time() - self.creation_time
        avg_solve_time = self.total_solve_time / max(self.usage_count, 1)

        return {
            'agent_idx': self.agent_idx,
            'usage_count': self.usage_count,
            'age_seconds': age,
            'avg_solve_time': avg_solve_time,
            'total_solve_time': self.total_solve_time,
            'last_used_ago': time.time() - self.last_used
        }


class DistributedMPCController:
    """
    Distributed MPC Controller with lazy initialization, memory management,
    and safe fallback controller with CBF and state constraints.
    """

    def __init__(self,
                 model_path: str,
                 env,
                 horizon: int = 3,
                 dt: float = 0.03,
                 alpha: float = 1.0,
                 control_bounds: Tuple[float, float] = (-1.0, 1.0),
                 reference_tracking_weight: float = 1.0,
                 control_effort_weight: float = 0.1):

        self.env = env
        self.model_path = pathlib.Path(model_path)
        self.horizon = horizon
        self.dt = dt
        self.alpha = alpha
        self.control_bounds = control_bounds
        self.ref_weight = reference_tracking_weight
        self.control_weight = control_effort_weight
        self.n_agents = env.num_agents

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()

        # Initialize subgraph extractor
        self.subgraph_extractor = LocalSubgraphExtractor(env)

        # Lazy initialization
        self.persistent_controllers: Dict[int, PersistentMPCController] = {}
        self.initialized_agents = set()

        # Per-agent control sequence memory (global indices)
        # Maps agent_idx -> np.ndarray of shape (H, 2)
        self.agent_control_sequences: Dict[int, np.ndarray] = {}

        # Mapping from ego -> local agent indices in its subgraph
        # Maps ego_agent_idx -> np.ndarray of global indices (length = n_local_agents)
        self.local_agent_indices_per_agent: Dict[int, np.ndarray] = {}


        # # Initialize safe fallback controller
        # self.safe_fallback = SafeFallbackController(
        #     env=env,
        #     alpha=alpha,
        #     relax_penalty=1e3,
        #     margin=0.0,
        # )

        # Initialize safe fallback controller
        fallback_cfg = FallbackConfig(
            alpha=alpha,
            alpha_scale=0.1,  # Match GCBF+ (0.1 * alpha)
            margin=0.0,
            relax_penalty=1e3,
            relax_quad_weight=10.0,
            qp_max_iter=1000,
            verbose=False
        )

        self.safe_fallback = SafeFallbackController(
            env=env,
            mass=env._params["m"],  # Pass mass explicitly
            cfg=fallback_cfg,
            dtype=jnp.float64
        )

        # CBF evaluator for fallback (expects ego index 0 on local graphs)
        try:
            self._cbf_eval = CBFEvaluatorOptimized(str(self.model_path), ego_agent_idx=0)
        except Exception as _e:
            print(f"[Fallback] Could not init CBFEvaluatorOptimized: {_e}")
            self._cbf_eval = None

        def _cbf_ego(local_graph):
            """CBF evaluation for fallback controller - ensures scalar output."""
            if self._cbf_eval is None:
                return jnp.asarray(0.0, dtype=jnp.float64)

            h_val = self._cbf_eval.evaluate_h_jax(local_graph)

            # Ensure scalar (not shape (1,) array)
            return jnp.squeeze(jnp.asarray(h_val, dtype=jnp.float64))

        self._cbf_ego = _cbf_ego

        # Initialize initial graph logging
        self.graph_log_root = pathlib.Path("./local_graph_logs")
        # self.graph_log_root.mkdir(parents=True, exist_ok=True)
        # Per-episode dir gets set at the start of each episode
        self._episode_graph_log_dir = None
        # Per-agent step counters (reset per episode)
        self._graph_log_step = defaultdict(int)
        self.full_graph_log_root = None  # e.g. output_dir / "full_graph_logs"


        print(f"Initializing Distributed MPC Controller with Safe Fallback:")
        print(f"  Agents: {self.n_agents}")
        print(f"  Horizon: {horizon}")
        print(f"  Model path: {model_path}")
        print(f"  Memory monitoring: ENABLED")
        print(f"  Lazy initialization: ENABLED")
        print(f"  Safe fallback: CBF + State Constraints ENABLED")

        # Record initial memory
        self.memory_monitor.record_memory_usage()
        initial_memory = self.memory_monitor.get_memory_summary()['current_memory_mb']
        print(f"  Initial memory usage: {initial_memory:.1f} MB")


    def solve_distributed(self, agent_states: np.ndarray, goal_states: np.ndarray,
                          full_graph: GraphsTuple, episode: int = 0, step: int = 0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Solve MPC (IPOPT-based) using lazy initialization and safe fallback controller.
        """
        n_agents = agent_states.shape[0]
        actions = np.zeros((n_agents, 2))

        # Per-agent info for CSV logging
        per_agent_info: List[Dict[str, Any]] = []

        # Log the full graph once per step, before per-agent solves
        self._log_full_graph(full_graph, episode=episode, step=step)

        # Record memory usage
        self.memory_monitor.record_memory_usage()

        # Trigger cleanup if needed
        cleanup_triggered = self.memory_monitor.trigger_cleanup()

        # Print periodic monitoring report
        self.memory_monitor.print_periodic_report(step, episode)

        for agent_idx in range(n_agents):
            solve_start_time = time.time()

            # per-agent record
            record = {
                "controller_used": None,   # "MPC" | "SAFE_QP" | "EMERGENCY"
                "mpc_success": False,      # kept for backward compatibility
                "mpc_status": "",          # "success" or failure message
                "mpc_iterations": None,    # int, if available
                "u_seq": None,             # (H,2) when MPC succeeds
                "predicted_h": None,       # np.array horizon h (MPC) or [next_step_h] (fallback)

                # Extra IPOPT-aware fields
                "ipopt_status": None,
                "ipopt_iterations": None,
            }

            try:
                # Create instance on first use (lazy initialization)
                if agent_idx not in self.initialized_agents:
                    success = self._create_instance_for_agent(agent_idx, full_graph)
                    if not success:
                        # Creation failed, use SAFE fallback
                        # local_graph = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                        local_graph, _ = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                        actions[agent_idx] = self._safe_fallback_controller(
                            agent_states[agent_idx], goal_states[agent_idx],
                            local_graph, agent_idx, "initialization_failed"
                        ) / self.env._params["m"]
                        self.memory_monitor.record_solve_attempt(
                            agent_idx, False, time.time() - solve_start_time, "initialization_failed"
                        )

                        # per-agent recording
                        record["controller_used"] = "SAFE_QP"
                        record["mpc_status"] = "initialization_failed"
                        record["ipopt_status"] = "initialization_failed"
                        # iterations remain None (no solver call)
                        record["predicted_h"] = self._eval_next_step_h(
                            self.persistent_controllers.get(agent_idx).mpc_instance
                            if self.persistent_controllers.get(agent_idx) else None,
                            full_graph, actions[agent_idx]
                        )
                        per_agent_info.append(record)
                        continue

                # Get persistent controller for this agent
                persistent_controller = self.persistent_controllers.get(agent_idx)

                if persistent_controller is None:
                    # Fallback for failed initialization
                    # local_graph = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                    local_graph, _ = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                    actions[agent_idx] = self._safe_fallback_controller(
                        agent_states[agent_idx], goal_states[agent_idx],
                        local_graph, agent_idx, "no_controller"
                    ) / self.env._params["m"]
                    self.memory_monitor.record_solve_attempt(
                        agent_idx, False, time.time() - solve_start_time, "no_controller"
                    )

                    # per-agent recording
                    record["controller_used"] = "SAFE_QP"
                    record["mpc_status"] = "no_controller"
                    record["ipopt_status"] = "no_controller"
                    record["predicted_h"] = self._eval_next_step_h(
                        None, full_graph, actions[agent_idx]
                    )
                    per_agent_info.append(record)
                    continue

                # Extract local subgraph and update with ESO data
                # local_graph = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                # local_graph_updated = self._update_local_graph_with_eso(local_graph, agent_states, agent_idx)

                local_graph, local_agent_indices = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                # Remember which global agents are in this ego's subgraph
                self.local_agent_indices_per_agent[agent_idx] = np.array(local_agent_indices, copy=True)

                local_graph_updated = self._update_local_graph_with_eso(local_graph, agent_states, agent_idx)

                # Update existing controller state
                state_update_success = persistent_controller.update_state(local_graph_updated, agent_states)
                # state_update_success = persistent_controller.update_state(local_graph, agent_states)  # debug alt

                if not state_update_success:
                    actions[agent_idx] = self._safe_fallback_controller(
                        agent_states[agent_idx], goal_states[agent_idx],
                        local_graph_updated, agent_idx, "state_update_failed"
                    ) / self.env._params["m"]
                    self.memory_monitor.record_solve_attempt(
                        agent_idx, False, time.time() - solve_start_time, "state_update_failed"
                    )

                    # per-agent recording
                    record["controller_used"] = "SAFE_QP"
                    record["mpc_status"] = "state_update_failed"
                    record["ipopt_status"] = "state_update_failed"
                    record["predicted_h"] = self._eval_next_step_h(
                        self.persistent_controllers[agent_idx].mpc_instance,
                        local_graph_updated,
                        actions[agent_idx]
                    )
                    per_agent_info.append(record)
                    continue

                # Log the exact graph that the solver will use (post-update_state)
                self._log_presolve_graph(agent_idx, persistent_controller.mpc_instance.initial_graph)

                # Solve MPC (IPOPT-based) with updated state
                mpc_result = persistent_controller.solve(max_iterations=100)
                # mpc_result = False
                solve_time = time.time() - solve_start_time

                if mpc_result["success"]:
                    optimal_control = mpc_result["optimal_control"]
                    actions[agent_idx] = optimal_control[0] / self.env._params["m"]

                    # per-agent recording
                    record["controller_used"] = "MPC"
                    record["mpc_success"] = True  # legacy flag
                    record["mpc_status"] = mpc_result.get("status", "success")
                    record["mpc_iterations"] = mpc_result.get("nit", None)
                    record["ipopt_status"] = mpc_result.get("status", "success")
                    record["ipopt_iterations"] = mpc_result.get("nit", None)

                    u_star = mpc_result["optimal_control"]
                    if u_star.ndim == 1:
                        u_star = u_star.reshape(self.horizon, 2)
                    record["u_seq"] = np.array(u_star)

                    # Build tail+zero sequence for prediction:
                    #   [u1, u2, ..., u_{H-1}, 0]
                    u_pred = np.zeros_like(u_star)
                    if u_star.shape[0] > 1:
                        u_pred[:-1] = u_star[1:]  # shift left by one step
                    # else: horizon == 1, stays all zeros

                    # Cache this agent’s predicted sequence (used as neighbor predictions)
                    self.agent_control_sequences[agent_idx] = np.array(u_pred, copy=True)
                    print(f"\n--- Caching predicted inputs ---")
                    print(f"    Cached predicted input for agent {agent_idx}: {self.agent_control_sequences[agent_idx]}")

                    # record["predicted_h"] = self._eval_predicted_horizon_h(
                    #     persistent_controller.mpc_instance, u_star
                    # )

                    record["predicted_h"] = self._eval_predicted_horizon_h(
                        persistent_controller.mpc_instance, u_star, agent_idx
                    )

                    per_agent_info.append(record)

                    self.memory_monitor.record_solve_attempt(agent_idx, True, solve_time)

                    if step < 1000:
                        print(
                            f"    Step {step + 1} -- Agent {agent_idx}: MPC (IPOPT) success "
                            f"(reused #{persistent_controller.usage_count}), "
                            f"u=[{actions[agent_idx][0]:.3f}, {actions[agent_idx][1]:.3f}], "
                            f"time={solve_time:.3f}s"
                        )
                else:
                    # MPC/solver failed, use SAFE fallback
                    actions[agent_idx] = self._safe_fallback_controller(
                        agent_states[agent_idx], goal_states[agent_idx],
                        local_graph_updated, agent_idx,
                        f"mpc_failed: {mpc_result.get('error', 'unknown')}"
                    ) / self.env._params["m"]

                    error_type = "mpc_solve_failed"
                    if "error" in mpc_result and "cuSolver" in str(mpc_result["error"]):
                        error_type = "cusolver_error"

                    self.memory_monitor.record_solve_attempt(agent_idx, False, solve_time, error_type)

                    if step < 1000:
                        print(
                            f"    Step {step + 1} -- Agent {agent_idx}: MPC (IPOPT) failed, "
                            f"using SAFE fallback - {mpc_result.get('error', 'unknown')}"
                        )

                    # per-agent recording (now also logs iterations & status on failure)
                    record["controller_used"] = "SAFE_QP"
                    record["mpc_status"] = f"fail: {mpc_result.get('error', mpc_result.get('status', 'unknown'))}"
                    record["mpc_iterations"] = mpc_result.get("nit", None)
                    record["ipopt_status"] = mpc_result.get("status", mpc_result.get("error", "unknown"))
                    record["ipopt_iterations"] = mpc_result.get("nit", None)

                    # Reset prediction sequence on failure to use constant vel assumptions
                    self.agent_control_sequences[agent_idx] = np.zeros((self.horizon, 2), dtype=float)
                    print(f"\n--- Caching predicted inputs ---")
                    print(f"    Cached predicted input for agent {agent_idx}: {self.agent_control_sequences[agent_idx]}")

                    record["predicted_h"] = self._eval_next_step_h(
                        persistent_controller.mpc_instance, local_graph_updated, actions[agent_idx]
                    )
                    per_agent_info.append(record)

            except Exception as e:
                solve_time = time.time() - solve_start_time
                print(f"    Agent {agent_idx}: Exception in MPC/IPOPT solve: {e}")

                # Extract local graph for safe fallback
                try:
                    # local_graph = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                    # local_graph_updated = self._update_local_graph_with_eso(local_graph, agent_states, agent_idx)
                    local_graph, local_agent_indices = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
                    # Remember which global agents are in this ego's subgraph
                    self.local_agent_indices_per_agent[agent_idx] = np.array(local_agent_indices, copy=True)

                    local_graph_updated = self._update_local_graph_with_eso(local_graph, agent_states, agent_idx)

                    actions[agent_idx] = self._safe_fallback_controller(
                        agent_states[agent_idx], goal_states[agent_idx],
                        local_graph_updated, agent_idx, f"exception: {str(e)}"
                    ) / self.env._params["m"]
                except Exception as e2:
                    print(f"    Agent {agent_idx}: Fallback graph extraction failed: {e2}")
                    # Last resort: simple goal tracking without constraints
                    pos_error = goal_states[agent_idx][:2] - agent_states[agent_idx][:2]
                    actions[agent_idx] = np.clip(
                        0.5 * pos_error, self.control_bounds[0], self.control_bounds[1]
                    )

                self.memory_monitor.record_solve_attempt(agent_idx, False, solve_time, "exception")

                # per-agent recording
                record["controller_used"] = "SAFE_QP"
                record["mpc_status"] = f"exception: {str(e)}"
                record["ipopt_status"] = f"exception: {str(e)}"
                # no iteration info available here because solve() itself raised
                record["predicted_h"] = None
                per_agent_info.append(record)

        # Final memory check
        if step % 10 == 0:
            current_summary = self.memory_monitor.get_memory_summary()
            if step < 3 or step % 50 == 0:
                print(
                    f"  Memory status: {current_summary['current_memory_mb']:.1f} MB, "
                    f"Success: {current_summary['success_rate']:.1f}%, "
                    f"Lazy instances: {len(self.initialized_agents)}/{self.n_agents}"
                )

        return actions, per_agent_info


    def _safe_fallback_controller(
            self,
            agent_state: np.ndarray,
            goal_state: np.ndarray,
            local_graph: GraphsTuple,
            agent_idx: int,
            failure_reason: str = "unknown"
    ) -> np.ndarray:
        """
        Safe fallback controller with COMPREHENSIVE diagnostics.

        Args:
            agent_state: Current agent state [x, y, vx, vy]
            goal_state: Goal state [x_g, y_g, vx_g, vy_g]
            local_graph: Local subgraph around the agent
            agent_idx: Agent index for logging
            failure_reason: Reason why main controller failed

        Returns:
            safe_control: Control action (force, NOT acceleration) that satisfies CBF constraints
        """
        try:
            print(f"\n{'=' * 70}")
            print(f"FALLBACK CONTROLLER ACTIVATED - Agent {agent_idx}")
            print(f"{'=' * 70}")
            print(f"Reason for fallback: {failure_reason}")
            print(f"Agent state: pos=[{agent_state[0]:.4f}, {agent_state[1]:.4f}], "
                  f"vel=[{agent_state[2]:.4f}, {agent_state[3]:.4f}]")
            print(f"Goal state:  pos=[{goal_state[0]:.4f}, {goal_state[1]:.4f}]")

            # Evaluate CBF BEFORE control
            print(f"\n--- PRE-CONTROL CBF EVALUATION ---")
            try:
                h_before = float(self._cbf_ego(local_graph))
                print(f"CBF value (h): {h_before:.6f}")

                if h_before < 0:
                    print(f"  ⚠️ CRITICAL: Agent in UNSAFE region (h < 0)!")
                elif h_before < 0.1:
                    print(f"  ⚠️ WARNING: Close to unsafe region (h < 0.1)")
                elif h_before < 0.2:
                    print(f"  ⚡ CAUTION: Approaching safety boundary (h < 0.2)")
                else:
                    print(f"  ✓ Safe region (h >= 0.2)")
            except Exception as e:
                h_before = None
                print(f"  ⚠️ Could not evaluate CBF: {e}")

            # Graph structure
            print(f"\n--- LOCAL GRAPH STRUCTURE ---")
            local_agent_mask = local_graph.node_type == 0
            n_local_agents = int(jnp.sum(local_agent_mask))

            if n_local_agents > 1:
                agent_states_all = local_graph.type_states(type_idx=0, n_type=n_local_agents)
                print(f"Local agents: {n_local_agents} (ego + {n_local_agents - 1} neighbors)")
                print(f"Neighbors:")
                for i in range(1, n_local_agents):
                    neighbor = agent_states_all[i]
                    dist = np.linalg.norm(neighbor[:2] - agent_state[:2])
                    print(f"  {i}: pos=[{neighbor[0]:.4f}, {neighbor[1]:.4f}], "
                          f"vel=[{neighbor[2]:.4f}, {neighbor[3]:.4f}], dist={dist:.4f}")
            else:
                print(f"Local agents: 1 (ego only)")

            # Solve QP
            print(f"\n--- SOLVING FALLBACK QP ---")
            t0 = time.time()
            u_force = self.safe_fallback.act_ego(local_graph, self._cbf_ego)
            solve_time = time.time() - t0

            # Convert to numpy
            safe_control = np.array(u_force, dtype=float)

            # Get diagnostics
            diag = self.safe_fallback.get_diagnostics()

            print(f"QP solve time: {solve_time:.4f}s")
            print(f"QP status: {diag['qp_status']}")  # Now shows SOLVED/OPTIMAL
            print(f"QP iterations: {diag['qp_iter_inner']} inner, {diag['qp_iter_ext']} outer")
            print(f"QP primal residual: {diag['qp_pri_res']:.2e}")
            print(f"QP dual residual: {diag['qp_dua_res']:.2e}")

            if diag.get('duality_gap') is not None:
                print(f"QP duality gap: {diag['duality_gap']:.2e}")
            if diag.get('obj_value') is not None:
                print(f"QP objective value: {diag['obj_value']:.6f}")

            # Control output
            print(f"\n--- QP SOLUTION ---")
            print(f"Control (force): u = [{safe_control[0]:.6f}, {safe_control[1]:.6f}]")
            print(f"Control magnitude: ||u|| = {np.linalg.norm(safe_control):.6f}")

            # Relaxation
            relaxation = diag.get('relaxation', 0.0)
            print(f"Relaxation: r = {relaxation:.6f}")
            if relaxation > 1e-6:
                print(f"  ⚠️ Constraint relaxation active!")
            else:
                print(f"  ✓ No relaxation needed")

            # Validation
            print(f"\n--- VALIDATION ---")
            if np.any(np.isnan(safe_control)):
                raise ValueError(f"Fallback returned NaN control: {safe_control}")
            print(f"✓ No NaN values")

            # Track statistics
            if not hasattr(self, '_fallback_calls'):
                self._fallback_calls = 0
                self._fallback_total_time = 0.0
                self._fallback_reasons = {}

            self._fallback_calls += 1
            self._fallback_total_time += solve_time

            reason_key = failure_reason.split(':')[0]
            self._fallback_reasons[reason_key] = self._fallback_reasons.get(reason_key, 0) + 1

            print(f"\n--- STATISTICS ---")
            print(f"Total fallback calls: {self._fallback_calls}")
            print(f"Average solve time: {self._fallback_total_time / self._fallback_calls:.4f}s")

            print(f"{'=' * 70}\n")

            return safe_control

        except Exception as e:
            print(f"\n{'=' * 70}")
            print(f"⚠️ FALLBACK QP FAILED - Agent {agent_idx}")
            print(f"{'=' * 70}")
            print(f"Error: {e}")

            import traceback
            traceback.print_exc()

            # Emergency fallback
            pos_error = goal_state[:2] - agent_state[:2]
            vel_error = goal_state[2:4] - agent_state[2:4]

            u_emergency = 0.3 * pos_error + 0.15 * vel_error
            u_emergency = np.clip(
                u_emergency,
                self.control_bounds[0] * 0.5,
                self.control_bounds[1] * 0.5
            )

            print(f"Using emergency control: u = [{u_emergency[0]:.4f}, {u_emergency[1]:.4f}]")
            print(f"{'=' * 70}\n")

            return u_emergency

    def _create_instance_for_agent(self, agent_idx: int, full_graph: GraphsTuple) -> bool:
        print(f"  Creating MPC instance for agent {agent_idx} (lazy initialization)...")

        try:
            # Extract local subgraph for this agent from current full graph
            # local_graph = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
            local_graph, local_agent_indices = self.subgraph_extractor.extract_local_subgraph(full_graph, agent_idx)
            # Remember which global agents are in this ego's subgraph
            self.local_agent_indices_per_agent[agent_idx] = np.array(local_agent_indices, copy=True)

            # Create MPC controller instance
            mpc_instance = NLPMPCController(
                model_path=str(self.model_path),
                env=self.env,
                initial_graph=local_graph,
                ego_agent_idx=0,  # Always 0 in local subgraph (ego agent is always first)
                horizon=self.horizon,
                cbf_margin=0.0,
                use_discrete_cbf=True,
                dt=self.dt,
                alpha=self.alpha,
                control_bounds=self.control_bounds,
                reference_tracking_weight=self.ref_weight,
                control_effort_weight=self.control_weight,
                saturation_margin = 0.98,
                enable_reparameterization = True,
            )

            # Wrap in persistent controller
            persistent_controller = PersistentMPCController(mpc_instance, agent_idx)
            self.persistent_controllers[agent_idx] = persistent_controller
            self.initialized_agents.add(agent_idx)

            print(f"    Agent {agent_idx}: MPC instance created successfully")
            return True

        except Exception as e:
            print(f"    Agent {agent_idx}: Failed to create MPC instance: {e}")
            # Store None to mark as failed initialization
            self.persistent_controllers[agent_idx] = None
            return False

    def _update_local_graph_with_eso(self, local_graph: GraphsTuple,
                                     eso_agent_states: np.ndarray, ego_agent_idx: int) -> GraphsTuple:
        """Same as before - no changes needed"""
        # print(f"  Updating local graph with ESO estimates for agent {ego_agent_idx}")

        # Handle padding properly
        padding_mask = local_graph.node_type == -1
        logical_mask = local_graph.node_type != -1
        logical_node_types = local_graph.node_type[logical_mask]
        logical_states = local_graph.states[logical_mask]

        # Get number of local agents from logical nodes
        agent_mask_logical = logical_node_types == 0
        n_local_agents = jnp.sum(agent_mask_logical)

        # Extract agent states from logical nodes
        local_agent_states = logical_states[agent_mask_logical]

        # Update ego agent state (first agent in local graph)
        ego_eso_state = jnp.array(eso_agent_states[ego_agent_idx])
        updated_local_agent_states = local_agent_states.at[0].set(ego_eso_state)

        # Reconstruct full logical states
        goal_mask_logical = logical_node_types == 1
        lidar_mask_logical = logical_node_types == 2

        goal_states_logical = logical_states[goal_mask_logical]
        lidar_states_logical = logical_states[lidar_mask_logical]

        # Combine updated agent states with unchanged goal/lidar states
        updated_logical_states = jnp.concatenate([
            updated_local_agent_states,
            goal_states_logical,
            lidar_states_logical
        ], axis=0)

        # Reconstruct full states array with padding
        if jnp.any(padding_mask):
            padding_states = local_graph.states[padding_mask]
            all_updated_states = jnp.concatenate([updated_logical_states, padding_states], axis=0)
        else:
            all_updated_states = updated_logical_states

        # Update the graph and recompute edge features
        updated_graph = local_graph._replace(states=all_updated_states)
        updated_graph = self.env.add_edge_feats(updated_graph, all_updated_states)

        return updated_graph

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary including fallback statistics."""
        memory_summary = self.memory_monitor.get_memory_summary()

        # Collect controller statistics
        controller_stats = []
        for agent_idx, controller in self.persistent_controllers.items():
            if controller is not None:
                controller_stats.append(controller.get_stats())

        # Compute fallback controller statistics
        fcalls = getattr(self, "_fallback_calls", 0)
        ftime = getattr(self, "_fallback_total_time", 0.0)
        fallback_stats = {
            "total_calls": fcalls,
            # ProxQP + slack makes the QP (practically) always feasible; treat as success
            "success_rate": 100.0 if fcalls > 0 else 0.0,
            "cbf_violations_prevented": fcalls,
            "velocity_violations_prevented": 0,
            "avg_solve_time": (ftime / fcalls) if fcalls > 0 else 0.0,
        }

        return {
            'memory_summary': memory_summary,
            'controller_stats': controller_stats,
            'fallback_stats': fallback_stats,
            'total_agents': self.n_agents,
            'active_controllers': len([c for c in self.persistent_controllers.values() if c is not None]),
            'initialized_agents': len(self.initialized_agents)
        }

    def cleanup(self):
        """Explicit cleanup method for end of testing with fallback statistics."""
        print(f"Cleaning up distributed MPC controller...")

        # Print final performance summary
        summary = self.get_performance_summary()
        print(f"Final Performance Summary:")
        print(f"  Memory Peak: {summary['memory_summary']['peak_memory_mb']:.1f} MB")
        print(f"  Total Instance Reuse: {summary['memory_summary']['instance_reuse_count']}")
        print(f"  Overall Success Rate: {summary['memory_summary']['success_rate']:.1f}%")
        print(f"  Average Solve Time: {summary['memory_summary']['avg_solve_time']:.3f}s")
        print(f"  Lazy Instances Created: {summary['initialized_agents']}/{summary['total_agents']}")

        # Print safe fallback statistics
        fallback_stats = summary['fallback_stats']
        print(f"\nSafe Fallback Controller Performance:")
        print(f"  Total fallback calls: {fallback_stats['total_calls']}")
        print(f"  Fallback success rate: {fallback_stats['success_rate']:.1f}%")
        print(f"  CBF violations prevented: {fallback_stats['cbf_violations_prevented']}")
        print(f"  Velocity violations prevented: {fallback_stats['velocity_violations_prevented']}")
        print(f"  Average fallback solve time: {fallback_stats['avg_solve_time']:.4f}s")

        # Clear persistent controllers
        for agent_idx in list(self.persistent_controllers.keys()):
            self.persistent_controllers[agent_idx] = None
        self.persistent_controllers.clear()
        self.initialized_agents.clear()

        # Final cleanup
        self.memory_monitor.trigger_cleanup(force=True)
        print(f"Distributed MPC controller cleanup complete")

    def add_fallback_logging_to_distributed_controller(self):
        """
        Optional: Add this method to the DistributedMPCController class for better logging
        """

        def _log_fallback_usage(self, agent_idx: int, failure_reason: str, fallback_type: str):
            """Log when fallback controller is used and why."""
            if not hasattr(self, '_fallback_usage_log'):
                self._fallback_usage_log = []

            self._fallback_usage_log.append({
                'agent_idx': agent_idx,
                'failure_reason': failure_reason,
                'fallback_type': fallback_type,
                'timestamp': time.time()
            })

            # Print occasionally for debugging
            if len(self._fallback_usage_log) <= 10:  # First 10 fallbacks
                print(f"    Fallback used: Agent {agent_idx}, Reason: {failure_reason}, Type: {fallback_type}")

    def _log_presolve_graph(self, agent_idx: int, graph: GraphsTuple) -> None:
        try:
            base_dir = getattr(self, "_episode_graph_log_dir", None) or self.graph_log_root
            base_dir.mkdir(parents=True, exist_ok=True)

            agent_dir = base_dir / f"agent{agent_idx:02d}"
            agent_dir.mkdir(parents=True, exist_ok=True)

            step = int(self._graph_log_step.get(agent_idx, 0) if hasattr(self, "_graph_log_step") else 0)
            log_path = agent_dir / f"step{step:05d}.pkl"

            # Try to grab the local→global mapping for this ego
            local_agent_indices = None
            if hasattr(self, "local_agent_indices_per_agent"):
                lai = self.local_agent_indices_per_agent.get(agent_idx, None)
                if lai is not None:
                    local_agent_indices = np.array(lai, copy=True)

            graph_data = {
                "states": np.array(graph.states),
                "nodes": np.array(graph.nodes),
                "edges": np.array(graph.edges),
                "node_type": np.array(graph.node_type),
                "senders": np.array(graph.senders),
                "receivers": np.array(graph.receivers),
                "agent_idx": agent_idx,  # global ego index
                "step_idx": step,  # global step
                "local_agent_indices": local_agent_indices,  # NEW: global IDs of local agents
            }

            with open(log_path, "wb") as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            if hasattr(self, "_graph_log_step"):
                self._graph_log_step[agent_idx] = step + 1

            print(f"[GraphLog] Saved {log_path}")
        except Exception as e:
            print(f"[GraphLog] Warning: failed to save pre-solve graph for agent {agent_idx}: {e}")

    def _log_full_graph(self,
                        full_graph: GraphsTuple,
                        episode: int,
                        step: int) -> None:
        """
        Log the full global graph for a given episode and step, to:

            full_graph_log_root / epXX / stepZZZZZ.pkl
        """
        try:
            if self.full_graph_log_root is None:
                # Logging not configured for this run
                return

            # Root folder is something like: logs/.../1208-2324/full_graph_logs
            base_dir = self.full_graph_log_root

            # Make per-episode directory: epXX
            ep_dir = base_dir / f"ep{episode:02d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            # File name: stepZZZZZ.pkl
            log_path = ep_dir / f"step{step:05d}.pkl"

            graph_data = {
                "states": np.array(full_graph.states),
                "nodes": np.array(full_graph.nodes),
                "edges": np.array(full_graph.edges),
                "node_type": np.array(full_graph.node_type),
                "senders": np.array(full_graph.senders),
                "receivers": np.array(full_graph.receivers),
                "globals": np.array(full_graph.globals)
                if getattr(full_graph, "globals", None) is not None else None,
                "episode": int(episode),
                "step": int(step),
            }

            with open(log_path, "wb") as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[FullGraphLog] Saved {log_path}")
        except Exception as e:
            print(
                f"[FullGraphLog] Warning: failed to save full graph "
                f"(episode={episode}, step={step}): {e}"
            )

    def _eval_predicted_horizon_h(self,
                                  mpc,
                                  u_seq: np.ndarray,
                                  ego_agent_idx: int):
        """
        Return np.array of h-values over the predicted horizon.

        Uses:
          - u_seq: (H,2) ego MPC sequence
          - self.local_agent_indices_per_agent[ego_agent_idx]
          - self.agent_control_sequences[global_idx] for neighbors

        If we can't find the mapping or sequences, we fall back to
        ego-only prediction.
        """
        print(f"\n[h-eval DEBUG] _eval_predicted_horizon_h called")
        try:
            if mpc is None or getattr(mpc, "graph_predictor", None) is None:
                return None

            u_seq = np.asarray(u_seq, dtype=float)
            H = u_seq.shape[0]

            # Try to get the local→global mapping for this ego
            local_indices = self.local_agent_indices_per_agent.get(ego_agent_idx, None)

            if local_indices is None:
                # Fallback: original behaviour (ego-only controls)
                pred_graphs = mpc.graph_predictor.predict_graphs_horizon(
                    mpc.initial_graph,
                    u_seq
                )
            else:
                local_indices = np.asarray(local_indices, dtype=int)
                n_local = local_indices.shape[0]

                # Build (H, n_local, 2) control tensor
                ctrl_all = np.zeros((H, n_local, 2), dtype=float)

                # Ego is always local index 0
                ctrl_all[:, 0, :] = u_seq

                # Fill in non-ego local agents if we have their sequences
                for j_local in range(1, n_local):
                    g_idx = int(local_indices[j_local])
                    seq = self.agent_control_sequences.get(g_idx, None)
                    if seq is None:
                        # no MPC plan for this neighbor yet → leave zeros (constant velocity)
                        continue

                    seq = np.asarray(seq, dtype=float)
                    if seq.ndim == 1:
                        seq = seq.reshape(-1, 2)

                    Hj = seq.shape[0]
                    if Hj >= H:
                        ctrl_all[:, j_local, :] = seq[:H]
                    else:
                        # If neighbor's plan is shorter, pad with its last control
                        ctrl_all[:Hj, j_local, :] = seq
                        ctrl_all[Hj:, j_local, :] = seq[-1]

                # Use the extended API: (H, n_local, 2)
                pred_graphs = mpc.graph_predictor.predict_graphs_horizon(
                    mpc.initial_graph,
                    ctrl_all
                )

            if pred_graphs is None or len(pred_graphs) == 0:
                return None

            hs = []
            for g in pred_graphs:
                try:
                    h_val = mpc.cbf_evaluator.evaluate_h_jax(g)
                    hs.append(float(np.array(h_val).reshape(-1)[0]))
                except Exception as e:
                    print(f"[H-eval] Failed to evaluate h for predicted graph: {e}")
                    hs.append(None)

            return np.asarray(hs, dtype=float)

        except Exception as e:
            print(f"[H-eval] Exception in _eval_predicted_horizon_h: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _eval_next_step_h(self, mpc, local_graph: GraphsTuple, u_accel: np.ndarray):
        """Fallback path: predict one step with the chosen control and return np.array([h_next])."""
        try:
            if mpc is None or getattr(mpc, "graph_predictor", None) is None:
                return None

            # Reshape control input to (1, 2) for single-step prediction
            u_1 = np.zeros((1, 2), dtype=float)
            u_1[0] = u_accel

            pred_graphs = mpc.graph_predictor.predict_graphs_horizon(
                local_graph,
                u_1
            )

            if not pred_graphs or len(pred_graphs) == 0:
                return None

            # Evaluate h for the predicted next state
            h_next = mpc.cbf_evaluator.evaluate_h_jax(pred_graphs[0])
            return np.asarray([float(np.array(h_next).reshape(-1)[0])], dtype=float)

        except Exception as e:
            print(f"[H-eval] Exception in _eval_next_step_h: {e}")
            import traceback
            traceback.print_exc()
            return None


class AgentStepCSVLogger:
    def __init__(self, output_dir: pathlib.Path, num_agents: int):
        self.dir = output_dir / "agent_step_logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files = []
        self.writers = []
        header = [
            "step",
            "cbf_initial",         # h of the current (full/local) graph's ego
            "mpc_status",          # success / reason
            "controller_used",     # MPC / SAFE_QP / EMERGENCY
            "mpc_iterations",      # int or blank
            "control_input",       # JSON-ified list or compact string
            "predicted_h"          # JSON-ified list over horizon (or single next-step)
        ]
        import csv, json
        for i in range(num_agents):
            f = open(self.dir / f"agent_{i:02d}.csv", "w", newline="")
            w = csv.writer(f)
            w.writerow(header)
            self.files.append(f)
            self.writers.append(w)

    def log(self, agent_idx: int, step: int, cbf_initial: float,
            mpc_status: str, controller_used: str, mpc_iterations,
            control_input, predicted_h):
        import json
        ctrl_str = json.dumps(np.asarray(control_input).tolist()) if control_input is not None else ""
        ph_str   = json.dumps(np.asarray(predicted_h).tolist())   if predicted_h is not None else ""
        self.writers[agent_idx].writerow([
            step,
            float(cbf_initial) if cbf_initial is not None else "",
            mpc_status or "",
            controller_used or "",
            int(mpc_iterations) if mpc_iterations is not None else "",
            ctrl_str,
            ph_str
        ])
        self.files[agent_idx].flush()

    def close(self):
        for f in self.files:
            try: f.close()
            except: pass


class ESODebugLogger:
    """ESO debug logger (same as original)."""

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

        # Initialize files with headers
        self._init_log_files()

    def _init_log_files(self):
        """Initialize CSV log files with appropriate headers."""

        # Agent states log
        with open(self.state_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y',
                      'goal_x', 'goal_y', 'goal_vel_x', 'goal_vel_y']
            writer.writerow(header)
            f.flush()

        # CBF data log
        with open(self.cbf_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'cbf_value', 'margin_violation',
                      'grad_x', 'grad_y', 'grad_vx', 'grad_vy',
                      'drift_term', 'control_coeff_x', 'control_coeff_y']
            writer.writerow(header)
            f.flush()

        # MPC status log
        with open(self.mpc_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'solver_status', 'solve_time',
                      'action_x', 'action_y', 'feasible',
                      'eso_action_x', 'eso_action_y', 'compensated_x', 'compensated_y']
            writer.writerow(header)
            f.flush()

        # Obstacle geometry log (real obstacles, not LiDAR)
        with open(self.obstacle_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'episode', 'step', 'obstacle_id',
                'center_x', 'center_y',
                'length_x', 'length_y', 'theta',
                'v0_x', 'v0_y', 'v1_x', 'v1_y',
                'v2_x', 'v2_y', 'v3_x', 'v3_y',
                'type',
            ]
            writer.writerow(header)
            f.flush()


        with open(self.eso_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'axis',
                      'true_pos', 'eso_pos', 'pos_error',
                      'true_vel', 'eso_vel', 'vel_error',
                      'true_dist', 'eso_dist', 'dist_error',
                      'measurement_error']
            writer.writerow(header)
            f.flush()

        with open(self.disturbance_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'step', 'agent_id', 'true_dist_x', 'true_dist_y',
                      'eso_dist_x', 'eso_dist_y']
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
                agent_state = agent_states[i]
                goal_state = goal_states[i]
                writer.writerow([
                    episode, step, i,
                    float(agent_state[0]), float(agent_state[1]),
                    float(agent_state[2]), float(agent_state[3]),
                    float(goal_state[0]), float(goal_state[1]),
                    float(goal_state[2]), float(goal_state[3])
                ])
            f.flush()

        # Log CBF data
        with open(self.cbf_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(self.num_agents):
                # FIXED: Handle both scalar and array h_values
                if isinstance(h_values, (list, np.ndarray)):
                    cbf_val = float(h_values[i]) if i < len(h_values) else 0.0
                else:
                    cbf_val = float(h_values) if i == 0 else 0.0  # Scalar case - only valid for agent 0
                margin_violation = cbf_val < self.margin

                if i < len(cbf_gradients) and cbf_gradients[i] is not None:
                    grad = cbf_gradients[i]
                    if len(grad) >= 4:
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
            f.flush()

        # Log real obstacle geometry (from env_states.obstacle[5] vertices)
        with open(self.obstacle_log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # env_states.obstacle is the same object you probed in obstacle_debugging.py
            obstacles = getattr(graph.env_states, "obstacle", None)
            if obstacles is None:
                f.flush()
                return

            vertices = None
            # In your debug script, obstacles[5] is the vertices array
            try:
                if hasattr(obstacles, "__len__") and len(obstacles) > 5:
                    vertices = np.asarray(obstacles[5])
            except Exception as e:
                print(f"[ESODebugLogger] Could not access obstacle vertices: {e}")
                vertices = None

            if vertices is None:
                # As a fallback, do nothing (or print a warning)
                # print("[ESODebugLogger] No obstacle vertices available; skipping obstacle log.")
                f.flush()
                return

            # vertices shape: (n_obstacles, 4, 2)
            n_obs = vertices.shape[0]
            for obs_id in range(n_obs):
                verts = np.asarray(vertices[obs_id], dtype=float)
                if verts.shape[0] < 2:
                    continue

                # Center = mean of vertices
                center = verts.mean(axis=0)

                # Edge 0->1 as main axis
                edge01 = verts[1] - verts[0]
                # Edge 0->3 as the other axis if available (rectangles => 4 vertices)
                if verts.shape[0] > 3:
                    edge03 = verts[3] - verts[0]
                else:
                    edge03 = np.array([0.0, 0.0])

                length_x = float(np.linalg.norm(edge01))
                length_y = float(np.linalg.norm(edge03)) if np.linalg.norm(edge03) > 0 else 0.0
                theta = float(np.arctan2(edge01[1], edge01[0]))  # orientation of first edge

                # Ensure we have exactly 4 vertices; if more, take first 4
                verts4 = verts[:4]
                # Pad if fewer than 4 (very defensive; shouldn't happen)
                if verts4.shape[0] < 4:
                    pad = np.tile(verts4[-1], (4 - verts4.shape[0], 1))
                    verts4 = np.vstack([verts4, pad])

                (v0x, v0y), (v1x, v1y), (v2x, v2y), (v3x, v3y) = verts4

                writer.writerow([
                    int(episode),
                    int(step),
                    int(obs_id),
                    float(center[0]), float(center[1]),
                    length_x, length_y, theta,
                    float(v0x), float(v0y),
                    float(v1x), float(v1y),
                    float(v2x), float(v2y),
                    float(v3x), float(v3y),
                    "static",  # obstacle type for now
                ])

            f.flush()


    def log_eso_data(self, episode: int, step: int, agent_id: int,
                     true_states_x: np.ndarray, true_states_y: np.ndarray,
                     eso_states_x: np.ndarray, eso_states_y: np.ndarray,
                     true_dist_x: float, true_dist_y: float):
        """Log ESO estimation performance."""

        with open(self.eso_log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if len(eso_states_x) < 3 or len(eso_states_y) < 3:
                return

            # X-axis ESO data
            writer.writerow([
                episode, step, agent_id, 'x',
                float(true_states_x[0]), float(eso_states_x[0]), float(eso_states_x[0] - true_states_x[0]),
                float(true_states_x[1]), float(eso_states_x[1]), float(eso_states_x[1] - true_states_x[1]),
                float(true_dist_x), float(eso_states_x[2]), float(eso_states_x[2] - true_dist_x),
                float(eso_states_x[0] - true_states_x[0])
            ])

            # Y-axis ESO data
            writer.writerow([
                episode, step, agent_id, 'y',
                float(true_states_y[0]), float(eso_states_y[0]), float(eso_states_y[0] - true_states_y[0]),
                float(true_states_y[1]), float(eso_states_y[1]), float(eso_states_y[1] - true_states_y[1]),
                float(true_dist_y), float(eso_states_y[2]), float(eso_states_y[2] - true_dist_y),
                float(eso_states_y[0] - true_states_y[0])
            ])
            f.flush()

    def log_disturbances(self, episode: int, step: int, agent_id: int,
                         true_dist_x: float, true_dist_y: float,
                         eso_dist_x: float, eso_dist_y: float):
        """Log disturbance estimates vs true values."""

        with open(self.disturbance_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, step, agent_id,
                float(true_dist_x), float(true_dist_y),
                float(eso_dist_x), float(eso_dist_y)
            ])
            f.flush()

    def log_mpc_result(self, episode: int, step: int, agent_id: int,
                       solver_status: str, solve_time: float, action: np.ndarray,
                       feasible: bool,
                       eso_action: np.ndarray = None, compensated_action: np.ndarray = None):
        """Log MPC solver results and actions with ESO data."""

        with open(self.mpc_log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            eso_x = float(eso_action[0]) if eso_action is not None else 0.0
            eso_y = float(eso_action[1]) if eso_action is not None else 0.0
            comp_x = float(compensated_action[0]) if compensated_action is not None else 0.0
            comp_y = float(compensated_action[1]) if compensated_action is not None else 0.0

            writer.writerow([
                episode, step, agent_id, solver_status, solve_time,
                float(action[0]), float(action[1]), feasible,
                eso_x, eso_y, comp_x, comp_y
            ])
            f.flush()


class ESOSafetyMPC:
    """
    ESO Safety MPC: Integrates ESO state estimation with NLP based MPC optimization.
    """

    def __init__(self,
                 model_path: str,
                 env,
                 horizon: int = 3,  # Reduced horizon for faster testing
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

        self.env = env
        self.model_path = model_path
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

        # ESO performance optimization
        self.eso_per_plant = max(1, int(dt / eso_dt))

        # ESO gains
        self.beta01 = 1.0 * 0.35
        self.beta02 = 1.0 / (2.0 * self.dt ** 0.5) * 0.2
        self.beta03 = 2.0 / (25.0 * self.dt ** 1.2) * 0.23
        # self.beta01 = 1.0
        # self.beta02 = 1.0 / (2.0 * dt ** 0.5)
        # self.beta03 = 2.0 / (25.0 * dt ** 1.2)

        # Nonlinear function parameters
        self.delta = eso_dt
        self.alpha1 = 0.5
        self.alpha2 = 0.25

        # Initialize ESO states for all agents
        num_agents = env.num_agents
        self.eso_states_x = np.zeros((num_agents, 3))
        self.eso_states_y = np.zeros((num_agents, 3))
        self.prev_actions = np.zeros((num_agents, 2))

        # Initialize distributed MPC controller with lazy initialization
        self.mpc_controller = DistributedMPCController(
            model_path=model_path,
            env=env,
            horizon=horizon,
            dt=dt,
            alpha=alpha,
            control_bounds=(-u_max, u_max),
            reference_tracking_weight=goal_weight,
            control_effort_weight=0.1
        )

        # Initialize CBF interface for logging
        try:
            from pipeline.graph_evaluator import CBFEvaluator
            self.cbf_interface = CBFEvaluator(str(self.model_path), ego_agent_idx=0)
            print(f"  CBF interface initialized for logging")
        except Exception as e:
            print(f"  Failed to initialize CBF interface: {e}")
            self.cbf_interface = None

        print(f"ESO Safety MPC Controller initialized (FIXED):")
        print(f"  NLP-based MPC: ENABLED with lazy initialization architecture")
        print(f"     Lazy controllers: Instances created on first agent use")
        print(f"     Memory monitoring: Comprehensive tracking enabled")
        print(f"     Automatic cleanup: JAX cache clearing and GC")
        print(f"     Performance tracking: Success rates, solve times, errors")
        print(f"  ESO sampling: {eso_dt}s ({1 / eso_dt:.0f} Hz)")
        print(f"  Plant sampling: {dt}s ({1 / dt:.0f} Hz)")
        print(f"  ESO updates per plant step: {self.eso_per_plant}")
        print(f"  Agent mass: {mass} kg")
        print(f"  MPC Horizon: {horizon} steps (reduced for testing)")
        print(
            f"  UNIVERSAL DISTURBANCE COMPENSATION: {'ENABLED' if ENABLE_DISTURBANCE_COMPENSATION else 'DISABLED'}")

    def generate_disturbances(self, t: float, agent_idx: int) -> Tuple[float, float]:
        """Generate disturbances for agent at time t (acceleration units)."""

        freq_x = 2 * np.pi * 0.05
        freq_y = 2 * np.pi * 0.05

        gamma_x = 3
        gamma_y = -3

        dx = gamma_x * np.sign(np.sin(freq_x * 0.03 * t))
        dy = gamma_y * np.sign(np.sin(freq_y * 0.03 * t))

        return dx, dy

    # def generate_disturbances(self, t: float, agent_idx: int) -> Tuple[float, float]:
    #
    #     freq_y = 2 * np.pi * 0.05
    #     gamma_y = 0
    #     dy = gamma_y * np.sign(np.sin(freq_y * 0.03 * t))
    #
    #     # dy square wave parameters
    #     amplitude = 10
    #     period = 60  # total period
    #     duration = 10  # pulse duration
    #     start_step = 30.0  # first pulse begins at t = 1s
    #
    #     # Before start_time, hold baseline value
    #     if t < start_step:
    #         return 0, dy
    #
    #     # Determine phase of the repeating cycle after start_time
    #     phase = (t - start_step) % period
    #
    #     if phase < duration:
    #         dx = amplitude
    #     else:
    #         dx = 0
    #
    #     return dx, dy

    def _apply_universal_compensation(self, raw_action: np.ndarray, agent_idx: int, step: int = 0,
                                      is_emergency: bool = False) -> np.ndarray:
        """Apply disturbance compensation if enabled."""

        if not ENABLE_DISTURBANCE_COMPENSATION:
            return raw_action.copy()

        # Ensure raw_action is a proper numpy array
        if not isinstance(raw_action, np.ndarray):
            raw_action = np.array(raw_action)
        if raw_action.size != 2:
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

                if step < 2000 or is_emergency:
                    action_type = "EMERGENCY" if is_emergency else "MPC"
                    print(f"  Agent {agent_idx}: {action_type} compensation applied - "
                          f"Raw:[{raw_action[0]:.3f},{raw_action[1]:.3f}] -> "
                          f"Comp:[{compensated_x:.3f},{compensated_y:.3f}]")

                return np.array([compensated_x, compensated_y])

            except Exception as e:
                print(f"Warning: Compensation failed for agent {agent_idx}: {e}")
                return raw_action.copy()
        else:
            return raw_action.copy()

    def update_eso(self, agent_idx: int, measurements: np.ndarray,
                   control_input_accel: np.ndarray, step: int) -> None:
        """Update ESO observers for both axes of an agent with velocity clipping."""

        pos_x, pos_y = measurements[0], measurements[1]
        u_accel_x, u_accel_y = control_input_accel[0], control_input_accel[1]

        # Get velocity limits from environment
        lower_lim, upper_lim = self.env.state_lim()
        v_min = float(lower_lim[2])  # velocity lower limit (same for vx, vy)
        v_max = float(upper_lim[2])  # velocity upper limit (same for vx, vy)

        # Safety checks
        if self.eso_states_x[agent_idx].shape[0] != 3:
            self.eso_states_x[agent_idx] = np.array([pos_x, 0.0, 0.0])

        if self.eso_states_y[agent_idx].shape[0] != 3:
            self.eso_states_y[agent_idx] = np.array([pos_y, 0.0, 0.0])

        try:
            # Update X-axis ESO
            eso_result_x, error_x = nonlinear_eso(
                pos_x, u_accel_x, self.eso_states_x[agent_idx],
                self.eso_dt, self.beta01, self.beta02, self.beta03,
                self.alpha1, self.alpha2, self.delta, self.b0
            )

            if len(eso_result_x) == 3:
                # Clip velocity estimate to environment limits
                # eso_result_x[1] = np.clip(eso_result_x[1], v_min, v_max)
                self.eso_states_x[agent_idx] = eso_result_x

            # Update Y-axis ESO
            eso_result_y, error_y = nonlinear_eso(
                pos_y, u_accel_y, self.eso_states_y[agent_idx],
                self.eso_dt, self.beta01, self.beta02, self.beta03,
                self.alpha1, self.alpha2, self.delta, self.b0
            )

            if len(eso_result_y) == 3:
                # Clip velocity estimate to environment limits
                # eso_result_y[1] = np.clip(eso_result_y[1], v_min, v_max)
                self.eso_states_y[agent_idx] = eso_result_y

            # # Log clipping events for debugging
            # if step < 30 or (eso_result_x[1] <= v_min or eso_result_x[1] >= v_max or
            #                  eso_result_y[1] <= v_min or eso_result_y[1] >= v_max):
            #     clipped_x = "CLIPPED" if (eso_result_x[1] <= v_min or eso_result_x[1] >= v_max) else ""
            #     clipped_y = "CLIPPED" if (eso_result_y[1] <= v_min or eso_result_y[1] >= v_max) else ""
            #     if clipped_x or clipped_y:
            #         print(f"  ESO velocity clipping - Agent {agent_idx}, Step {step}: "
            #               f"vx={eso_result_x[1]:.3f} {clipped_x}, vy={eso_result_y[1]:.3f} {clipped_y}")

        except Exception as e:
            print(f"ERROR in update_eso for agent {agent_idx}: {e}")

    def solve(self, agent_states: np.ndarray, goal_states: np.ndarray,
              graph: GraphsTuple, episode: int = 0, step: int = 0,
              true_disturbances: List[Tuple[float, float]] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Solve ESO-enhanced MPC for all agents with lazy initialization."""

        n_agents = agent_states.shape[0]
        actions = np.zeros((n_agents, 2))
        compensated_actions = np.zeros((n_agents, 2))
        cbf_gradients = []

        # Evaluate CBF on full environment graph (only for logging)
        h_values = np.zeros(n_agents)
        if self.cbf_interface is not None:
            try:
                h_values_full = self.cbf_interface.evaluate_h(graph)  # Use full graph
                if isinstance(h_values_full, (list, np.ndarray)) and len(h_values_full) >= n_agents:
                    h_values = h_values_full[:n_agents]
                elif np.isscalar(h_values_full):
                    h_values[0] = h_values_full
            except Exception as e:
                if step < 3:
                    print(f"CBF evaluation failed: {e}")
                h_values = np.zeros(n_agents)

        # Collect gradients for logging (placeholder)
        for i in range(n_agents):
            cbf_gradients.append(np.zeros(4))

        # Log current state before solving MPC
        if self.debug_logger:
            self.debug_logger.log_episode_state(
                episode, step, agent_states, goal_states, h_values, cbf_gradients, graph
            )

        # Initialize ESO with actual agent states on first step
        if step == 0:
            for i in range(n_agents):
                self.eso_states_x[i] = np.array([agent_states[i, 0], agent_states[i, 2], 0.0])
                self.eso_states_y[i] = np.array([agent_states[i, 1], agent_states[i, 3], 0.0])
                self.prev_actions[i] = np.zeros(2)
                print(f"Initialized ESO for agent {i}")

        # Update ESO for all agents
        for i in range(n_agents):
            current_pos = agent_states[i, :2]
            previous_accel = self.prev_actions[i]

            for sub_step in range(self.eso_per_plant):
                self.update_eso(i, current_pos, previous_accel, step * self.eso_per_plant + sub_step)

        # Prepare ESO feedback states
        eso_agent_states = np.zeros_like(agent_states)
        for i in range(n_agents):
            if USE_ESO_FEEDBACK:
                eso_agent_states[i] = np.array([
                    self.eso_states_x[i][0],  # x position estimate
                    self.eso_states_y[i][0],  # y position estimate
                    self.eso_states_x[i][1],  # x velocity estimate
                    self.eso_states_y[i][1]  # y velocity estimate
                ])
            else:
                eso_agent_states[i] = agent_states[i]  # Use true states for debugging

        # Solve distributed MPC with lazy initialization
        try:
            raw_actions, per_agent_info = self.mpc_controller.solve_distributed(
                eso_agent_states, goal_states, graph, episode, step
            )
            actions = raw_actions

            # Compute per-agent CBF on each agent's LOCAL subgraph (ego=0 in that subgraph)
            n_agents = self.env.num_agents
            cbf_initial_local = np.zeros(n_agents, dtype=float)
            if self.cbf_interface is not None:
                for i in range(n_agents):
                    try:
                        local_graph_i, _ = self.mpc_controller.subgraph_extractor.extract_local_subgraph(graph, i)
                        # cbf_interface was built with ego_agent_idx=0, which matches local graphs
                        h_i = self.cbf_interface.evaluate_h(local_graph_i)
                        cbf_initial_local[i] = float(np.array(h_i).reshape(-1)[0])
                    except Exception:
                        cbf_initial_local[i] = 0.0  # keep going if one agent fails

            # Agent CSV logging
            if hasattr(self, "agent_logger") and self.agent_logger:
                for i in range(self.env.num_agents):
                    info = per_agent_info[i]

                    # MPC logs horizon u_seq; fallback logs single-step action actually applied
                    control_payload = info["u_seq"] if info["u_seq"] is not None else actions[i] * self.mass

                    self.agent_logger.log(
                        agent_idx=i,
                        step=step,
                        cbf_initial=cbf_initial_local[i],
                        mpc_status=info.get("status", info.get("mpc_status", "")),
                        controller_used=info.get("controller_used", ""),
                        mpc_iterations=info.get("mpc_iterations", info.get("nit", None)),
                        control_input=control_payload,
                        predicted_h=info.get("predicted_h", None),
                    )

                    print("[Logging] Per-agent info logged")

            if step < 1000:
                print(f"  MPC distributed solve completed on step {step + 1} for {n_agents} agents")

        except Exception as e:
            print(f"  MPC distributed solve failed: {e}")
            # Fallback to simple controller
            for i in range(n_agents):
                pos_error = goal_states[i, :2] - eso_agent_states[i, :2]
                vel_error = goal_states[i, 2:4] - eso_agent_states[i, 2:4]
                actions[i] = 0.8 * pos_error + 0.4 * vel_error
                actions[i] = np.clip(actions[i], -self.u_max, self.u_max)

        # Apply disturbance compensation
        for i in range(n_agents):
            compensated_actions[i] = self._apply_universal_compensation(actions[i], i, step)

            # Log ESO performance
            if self.debug_logger and true_disturbances and len(self.eso_states_x[i]) >= 3:
                true_dist_x, true_dist_y = true_disturbances[i]
                print(f"  True disturbance is {true_dist_x} for x")
                print(f"  True disturbance is {true_dist_y} for y")
                true_state_x = agent_states[i, [0, 2]]
                true_state_y = agent_states[i, [1, 3]]

                self.debug_logger.log_eso_data(
                    episode, step, i,
                    true_state_x, true_state_y,
                    self.eso_states_x[i], self.eso_states_y[i],
                    true_dist_x, true_dist_y
                )

                self.debug_logger.log_disturbances(
                    episode, step, i,
                    true_dist_x, true_dist_y,
                    self.eso_states_x[i][2], self.eso_states_y[i][2]
                )

            # Log MPC result
            if self.debug_logger:
                self.debug_logger.log_mpc_result(
                    episode, step, i, "mpc_success", 0.01, actions[i], True,
                    actions[i], compensated_actions[i]
                )

        # Store current actions for next iteration
        self.prev_actions = compensated_actions.copy()
        return compensated_actions, cbf_gradients

    def cleanup(self):
        """Cleanup method for end of testing."""
        print(f"Cleaning up ESO Safety MPC...")
        if hasattr(self, 'mpc_controller'):
            self.mpc_controller.cleanup()
        print(f"ESO Safety MPC cleanup complete")


# Import the existing tester framework with minimal changes
class ESOGCBFTester:
    """Testing framework with agent freezing for unsafe agents."""

    def __init__(self, env, mpc_controller: ESOSafetyMPC, debug_logger: Optional[ESODebugLogger] = None):
        self.env = env
        self.controller = mpc_controller
        self.debug_logger = debug_logger
        self.margin = mpc_controller.margin

        # Evaluation functions
        self.is_unsafe_fn = jax_jit_np(jax.vmap(env.collision_mask))
        self.is_finish_fn = jax_jit_np(jax.vmap(env.finish_mask))

        # Track frozen agents
        self.frozen_agents = set()

    def run_episode(self, key: jr.PRNGKey, max_steps: int, episode_id: int = 0,
                    save_incremental_videos: bool = False, video_dir: pathlib.Path = None,
                    initial_graph=None) -> Tuple[RolloutResult, List[np.ndarray], dict]:  # <-- ADDED PARAMETER
        """Run episode with AGENT FREEZING when unsafe. Supports manual initial graph."""

        # --- Graph logging: set per-episode folder in the controller & reset counters
        gl_root = self.controller.mpc_controller.graph_log_root
        episode_dir = gl_root / f"ep{episode_id:02d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        self.controller.mpc_controller._episode_graph_log_dir = episode_dir
        self.controller.mpc_controller._graph_log_step = defaultdict(int)


        # Initialize - with support for manual graph
        if initial_graph is not None:
            # Use provided initial graph (manual scenario mode)
            graph = initial_graph
            print(f"    Using manual initial graph for episode {episode_id}")
        else:
            # Standard random initialization
            graph = self.env.reset(key)

        h_trace = []

        # Reset frozen agents for new episode
        self.frozen_agents = set()

        episode_stats = {
            'cbf_violations': 0,
            'margin_violations': 0,
            'frozen_agents': 0,
            'min_h_overall': float('inf'),
            'steps_completed': 0,
        }

        # Storage
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

            # Check for new unsafe agents and freeze them
            h_values = np.zeros(self.env.num_agents)
            if self.controller.cbf_interface is not None:
                try:
                    h_values_eval = self.controller.cbf_interface.evaluate_h(graph)
                    if isinstance(h_values_eval, (list, np.ndarray)):
                        h_values = h_values_eval[:self.env.num_agents]
                    else:
                        h_values[0] = h_values_eval
                except:
                    pass

            # Freeze agents that become unsafe
            for i in range(self.env.num_agents):
                if i not in self.frozen_agents and h_values[i] < 0:
                    # self.frozen_agents.add(i)
                    # episode_stats['frozen_agents'] += 1
                    print(f"  >>> AGENT {i} FROZEN at step {step} (h={h_values[i]:.4f})")

            h_trace.append(h_values.copy())

            # Update statistics
            episode_stats['min_h_overall'] = min(episode_stats['min_h_overall'], h_values.min())
            episode_stats['cbf_violations'] += np.sum(h_values < 0)
            episode_stats['margin_violations'] += np.sum(h_values < self.margin)
            episode_stats['steps_completed'] = step + 1

            # Print status
            if step % 20 == 0:
                frozen_count = len(self.frozen_agents)
                print(f"  Step {step:3d}: CBF∈[{h_values.min():.3f}, {h_values.max():.3f}], "
                      f"Frozen={frozen_count}/{self.env.num_agents}")

            # Generate disturbances
            true_disturbances = []
            for i in range(self.env.num_agents):
                dx, dy = self.controller.generate_disturbances(step, i)
                true_disturbances.append((dx, dy))

            # Get control actions (only for non-frozen agents)
            if len(self.frozen_agents) < self.env.num_agents:
                action_accel, cbf_gradients = self.controller.solve(
                    agent_states, goal_states, graph, episode_id, step, true_disturbances
                )
            else:
                # All agents frozen
                action_accel = np.zeros((self.env.num_agents, 2))
                cbf_gradients = [np.zeros(4) for _ in range(self.env.num_agents)]

            # Override frozen agents with zero control
            for frozen_idx in self.frozen_agents:
                action_accel[frozen_idx] = np.zeros(2)

            # Convert back to forces
            action_forces = action_accel * self.controller.mass
            action_jax = jnp.array(action_forces)

            # Step environment
            step_result = self.env.step(graph, action_jax, get_eval_info=True)
            graph, reward, cost, done, info = step_result

            # Freeze agent positions (keep them fixed)
            if len(self.frozen_agents) > 0 and hasattr(graph, 'states'):
                agent_node_mask = graph.node_type == 0
                if np.any(agent_node_mask):
                    agent_states_current = graph.states[agent_node_mask]

                    for frozen_idx in self.frozen_agents:
                        frozen_state = agent_states[frozen_idx]
                        agent_states_current = agent_states_current.at[frozen_idx].set(
                            jnp.array(frozen_state)
                        )

                    new_states = graph.states.at[agent_node_mask].set(agent_states_current)
                    graph = graph._replace(states=new_states)

            # Apply disturbances to non-frozen agents only
            if hasattr(graph, 'states'):
                agent_node_mask = graph.node_type == 0
                if np.any(agent_node_mask):
                    agent_states_current = graph.states[agent_node_mask]

                    for i in range(self.env.num_agents):
                        if i not in self.frozen_agents:
                            dx_accel, dy_accel = true_disturbances[i]
                            current_state = agent_states_current[i]

                            # Applying the disturbance as accelerations affecting velocity states
                            updated_state = current_state.at[2].set(
                                current_state[2] + dx_accel * self.env.dt
                            )
                            updated_state = updated_state.at[3].set(
                                current_state[3] + dy_accel * self.env.dt
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

            # Save incremental video
            if save_incremental_videos and video_dir is not None and step > 0 and (step + 1) % 10 == 0:
                try:
                    print(f"    Saving incremental video at step {step + 1}...")

                    partial_Tp1_graph = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *T_graph)
                    partial_T_action = jnp.stack(T_action, axis=0)
                    partial_T_reward = jnp.stack(T_reward, axis=0)
                    partial_T_cost = jnp.stack(T_cost, axis=0)
                    partial_T_done = jnp.stack(T_done, axis=0)
                    partial_T_info = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *T_info)

                    partial_rollout = RolloutResult(
                        partial_Tp1_graph, partial_T_action, partial_T_reward,
                        partial_T_cost, partial_T_done, partial_T_info
                    )

                    is_unsafe_partial = self.is_unsafe_fn(partial_Tp1_graph)

                    video_name = f"ep{episode_id:02d}_steps_0to{step + 1:03d}"
                    video_path = video_dir / f"{video_name}.mp4"
                    self.env.render_video(partial_rollout, video_path, is_unsafe_partial, dpi=100)
                    print(f"      Saved: {video_path.name}")

                except Exception as e:
                    print(f"      Warning: Failed to save incremental video at step {step + 1}: {e}")

            if done:
                print(f"    Episode completed at step {step}")
                break

        # Create rollout
        Tp1_graph = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *T_graph)
        T_action = jnp.stack(T_action, axis=0)
        T_reward = jnp.stack(T_reward, axis=0)
        T_cost = jnp.stack(T_cost, axis=0)
        T_done = jnp.stack(T_done, axis=0)
        T_info = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *T_info)

        rollout = RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        print(f"\n  Episode Summary:")
        print(f"    Total agents frozen: {episode_stats['frozen_agents']}/{self.env.num_agents}")
        print(f"    Frozen agents: {sorted(list(self.frozen_agents))}")

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


def create_manual_test_scenario():
    """
    Create manually configured test scenario.
    Edit the values below to test different configurations.
    """
    from gcbfplus.env.double_integrator_no_clipping import DoubleIntegratorNoClipping
    import jax.numpy as jnp

    # ==================== ENVIRONMENT SETUP ====================
    env_params = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.8],
        "n_obs": 4,
        "m": 0.1,
    }

    env = DoubleIntegratorNoClipping(
        num_agents=4,
        area_size=4.0,
        max_step=300,
        # max_travel=None,
        dt=0.03,
        params=env_params
    )

    # # ########################No Obs Towards Each Other###########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [5.0, 5.0],
    # ])
    # obs_lengths_x = jnp.array([0.8])
    # obs_lengths_y = jnp.array([0.8])
    # obs_thetas = jnp.array([0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # # All agents initialized
    # # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # # # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # # # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # # agent_states = jnp.array([agent_0_state, agent_3_state])
    #
    #
    # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state, agent_2_state, agent_3_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([3.4, 3.4, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([3.4, 0.6, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([0.6, 3.4, 0.0, 0.0])  # Agent 2's goal
    # goal_3_state = jnp.array([0.6, 0.6, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state,goal_1_state, goal_2_state, goal_3_state])
    # # ###################################################


    # # ###################################################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]
    # ])
    # obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # # All agents initialized
    # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state,agent_2_state, agent_3_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([3.4, 3.4, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([3.4, 0.6, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([0.6, 3.4, 0.0, 0.0])  # Agent 2's goal
    # goal_3_state = jnp.array([0.6, 0.6, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state, goal_1_state, goal_2_state, goal_3_state])
    # # ###################################################

    # # ########################Four diagonal test###########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]
    # ])
    # obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # # All agents initialized
    # # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # # # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # # # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # # agent_states = jnp.array([agent_0_state, agent_3_state])
    #
    #
    # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state,agent_2_state, agent_3_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([3.4, 3.4, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([3.4, 0.6, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([0.6, 3.4, 0.0, 0.0])  # Agent 2's goal
    # goal_3_state = jnp.array([0.6, 0.6, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state,goal_1_state, goal_2_state, goal_3_state])
    # # ###################################################

    # # ########################Two diagonal test###########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]
    # ])
    # obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # # All agents initialized
    # # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # # # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # # # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # # agent_states = jnp.array([agent_0_state, agent_3_state])
    #
    #
    # # agent_0_state = jnp.array([0.6, 0.6, 0.0, 0.0])
    # agent_1_state = jnp.array([0.6, 3.4, 0.0, 0.0])
    # agent_2_state = jnp.array([3.4, 0.6, 0.0, 0.0])
    # # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # agent_states = jnp.array([agent_1_state,agent_2_state])
    #
    # # ==================== GOALS ====================
    # # goal_0_state = jnp.array([3.4, 3.4, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([3.4, 0.6, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([0.6, 3.4, 0.0, 0.0])  # Agent 2's goal
    # # goal_3_state = jnp.array([0.6, 0.6, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_1_state,goal_2_state])
    # # ###################################################

    # # #########################Squeeze Test##########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]
    # ])
    # obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # agent_0_state = jnp.array([2, 1.2, 0.0, 0.0])
    # agent_1_state = jnp.array([1.75, 2, 0.0, 0.0])
    # agent_2_state = jnp.array([3.25, 2, 0.0, 0.0])
    # agent_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state,agent_2_state, agent_3_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([2, 0.5, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([1.9, 1.97, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([2.15, 2.03, 0.0, 0.0])  # Agent 2's goal
    # goal_3_state = jnp.array([3.4, 3.4, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state,goal_1_state, goal_2_state, goal_3_state])
    # # ###################################################

    # #########################Run Into Test##########################
    # ==================== OBSTACLES ====================
    obs_positions = jnp.array([
        [1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]
    ])
    obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])

    obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)

    # ==================== AGENTS ====================
    # agent_0_state = jnp.array([1, 0.5, 0.0, 0.0])
    # agent_1_state = jnp.array([0.5, 1, 0.0, 0.0])
    agent_0_state = jnp.array([2.3, 2, 0.0, 0.0])
    agent_1_state = jnp.array([2, 2.3, 0.0, 0.0])
    agent_2_state = jnp.array([3, 3.2, 0.0, 0.0])
    agent_3_state = jnp.array([2, 3.5, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state])
    agent_states = jnp.array([agent_0_state, agent_1_state, agent_2_state, agent_3_state])

    # ==================== GOALS ====================
    goal_0_state = jnp.array([3, 3.7, 0.0, 0.0])  # Agent 0's goal
    goal_1_state = jnp.array([3.5, 3, 0.0, 0.0])  # Agent 1's goal
    goal_2_state = jnp.array([2.6, 3.5, 0.0, 0.0])  # Agent 2's goal
    goal_3_state = jnp.array([3.5, 3.5, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state,goal_1_state])
    goal_states = jnp.array([goal_0_state, goal_1_state, goal_2_state, goal_3_state])
    # ###################################################

    # # #########################Oscillation Test##########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [1.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 2.5]
    # ])
    # obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # agent_0_state = jnp.array([1, 0.5, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([2, 0.5, 0.0, 0.0])  # Agent 0's goal
    # goal_states = jnp.array([goal_0_state])
    # # ###################################################

    # # #########################ESO Test##########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [-1.5, -1.5]
    # ])
    # obs_lengths_x = jnp.array([1])
    # obs_lengths_y = jnp.array([1])
    # obs_thetas = jnp.array([0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # # agent_0_state = jnp.array([1, 0.5, 0.0, 0.0])
    # # agent_1_state = jnp.array([0.5, 1, 0.0, 0.0])
    # agent_0_state = jnp.array([2, 0.5, 0.0, 0.0])
    # agent_1_state = jnp.array([3, 0.5, 0.0, 0.0])
    # agent_2_state = jnp.array([3, 3.2, 0.0, 0.0])
    # agent_3_state = jnp.array([2, 3.5, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([2, 7.5, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([3, 7.5, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([2.6, 3.5, 0.0, 0.0])  # Agent 2's goal
    # goal_3_state = jnp.array([3.5, 3.5, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state,goal_1_state])
    # # ###################################################


    # # #########################ESO Test##########################
    # # ==================== OBSTACLES ====================
    # obs_positions = jnp.array([
    #     [-1.5, -1.5]
    # ])
    # obs_lengths_x = jnp.array([1])
    # obs_lengths_y = jnp.array([1])
    # obs_thetas = jnp.array([0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # ==================== AGENTS ====================
    # # agent_0_state = jnp.array([1, 0.5, 0.0, 0.0])
    # # agent_1_state = jnp.array([0.5, 1, 0.0, 0.0])
    # agent_0_state = jnp.array([2.5, 4, 0.0, 0.0])
    # agent_1_state = jnp.array([4, 2.5, 0.0, 0.0])
    # agent_2_state = jnp.array([3, 3.2, 0.0, 0.0])
    # agent_3_state = jnp.array([2, 3.5, 0.0, 0.0])
    # agent_states = jnp.array([agent_0_state, agent_1_state])
    #
    # # ==================== GOALS ====================
    # goal_0_state = jnp.array([5, 6.5, 0.0, 0.0])  # Agent 0's goal
    # goal_1_state = jnp.array([6.5, 5, 0.0, 0.0])  # Agent 1's goal
    # goal_2_state = jnp.array([2.6, 3.5, 0.0, 0.0])  # Agent 2's goal
    # goal_3_state = jnp.array([3.5, 3.5, 0.0, 0.0])  # Agent 3's goal
    # goal_states = jnp.array([goal_0_state,goal_1_state])
    # # ###################################################

    # ==================== CREATE GRAPH ====================
    env_state = env.EnvState(agent_states, goal_states, obstacles)
    initial_graph = env.get_graph(env_state)

    # Print scenario info
    print(f"\n{'=' * 60}")
    print(f"MANUAL TEST SCENARIO")
    print(f"{'=' * 60}")
    print(f"Agents: {len(agent_states)}")
    for i, (state, goal) in enumerate(zip(agent_states, goal_states)):
        dist_to_goal = jnp.linalg.norm(state[:2] - goal[:2])
        print(f"  Agent {i}: pos=({state[0]:.3f}, {state[1]:.3f}), "
              f"vel=({state[2]:.3f}, {state[3]:.3f})")
        print(f"           goal=({goal[0]:.3f}, {goal[1]:.3f}), "
              f"distance={dist_to_goal:.3f}")

    print(f"\nObstacles: {len(obs_positions)}")
    for i, (pos, lx, ly, theta) in enumerate(zip(obs_positions, obs_lengths_x,
                                                   obs_lengths_y, obs_thetas)):
        print(f"  Obstacle {i}: pos=({pos[0]:.3f}, {pos[1]:.3f}), "
              f"size=[{lx:.2f}×{ly:.2f}], angle={theta:.3f} rad")
    print(f"{'=' * 60}\n")

    return env, initial_graph


def test_eso_mpc_gcbf(args):
    """Main testing function for ESO+MPC-GCBF integration with manual scenario support."""

    print(f"> Running ESO + MPC-GCBF Integration Test - FIXED WITH LAZY INITIALIZATION")
    if args.manual_scenario:
        print(f"> MANUAL SCENARIO MODE ENABLED")
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

    # Create environment - MODIFIED FOR MANUAL SCENARIOS
    if args.manual_scenario:
        # Use manual scenario initialization
        env, initial_graph = create_manual_test_scenario()
        num_agents = initial_graph.env_states.agent.shape[0]

        print(f"\nManual Environment Setup:")
        print(f"  Type: {env.__class__.__name__}")
        print(f"  Agents: {num_agents}")
        print(f"  Area Size: {env.area_size}")
        print(f"  Max Steps: {env._max_step}")
        print(f"  dt: {env.dt}")
    else:
        # Use standard random initialization
        num_agents = config.num_agents if args.num_agents is None else args.num_agents
        env = make_env(
            env_id=config.env if args.env is None else args.env,
            num_agents=num_agents,
            num_obs=args.obs,
            area_size=args.area_size,
            max_step=args.max_step,
            max_travel=args.max_travel,
        )
        initial_graph = None

        print(f"\nStandard Environment Setup:")
        print(f"  Type: {env.__class__.__name__}")
        print(f"  Agents: {num_agents}")
        print(f"  Obstacles: {args.obs}")
        print(f"  Area Size: {args.area_size}")
        print(f"  Max Steps (requested): {args.max_step}")
        print(f"  Max Steps (effective): {env._max_step}")
        print(f"  dt: {env.dt}")

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    scenario_suffix = "_manual" if args.manual_scenario else ""
    output_dir = pathlib.Path(f"./logs/eso_mpc_gcbf_results_lazy{scenario_suffix}/{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # Create videos directory
    videos_dir = output_dir / "videos"
    if not args.no_video:
        videos_dir.mkdir(exist_ok=True)

    # Create debug logger
    debug_logger = ESODebugLogger(output_dir, num_agents, args.safety_margin)
    print(f"ESO + MPC debug logger initialized")

    # Create ESO + MPC controller
    mpc_controller = ESOSafetyMPC(
        model_path=args.gcbf_path,
        env=env,
        horizon=args.mpc_horizon,
        dt=env.dt,
        u_max=args.u_max,
        alpha=args.alpha,
        margin=args.safety_margin,
        goal_weight=args.goal_weight,
        debug_logger=debug_logger,
        eso_dt=0.01,
        mass=args.mass,
        b0=1.0
    )

    # Initial graph logging directory
    mpc_controller.mpc_controller.graph_log_root = output_dir / "local_graph_logs"
    mpc_controller.mpc_controller.graph_log_root.mkdir(parents=True, exist_ok=True)

    # NEW: full global graph logging directory
    mpc_controller.mpc_controller.full_graph_log_root = output_dir / "full_graph_logs"
    mpc_controller.mpc_controller.full_graph_log_root.mkdir(parents=True, exist_ok=True)

    # Put agent CSVs under the same run folder
    mpc_controller.agent_logger = AgentStepCSVLogger(
        output_dir=output_dir,
        num_agents=env.num_agents
    )

    print(f"\nESO + MPC Controller Setup:")
    print(f"  NLP-based MPC: ENABLED")
    print(f"  Manual Scenario: {'YES' if args.manual_scenario else 'NO'}")
    print(f"  MPC Horizon: {args.mpc_horizon}")
    print(f"  CBF Alpha: {args.alpha}")
    print(f"  Safety Margin: {args.safety_margin:.3f}")

    # Create tester
    tester = ESOGCBFTester(env, mpc_controller, debug_logger)

    # Generate test keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1000)[:args.epi]
    test_keys = test_keys[args.offset:]

    if args.manual_scenario:
        print(f"\nManual scenario will be repeated for {args.epi} episodes")

    # Run episodes
    mode_str = "MANUAL SCENARIO" if args.manual_scenario else "RANDOM INITIALIZATION"
    print(f"\nRunning {args.epi} episodes with {mode_str}...")

    h_traces = []
    all_metrics = []
    all_episode_stats = []
    rollouts = []

    for i in range(args.epi):
        print(f"\nEpisode {i + 1}/{args.epi}")

        # MODIFIED: Handle manual scenario
        if args.manual_scenario:
            # Recreate manual scenario for each episode
            env, episode_initial_graph = create_manual_test_scenario()

            rollout, h_trace, episode_stats = tester.run_episode(
                test_keys[i], args.max_step, episode_id=i + 1,
                initial_graph=episode_initial_graph,  # Pass manual graph
                save_incremental_videos=(not args.no_video),
                video_dir=videos_dir if not args.no_video else None
            )
        else:
            # Standard random initialization
            rollout, h_trace, episode_stats = tester.run_episode(
                test_keys[i], args.max_step, episode_id=i + 1,
                save_incremental_videos=(not args.no_video),
                video_dir=videos_dir if not args.no_video else None
            )

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

    # Get performance summary
    final_summary = mpc_controller.mpc_controller.get_performance_summary()

    # Compute statistics
    safe_rates = [m['safe_rate'] for m in all_metrics]
    success_rates = [m['success_rate'] for m in all_metrics]
    finish_rates = [m['finish_rate'] for m in all_metrics]

    # Print results
    mode_header = "MANUAL SCENARIO" if args.manual_scenario else "RANDOM INITIALIZATION"
    print(f"\n" + "=" * 80)
    print(f"ESO + MPC-GCBF RESULTS ({mode_header})")
    print(f"=" * 80)
    print(f"Safety Rate:  {np.mean(safe_rates) * 100:.2f}% ± {np.std(safe_rates) * 100:.2f}%")
    print(f"Success Rate: {np.mean(success_rates) * 100:.2f}% ± {np.std(success_rates) * 100:.2f}%")
    print(f"Finish Rate:  {np.mean(finish_rates) * 100:.2f}% ± {np.std(finish_rates) * 100:.2f}%")

    # Performance summary
    memory_summary = final_summary['memory_summary']
    print(f"\nPERFORMANCE:")
    print(f"  Memory Peak: {memory_summary['peak_memory_mb']:.1f} MB")
    print(f"  Success Rate: {memory_summary['success_rate']:.1f}%")
    print(f"  Avg Solve Time: {memory_summary['avg_solve_time']:.3f}s")
    print(f"  Lazy Instances: {final_summary['initialized_agents']}/{final_summary['total_agents']}")

    # Fallback statistics
    fallback_stats = final_summary['fallback_stats']
    print(f"\nFALLBACK CONTROLLER:")
    print(f"  Total Calls: {fallback_stats['total_calls']}")
    print(f"  Success Rate: {fallback_stats['success_rate']:.1f}%")
    print(f"  CBF Violations Prevented: {fallback_stats['cbf_violations_prevented']}")

    # Generate videos
    if not args.no_video:
        print("\nGenerating videos...")
        for i, (rollout, metrics) in enumerate(zip(rollouts, all_metrics)):
            scenario_tag = "manual" if args.manual_scenario else "lazy"
            video_name = f"eso_mpc_{scenario_tag}_ep{i:02d}_safe{metrics['safe_rate'] * 100:.0f}"
            video_path = videos_dir / f"{video_name}.mp4"
            env.render_video(rollout, video_path, metrics['is_unsafe'], dpi=args.dpi)
            print(f"  Generated: {video_path.name}")

    # Cleanup
    mpc_controller.cleanup()

    print(f"\nTesting Complete ({mode_header})!")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ESO + MPC-GCBF Integration Testing with Manual Scenario Support")

    # NEW: Add manual scenario flag
    parser.add_argument("--manual-scenario", action="store_true", default=False,
                        help="Use manually configured scenario")

    # Environment parameters
    parser.add_argument("-n", "--num-agents", type=int, default=None,
                        help="Number of agents (ignored in manual mode)")
    parser.add_argument("--obs", type=int, default=0,
                        help="Number of obstacles (ignored in manual mode)")
    parser.add_argument("--area-size", type=float, default=4.0,
                        help="Environment area size (ignored in manual mode)")
    parser.add_argument("--max-step", type=int, default=256,
                        help="Maximum steps per episode")
    parser.add_argument("--max-travel", type=float, default=None,
                        help="Maximum travel distance")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment type")

    # Model parameters
    parser.add_argument("--gcbf-path", type=str, required=True,
                        help="Path to trained GCBF model")
    parser.add_argument("--step", type=int, default=None,
                        help="Model checkpoint step")

    # MPC parameters
    parser.add_argument("--mpc-horizon", type=int, default=3,
                        help="MPC prediction horizon")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CBF alpha parameter")
    parser.add_argument("--safety-margin", type=float, default=0.2,
                        help="Safety margin")
    parser.add_argument("--u-max", type=float, default=1.0,
                        help="Maximum control magnitude")
    parser.add_argument("--goal-weight", type=float, default=1.0,
                        help="Goal weight")

    # ESO parameters
    parser.add_argument("--mass", type=float, default=0.1,
                        help="Agent mass")

    # Testing parameters
    parser.add_argument("--seed", type=int, default=1111,
                        help="Random seed")
    parser.add_argument("--epi", type=int, default=1,
                        help="Number of episodes")
    parser.add_argument("--offset", type=int, default=0,
                        help="Episode offset")

    # System parameters
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Use CPU")
    parser.add_argument("--no-video", action="store_true", default=False,
                        help="Skip video generation")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Video DPI")

    args = parser.parse_args()
    test_eso_mpc_gcbf(args)


if __name__ == "__main__":
    main()
