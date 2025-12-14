#!/usr/bin/env python3
"""
Debug script to verify which agent we're actually evaluating in the CBF evaluator.
"""

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_evaluator import CBFEvaluator
from test_graph_update8 import create_test_scenario


def debug_agent_indexing(model_path: str):
    """
    Debug function to verify we're evaluating the correct agent.
    """
    print("=" * 70)
    print("DEBUGGING AGENT INDEXING IN CBF EVALUATOR")
    print("=" * 70)

    try:
        # Load evaluator and create scenario
        evaluator = CBFEvaluator(model_path, ego_agent_idx=0)
        env, initial_graph = create_test_scenario()

        # Get agent states
        agent_states = initial_graph.type_states(type_idx=0, n_type=env.num_agents)
        print(f"\nAgent states in graph:")
        for i in range(env.num_agents):
            state = agent_states[i]
            print(f"  Agent {i}: pos=({state[0]:.3f}, {state[1]:.3f}), vel=({state[2]:.3f}, {state[3]:.3f})")

        # From test scenario, we know:
        # ego_state = [0.8, 1.0, -1, -1.2]  <- Should be agent 0
        # other_state = [0.6, 0.8, 2.5, -0.4] <- Should be agent 1

        expected_ego_pos = np.array([0.8, 1.0])
        expected_other_pos = np.array([0.6, 0.8])

        print(f"\nExpected positions:")
        print(f"  Ego (should be agent 0): pos=({expected_ego_pos[0]:.3f}, {expected_ego_pos[1]:.3f})")
        print(f"  Other (should be agent 1): pos=({expected_other_pos[0]:.3f}, {expected_other_pos[1]:.3f})")

        # Check if agent 0 matches expected ego position
        agent0_pos = agent_states[0, :2]
        agent1_pos = agent_states[1, :2]

        ego_match_0 = np.allclose(agent0_pos, expected_ego_pos, atol=0.01)
        ego_match_1 = np.allclose(agent1_pos, expected_ego_pos, atol=0.01)

        print(f"\nPosition matching:")
        print(f"  Agent 0 matches expected ego: {ego_match_0}")
        print(f"  Agent 1 matches expected ego: {ego_match_1}")

        if ego_match_0:
            print("✓ Agent 0 is the ego agent (CORRECT)")
        elif ego_match_1:
            print("⚠ Agent 1 is the ego agent - INDEX MISMATCH!")
        else:
            print("✗ Neither agent matches expected ego position")

        # Test CBF evaluation for both agents
        print(f"\n" + "-" * 50)
        print("CBF EVALUATION FOR ALL AGENTS")
        print("-" * 50)

        # Get CBF values for all agents
        h_all = evaluator.evaluate_cbf_all(initial_graph)
        print(f"All CBF values: {h_all}")
        print(f"CBF shapes: {h_all.shape}")

        for i in range(len(h_all)):
            h_val = float(h_all[i])
            safety = "SAFE" if h_val > 0 else "UNSAFE"
            print(f"  Agent {i}: h = {h_val:.6f} ({safety})")

        # Test with different ego indices
        print(f"\n" + "-" * 50)
        print("TESTING DIFFERENT EGO INDICES")
        print("-" * 50)

        for ego_idx in range(env.num_agents):
            test_evaluator = CBFEvaluator(model_path, ego_agent_idx=ego_idx)
            h_ego = test_evaluator.evaluate_h(initial_graph)
            jacobian = test_evaluator.evaluate_jacobian(initial_graph)

            agent_pos = agent_states[ego_idx, :2]
            agent_vel = agent_states[ego_idx, 2:]

            print(f"\nEgo index {ego_idx}:")
            print(f"  Position: ({agent_pos[0]:.3f}, {agent_pos[1]:.3f})")
            print(f"  Velocity: ({agent_vel[0]:.3f}, {agent_vel[1]:.3f})")
            print(f"  CBF value: {h_ego:.6f}")
            print(f"  Jacobian: [{jacobian[0]:.4f}, {jacobian[1]:.4f}, {jacobian[2]:.4f}, {jacobian[3]:.4f}]")
            print(f"  |∇h|: {np.linalg.norm(jacobian):.6f}")

        # Physical reasoning check
        print(f"\n" + "-" * 50)
        print("PHYSICAL REASONING CHECK")
        print("-" * 50)

        # Agent at [0.8, 1.0] is closer to obstacles at [1.0, 1.2] and [0.6, 1.2]
        # Agent at [0.6, 0.8] is further from obstacles
        # So agent 0 should have LOWER CBF value (more dangerous)

        h0 = float(h_all[0])
        h1 = float(h_all[1])

        print(f"Agent 0 (pos=[0.8, 1.0]): h = {h0:.6f}")
        print(f"Agent 1 (pos=[0.6, 0.8]): h = {h1:.6f}")
        print(f"Agent 0 closer to obstacles: {h0 < h1}")

        if h0 < h1:
            print("✓ CBF values make physical sense")
        else:
            print("⚠ CBF values might be swapped or unexpected")

        # Test the evaluator's ego selection
        ego_evaluator = CBFEvaluator(model_path, ego_agent_idx=0)
        ego_h = ego_evaluator.evaluate_h(initial_graph)

        print(f"\nEvaluator with ego_idx=0 returns: {ego_h:.6f}")
        print(f"This matches h_all[0]: {abs(ego_h - h0) < 1e-8}")

        if abs(ego_h - h0) < 1e-8:
            print("✓ Evaluator correctly selects agent 0")
        else:
            print("✗ Evaluator indexing problem!")

    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()


def create_position_swapped_test():
    """Create a test where we manually swap agent positions to verify indexing."""
    print(f"\n" + "=" * 70)
    print("POSITION SWAP TEST")
    print("=" * 70)

    # This test isn't fully implemented but shows the concept
    print("To fully verify, we could:")
    print("1. Create graphs with swapped agent positions")
    print("2. Check if CBF evaluator follows the position or the index")
    print("3. Verify Jacobian points in correct direction for each agent")


if __name__ == "__main__":
    model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorMPC/gcbf+/seed0_20250626023916"

    debug_agent_indexing(model_path)
    create_position_swapped_test()