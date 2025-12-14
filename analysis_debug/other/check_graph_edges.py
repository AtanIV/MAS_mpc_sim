#!/usr/bin/env python3
"""
Diagnostic script to check logged graph structure and edge connectivity.
"""

import pickle
import pathlib
import numpy as np
from typing import Dict, List, Tuple


def load_graph(pkl_path: pathlib.Path) -> Dict:
    """Load a single graph from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def analyze_edges(data: Dict) -> Dict[str, List[Tuple[int, int]]]:
    """Analyze and categorize edges by type (bidirectional detection)."""
    edges = {
        'agent_agent': [],
        'agent_goal': [],
        'agent_lidar': [],
        'goal_agent': [],
        'lidar_agent': [],
        'unknown': []
    }

    senders = data['senders']
    receivers = data['receivers']
    node_types = data['node_type']

    for s, r in zip(senders, receivers):
        s_type = node_types[s]
        r_type = node_types[r]

        # Skip padding nodes
        if s_type == -1 or r_type == -1:
            continue

        # Classify edge (check both directions)
        if s_type == 0 and r_type == 0:
            edges['agent_agent'].append((s, r))
        elif (s_type == 0 and r_type == 1) or (s_type == 1 and r_type == 0):
            # Agent↔Goal (either direction)
            if s_type == 0:
                edges['agent_goal'].append((s, r))  # agent→goal
            else:
                edges['goal_agent'].append((s, r))  # goal→agent
        elif (s_type == 0 and r_type == 2) or (s_type == 2 and r_type == 0):
            # Agent↔LiDAR (either direction)
            if s_type == 0:
                edges['agent_lidar'].append((s, r))  # agent→lidar
            else:
                edges['lidar_agent'].append((s, r))  # lidar→agent
        else:
            edges['unknown'].append((s, r, s_type, r_type))

    return edges


def analyze_nodes(data: Dict) -> Dict:
    """Analyze node types and structure."""
    node_types = data['node_type']
    states = data['states']

    # Count by type
    type_counts = {
        'agents': (node_types == 0).sum(),
        'goals': (node_types == 1).sum(),
        'lidar': (node_types == 2).sum(),
        'padding': (node_types == -1).sum()
    }

    # Get indices
    logical_mask = node_types != -1
    logical_indices = np.where(logical_mask)[0]

    agent_indices = np.where(node_types == 0)[0]
    goal_indices = np.where(node_types == 1)[0]
    lidar_indices = np.where(node_types == 2)[0]
    padding_indices = np.where(node_types == -1)[0]

    return {
        'counts': type_counts,
        'logical_indices': logical_indices,
        'agent_indices': agent_indices,
        'goal_indices': goal_indices,
        'lidar_indices': lidar_indices,
        'padding_indices': padding_indices,
        'states': states
    }


def print_detailed_edge_info(edges: Dict, node_info: Dict):
    """Print detailed information about each edge type."""
    states = node_info['states']

    print("\n" + "=" * 70)
    print("DETAILED EDGE ANALYSIS")
    print("=" * 70)

    # Agent-Agent edges
    if edges['agent_agent']:
        print(f"\n🔵 Agent-Agent Edges: {len(edges['agent_agent'])}")
        for i, (s, r) in enumerate(edges['agent_agent'][:5]):
            s_pos = states[s, :2]
            r_pos = states[r, :2]
            dist = np.linalg.norm(s_pos - r_pos)
            print(f"   [{i}] Agent {s} → Agent {r} | Distance: {dist:.3f}m")
        if len(edges['agent_agent']) > 5:
            print(f"   ... and {len(edges['agent_agent']) - 5} more")
    else:
        print(f"\n🔵 Agent-Agent Edges: 0")

    # Agent-Goal edges (both directions)
    total_agent_goal = len(edges['agent_goal']) + len(edges['goal_agent'])
    if total_agent_goal > 0:
        print(f"\n🟢 Agent↔Goal Edges: {total_agent_goal}")

        if edges['agent_goal']:
            print(f"   Agent→Goal: {len(edges['agent_goal'])}")
            for i, (s, r) in enumerate(edges['agent_goal'][:3]):
                s_pos = states[s, :2]
                r_pos = states[r, :2]
                dist = np.linalg.norm(s_pos - r_pos)
                print(f"     [{i}] Agent {s} → Goal {r} | dist={dist:.3f}m")

        if edges['goal_agent']:
            print(f"   Goal→Agent: {len(edges['goal_agent'])}")
            for i, (s, r) in enumerate(edges['goal_agent'][:3]):
                s_pos = states[s, :2]
                r_pos = states[r, :2]
                dist = np.linalg.norm(s_pos - r_pos)
                print(f"     [{i}] Goal {s} → Agent {r} | dist={dist:.3f}m")
    else:
        print(f"\n🟢 Agent↔Goal Edges: 0 ⚠️")

    # Agent-LiDAR edges (both directions)
    total_agent_lidar = len(edges['agent_lidar']) + len(edges['lidar_agent'])
    if total_agent_lidar > 0:
        print(f"\n🟠 Agent↔LiDAR Edges: {total_agent_lidar} ✓")

        if edges['agent_lidar']:
            print(f"   Agent→LiDAR: {len(edges['agent_lidar'])}")
            for i, (s, r) in enumerate(edges['agent_lidar'][:5]):
                s_pos = states[s, :2]
                r_pos = states[r, :2]
                dist = np.linalg.norm(s_pos - r_pos)
                print(f"     [{i}] Agent {s} → LiDAR {r} | dist={dist:.3f}m")

        if edges['lidar_agent']:
            print(f"   LiDAR→Agent: {len(edges['lidar_agent'])} (standard GCBF direction)")
            for i, (s, r) in enumerate(edges['lidar_agent'][:5]):
                s_pos = states[s, :2]
                r_pos = states[r, :2]
                dist = np.linalg.norm(s_pos - r_pos)
                print(f"     [{i}] LiDAR {s} → Agent {r} | dist={dist:.3f}m")
            if len(edges['lidar_agent']) > 5:
                print(f"     ... and {len(edges['lidar_agent']) - 5} more")
    else:
        print(f"\n🟠 Agent↔LiDAR Edges: 0 ❌")
        print("   ⚠️  NO AGENT-LIDAR EDGES FOUND!")

    # Unknown edges
    if edges['unknown']:
        print(f"\n⚪ Unknown Edges: {len(edges['unknown'])}")
        for i, (s, r, s_type, r_type) in enumerate(edges['unknown'][:5]):
            print(f"   [{i}] Node {s} (type={s_type}) → Node {r} (type={r_type})")


def check_lidar_connectivity_logic(node_info: Dict, edges: Dict):
    """Check if LiDAR nodes should be connected to agents."""
    num_agents = node_info['counts']['agents']
    num_lidar = node_info['counts']['lidar']

    # Count both directions
    total_agent_lidar = len(edges['agent_lidar']) + len(edges['lidar_agent'])

    print("\n" + "=" * 70)
    print("LIDAR CONNECTIVITY LOGIC CHECK")
    print("=" * 70)

    print(f"\nExpected connectivity:")
    print(f"  Agents: {num_agents}")
    print(f"  LiDAR hits: {num_lidar}")
    print(f"  Expected edges (if all active): {num_agents * num_lidar}")
    print(f"  Actual edges: {total_agent_lidar}")
    print(f"    Agent→LiDAR: {len(edges['agent_lidar'])}")
    print(f"    LiDAR→Agent: {len(edges['lidar_agent'])} (← standard GCBF direction)")

    if num_lidar > 0 and total_agent_lidar == 0:
        print("\n❌ PROBLEM: LiDAR nodes exist but NO edges!")
    elif num_lidar == 0:
        print("\n⚠️  No LiDAR hits in this graph")
    elif total_agent_lidar < num_agents * num_lidar:
        print(f"\n⚠️  Partial connectivity: only {total_agent_lidar}/{num_agents * num_lidar} possible edges")
        print("     (This is normal if some LiDAR rays are masked/inactive)")
    else:
        print("\n✓ Full connectivity (all agents connected to all LiDAR hits)")


def print_raw_edge_data(data: Dict, max_edges: int = 20):
    """Print raw edge data for debugging."""
    print("\n" + "=" * 70)
    print("RAW EDGE DATA")
    print("=" * 70)

    senders = data['senders']
    receivers = data['receivers']
    node_types = data['node_type']

    print(f"\nTotal edges: {len(senders)}")
    print(f"\nFirst {min(max_edges, len(senders))} edges:")
    print(f"{'Idx':<5} {'Sender':<8} {'S_Type':<8} {'Receiver':<10} {'R_Type':<8} {'Classification':<20}")
    print("-" * 70)

    for i in range(min(max_edges, len(senders))):
        s = senders[i]
        r = receivers[i]
        s_type = node_types[s]
        r_type = node_types[r]

        if s_type == -1 or r_type == -1:
            classification = "PADDING (ignored)"
        elif s_type == 0 and r_type == 0:
            classification = "Agent-Agent"
        elif s_type == 0 and r_type == 1:
            classification = "Agent-Goal"
        elif s_type == 0 and r_type == 2:
            classification = "Agent-LiDAR ⭐"
        else:
            classification = f"Unknown ({s_type}->{r_type})"

        print(f"{i:<5} {s:<8} {s_type:<8} {r:<10} {r_type:<8} {classification:<20}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Diagnostic tool for logged graphs')
    parser.add_argument('log_dir', type=str,
                        help='Path to local_graph_logs directory')
    parser.add_argument('--agent', type=int, default=0,
                        help='Agent index to check')
    parser.add_argument('--step', type=int, default=0,
                        help='Step number to check')
    parser.add_argument('--show-raw', action='store_true',
                        help='Show raw edge data')
    parser.add_argument('--check-multiple', action='store_true',
                        help='Check first 5 steps')

    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)

    if not log_dir.exists():
        print(f"❌ Error: {log_dir} does not exist")
        return

    # Find agent directory
    agent_dirs = sorted(log_dir.glob(f"ep*/agent{args.agent:02d}"))

    if not agent_dirs:
        print(f"❌ Error: No data found for agent {args.agent}")
        return

    agent_dir = agent_dirs[0]
    episode_name = agent_dir.parent.name

    print("=" * 70)
    print("GRAPH DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"\nAnalyzing: {agent_dir}")
    print(f"Episode: {episode_name}")
    print(f"Agent: {args.agent}")

    if args.check_multiple:
        # Check first 5 steps
        step_files = sorted(agent_dir.glob("step*.pkl"))[:5]

        print(f"\nChecking first {len(step_files)} steps...\n")

        for step_file in step_files:
            step_num = int(step_file.stem.replace('step', ''))
            print(f"\n{'=' * 70}")
            print(f"STEP {step_num}")
            print('=' * 70)

            data = load_graph(step_file)
            if data is None:
                continue

            node_info = analyze_nodes(data)
            edges = analyze_edges(data)

            print(f"\nNode counts: {node_info['counts']}")
            print(f"Edge counts: Agent-Agent={len(edges['agent_agent'])}, "
                  f"Agent-Goal={len(edges['agent_goal'])}, "
                  f"Agent-LiDAR={len(edges['agent_lidar'])}")

            if node_info['counts']['lidar'] > 0 and len(edges['agent_lidar']) == 0:
                print("❌ LIDAR EDGE ISSUE DETECTED!")

    else:
        # Check single step
        step_file = agent_dir / f"step{args.step:05d}.pkl"

        if not step_file.exists():
            print(f"\n❌ Error: {step_file} does not exist")
            return

        print(f"\nStep: {args.step}")
        print(f"File: {step_file}")

        data = load_graph(step_file)
        if data is None:
            return

        # Basic info
        print("\n" + "=" * 70)
        print("BASIC GRAPH INFO")
        print("=" * 70)
        print(f"\nKeys in data: {list(data.keys())}")
        print(f"States shape: {data['states'].shape}")
        print(f"Node types shape: {data['node_type'].shape}")
        print(f"Number of edges: {len(data['senders'])}")

        # Analyze nodes
        node_info = analyze_nodes(data)

        print("\n" + "=" * 70)
        print("NODE ANALYSIS")
        print("=" * 70)
        print(f"\nNode counts:")
        for node_type, count in node_info['counts'].items():
            print(f"  {node_type.capitalize():<10}: {count}")

        print(f"\nNode indices by type:")
        print(f"  Agents: {node_info['agent_indices']}")
        print(f"  Goals: {node_info['goal_indices']}")
        print(f"  LiDAR: {node_info['lidar_indices']}")

        # Analyze edges
        edges = analyze_edges(data)

        # In the main() function, replace the EDGE SUMMARY section:
        print("\n" + "=" * 70)
        print("EDGE SUMMARY")
        print("=" * 70)
        print(f"\n🔗 Edge counts:")
        print(f"  Agent-Agent: {len(edges['agent_agent'])}")
        print(f"  Agent→Goal:  {len(edges['agent_goal'])}")
        print(f"  Goal→Agent:  {len(edges['goal_agent'])} (standard GCBF)")
        print(f"  Agent→LiDAR: {len(edges['agent_lidar'])}")
        print(f"  LiDAR→Agent: {len(edges['lidar_agent'])} (standard GCBF)")

        total_ag = len(edges['agent_goal']) + len(edges['goal_agent'])
        total_al = len(edges['agent_lidar']) + len(edges['lidar_agent'])

        if total_ag == 0:
            print(f"  ⚠️  No agent-goal connectivity!")
        if node_info['counts']['lidar'] > 0 and total_al == 0:
            print(f"  ❌ MISSING: LiDAR nodes exist but no agent-lidar edges!")

        if edges['unknown']:
            print(f"  Unknown:     {len(edges['unknown'])}")

        # Detailed edge info
        print_detailed_edge_info(edges, node_info)

        # Check connectivity logic
        check_lidar_connectivity_logic(node_info, edges)

        # Raw edge data if requested
        if args.show_raw:
            print_raw_edge_data(data)

        # Final diagnosis
        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)

        if node_info['counts']['lidar'] > 0 and len(edges['agent_lidar']) == 0:
            print("\n❌ CRITICAL ISSUE: Agent-LiDAR edges are MISSING!")
            print("\nAction items:")
            print("1. Check graph_predictor.py, method: _construct_local_graph()")
            print("2. Look for the agent-lidar edge construction block")
            print("3. Verify it's creating edges between agents (type 0) and lidar (type 2)")
            print("4. Check if edges are being filtered somewhere before logging")
        elif node_info['counts']['lidar'] == 0:
            print("\n✓ No LiDAR hits in this step (might be expected)")
        else:
            print("\n✓ Agent-LiDAR edges are present!")
            print(f"   Found {len(edges['agent_lidar'])} connections")


if __name__ == "__main__":
    main()



#python check_graph_edges.py /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/eso_sqp_gcbf_results_lazy_manual/1020-0109/local_graph_logs --agent 0 --step 52
