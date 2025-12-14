#!/usr/bin/env python3
"""
Obstacle Debugging Script for ESO+MPC GCBF Integration

This script helps diagnose why obstacles aren't showing in visualization
"""

import os
import sys
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# # Add your project path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcbfplus.env import make_env


def debug_obstacle_creation(gcbf_path: str, area_size: float = 4.0, num_obs: int = 6, num_agents: int = 4):
    """Debug obstacle creation and structure"""

    print("=" * 60)
    print("OBSTACLE DEBUGGING")
    print("=" * 60)

    # Load config
    with open(os.path.join(gcbf_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Create environment
    env = make_env(
        env_id=config.env,
        num_agents=num_agents,
        num_obs=num_obs,  # Should create obstacles
        area_size=area_size,
        max_step=256,
        max_travel=None,
    )

    print(f"Environment created: {env.__class__.__name__}")
    print(f"Environment parameters: {env._params}")

    # Reset environment to generate obstacles
    key = jr.PRNGKey(1111)
    initial_graph = env.reset(key)

    print(f"\nInitial graph structure:")
    print(f"  States shape: {initial_graph.states.shape}")
    print(f"  Node types: {jnp.unique(initial_graph.node_type)}")
    print(f"  Env states type: {type(initial_graph.env_states)}")

    # Debug obstacle structure
    obstacles = initial_graph.env_states.obstacle
    print(f"\nObstacle structure analysis:")
    print(f"  Obstacle type: {type(obstacles)}")
    print(f"  Obstacle length: {len(obstacles) if hasattr(obstacles, '__len__') else 'No len'}")

    # Try to access obstacle components
    if hasattr(obstacles, '__len__') and len(obstacles) > 0:
        for i in range(min(len(obstacles), 10)):  # Check first 10 components
            component = obstacles[i]
            print(
                f"  obstacles[{i}]: {type(component)} - shape: {component.shape if hasattr(component, 'shape') else 'no shape'}")

    # Check if obstacles[5] exists (what visualization expects)
    try:
        if hasattr(obstacles, '__len__') and len(obstacles) > 5:
            vertices = obstacles[5]
            print(f"\nFound obstacles[5] (vertices):")
            print(f"  Type: {type(vertices)}")
            print(f"  Shape: {vertices.shape if hasattr(vertices, 'shape') else 'no shape'}")
            if hasattr(vertices, 'shape') and len(vertices.shape) >= 2:
                print(f"  Number of obstacles: {vertices.shape[0]}")
                print(f"  Vertices per obstacle: {vertices.shape[1] if len(vertices.shape) > 1 else 'N/A'}")
                return vertices
    except Exception as e:
        print(f"  Error accessing obstacles[5]: {e}")

    # Try alternative obstacle access methods
    print(f"\nTrying alternative obstacle access...")

    # Check if it's a named tuple or has attributes
    if hasattr(obstacles, '_fields'):
        print(f"  NamedTuple fields: {obstacles._fields}")

    # Check common obstacle attributes
    for attr in ['vertices', 'positions', 'shapes', 'polygons']:
        if hasattr(obstacles, attr):
            val = getattr(obstacles, attr)
            print(
                f"  Found attribute '{attr}': {type(val)} - shape: {val.shape if hasattr(val, 'shape') else 'no shape'}")

    return None


def visualize_obstacles_debug(vertices, area_size: float = 4.0):
    """Create a debug visualization of obstacles"""

    if vertices is None:
        print("No obstacle vertices found for visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    print(f"\nCreating obstacle visualization...")
    print(f"Vertices shape: {vertices.shape}")

    n_obstacles = vertices.shape[0]
    print(f"Number of obstacles to plot: {n_obstacles}")

    for i in range(n_obstacles):
        obs_vertices = vertices[i]
        print(f"  Obstacle {i} vertices shape: {obs_vertices.shape}")
        print(f"  Obstacle {i} vertices: {obs_vertices}")

        try:
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
            ax.text(center[0], center[1], f'Obs{i}', fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        except Exception as e:
            print(f"  Error plotting obstacle {i}: {e}")

    ax.set_title(f'Obstacle Debug Visualization ({n_obstacles} obstacles)', fontsize=14)
    plt.tight_layout()
    plt.savefig('obstacle_debug.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Debug visualization saved as 'obstacle_debug.png'")


def test_alternative_obstacle_plotting(graph, area_size: float = 4.0):
    """Test alternative ways to plot obstacles"""

    print(f"\nTesting alternative obstacle plotting methods...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    obstacles = graph.env_states.obstacle

    # Method 1: Try accessing different indices
    axes[0].set_title('Method 1: obstacles[5]')
    try:
        if len(obstacles) > 5:
            vertices = obstacles[5]
            for i, obs_vertices in enumerate(vertices):
                polygon = patches.Polygon(obs_vertices, alpha=0.3, color='red')
                axes[0].add_patch(polygon)
            axes[0].text(0.1, 0.9, f'Success: {len(vertices)} obstacles', transform=axes[0].transAxes)
    except Exception as e:
        axes[0].text(0.1, 0.9, f'Failed: {str(e)[:30]}...', transform=axes[0].transAxes)

    # Method 2: Try different obstacle structure
    axes[1].set_title('Method 2: obstacles[0] as positions')
    try:
        if len(obstacles) > 0:
            positions = obstacles[0]  # Might be positions
            lengths_x = obstacles[1] if len(obstacles) > 1 else jnp.ones(len(positions)) * 0.2
            lengths_y = obstacles[2] if len(obstacles) > 2 else jnp.ones(len(positions)) * 0.2

            for i, (pos, lx, ly) in enumerate(zip(positions, lengths_x, lengths_y)):
                rect = patches.Rectangle(
                    (pos[0] - lx / 2, pos[1] - ly / 2), lx, ly,
                    alpha=0.3, color='blue'
                )
                axes[1].add_patch(rect)
            axes[1].text(0.1, 0.9, f'Success: {len(positions)} obstacles', transform=axes[1].transAxes)
    except Exception as e:
        axes[1].text(0.1, 0.9, f'Failed: {str(e)[:30]}...', transform=axes[1].transAxes)

    # Method 3: Extract from lidar hits
    axes[2].set_title('Method 3: From LiDAR data')
    try:
        # Get lidar nodes
        lidar_mask = graph.node_type == 2
        if jnp.any(lidar_mask):
            lidar_states = graph.states[lidar_mask]

            # Plot lidar hit points
            axes[2].scatter(lidar_states[:, 0], lidar_states[:, 1],
                            c='orange', s=10, alpha=0.7, label='LiDAR hits')
            axes[2].text(0.1, 0.9, f'LiDAR: {len(lidar_states)} points', transform=axes[2].transAxes)
    except Exception as e:
        axes[2].text(0.1, 0.9, f'Failed: {str(e)[:30]}...', transform=axes[2].transAxes)

    plt.tight_layout()
    plt.savefig('obstacle_methods_debug.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Alternative methods visualization saved as 'obstacle_methods_debug.png'")


if __name__ == "__main__":
    # Update this path to your GCBF model
    gcbf_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorMPC/gcbf+/seed0_20250626023916"

    # Debug obstacle creation
    vertices = debug_obstacle_creation(gcbf_path, area_size=4.0, num_obs=6, num_agents=4)

    # Create debug visualization if vertices found
    if vertices is not None:
        visualize_obstacles_debug(vertices, area_size=4.0)

    # Test with actual environment reset
    print(f"\n" + "=" * 60)
    print("TESTING WITH ACTUAL ENVIRONMENT")
    print("=" * 60)

    with open(os.path.join(gcbf_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    env = make_env(
        env_id=config.env,
        num_agents=4,
        num_obs=6,
        area_size=4.0,
    )

    key = jr.PRNGKey(1111)
    graph = env.reset(key)

    # Test alternative plotting methods
    test_alternative_obstacle_plotting(graph, area_size=4.0)

    print(f"\nDebugging complete! Check the generated PNG files.")
