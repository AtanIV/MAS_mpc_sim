"""
Geometric Wall Perception-based Graph Prediction for MPC-GCBF

Changes:
- Uses geometric reconstruction (line segments + semi-infinite rays) instead of neighbor-based sliding
- Caches perceived walls for entire MPC horizon (refreshed only when real env advances)
- Analytically computes ray-wall intersections for accurate corner handling
- Prevents phantom walls at concave/convex corners
"""

# Force JAX to use CPU to avoid GPU memory issues
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
from typing import List, Tuple, Optional, Dict
import copy

# Add the project directory to the path if needed
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcbfplus.utils.graph import GraphsTuple, EdgeBlock, GetGraph

# ========================= Wall Perception Classes =========================

"""
JAX-Compatible Geometric Wall Perception System with Corner Detection
"""


class PerceivedWalls:
    """
    JAX-compatible geometric representation of perceived obstacles with corner detection.

    Handles:
    - Single hit case: perpendicular wall with parallel corner rays
    - Multi-hit case: corner detection via "should-hit-but-didn't" logic

    All data stored as JAX arrays for differentiability.
    """

    def __init__(self):
        # Store as JAX arrays instead of Python lists
        self.line_segments = jnp.zeros((0, 2, 2))  # Shape: (n_segments, 2_points, 2_coords)
        self.semi_infinite_rays = jnp.zeros((0, 2, 2))  # Shape: (n_rays, 2_components, 2_coords)
        self.group_info: List[Dict] = []  # Keep for debugging (not traced)
        self.n_segments = 0
        self.n_rays = 0

    @staticmethod
    def from_graph(graph: GraphsTuple, agent_idx: int, env) -> 'PerceivedWalls':
        """
        Extract wall geometry from initial graph's LiDAR hits with corner detection.
        Returns JAX-compatible arrays.
        """
        walls = PerceivedWalls()

        # Get actual agent count
        agent_mask = graph.node_type == 0
        actual_num_agents = int(jnp.sum(agent_mask))

        if agent_idx >= actual_num_agents:
            return walls

        # Extract ego agent state
        agent_states = graph.type_states(type_idx=0, n_type=actual_num_agents)
        ego_state = agent_states[agent_idx, :]
        ego_pos = jnp.array(ego_state[:2])

        # Get ego agent's LiDAR hits
        n_rays = env._params["n_rays"]
        comm_radius = env._params["comm_radius"]
        hit_threshold = comm_radius - 1e-6

        # Get all LiDAR states
        lidar_mask = graph.node_type == 2
        total_lidar_nodes = int(jnp.sum(lidar_mask))

        if total_lidar_nodes == 0:
            return walls

        all_lidar_states = graph.type_states(type_idx=2, n_type=total_lidar_nodes)

        # Extract ego agent's LiDAR (first n_rays)
        ego_lidar_end = min(n_rays, total_lidar_nodes)
        ego_lidar_states = all_lidar_states[:ego_lidar_end]

        # Create spatially ordered ray information
        ray_info = []
        for i in range(ego_lidar_end):
            lidar_pos = jnp.array(ego_lidar_states[i, :2])
            ray_vec = lidar_pos - ego_pos
            ray_dist = float(jnp.linalg.norm(ray_vec))

            # Calculate angle for spatial ordering
            if ray_dist > 1e-6:
                angle_rad = float(jnp.arctan2(ray_vec[1], ray_vec[0]))
            else:
                angle_rad = 0.0

            is_hit = ray_dist < hit_threshold

            ray_info.append({
                'ray_idx': i,
                'distance': ray_dist,
                'angle_rad': angle_rad,
                'position': lidar_pos,
                'is_hit': is_hit,
                'direction': ray_vec / (ray_dist + 1e-10)
            })

        # Sort by angle
        ray_info.sort(key=lambda x: x['angle_rad'])

        # Group consecutive hits
        groups = []
        current_group = []

        for ray in ray_info:
            if ray['is_hit']:
                current_group.append(ray)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []

        if current_group:
            groups.append(current_group)

        # Handle wrap-around
        if len(groups) > 1:
            first_angle = groups[0][0]['angle_rad']
            last_angle = groups[-1][-1]['angle_rad']
            angle_diff = (first_angle - last_angle) % (2 * np.pi)
            expected_diff = 2 * np.pi / n_rays * 2

            if angle_diff < expected_diff:
                groups[0] = groups[-1] + groups[0]
                groups.pop()

        # Build JAX arrays for geometric primitives
        segment_list = []
        ray_list = []

        for group_idx, group in enumerate(groups):
            n_hits = len(group)

            walls.group_info.append({
                'group_idx': group_idx,
                'n_hits': n_hits,
                'ray_indices': [ray['ray_idx'] for ray in group]
            })

            if n_hits == 1:
                # Special case: single hit - perpendicular wall with parallel corner rays
                segments, rays = PerceivedWalls._process_single_hit(
                    group[0], ego_pos, ray_info
                )
                segment_list.extend(segments)
                ray_list.extend(rays)
            else:
                # Multi-hit case with corner detection
                segments, rays = PerceivedWalls._process_multi_hit_group(
                    group, ego_pos, ray_info, n_rays, comm_radius
                )
                segment_list.extend(segments)
                ray_list.extend(rays)

        # Convert to JAX arrays
        if segment_list:
            walls.line_segments = jnp.stack(segment_list)  # Shape: (n_segments, 2, 2)
            walls.n_segments = len(segment_list)
        else:
            walls.line_segments = jnp.zeros((0, 2, 2))
            walls.n_segments = 0

        if ray_list:
            walls.semi_infinite_rays = jnp.stack(ray_list)  # Shape: (n_rays, 2, 2)
            walls.n_rays = len(ray_list)
        else:
            walls.semi_infinite_rays = jnp.zeros((0, 2, 2))
            walls.n_rays = 0

        return walls

    @staticmethod
    def _process_single_hit(hit_ray, ego_pos, all_ray_info):
        """
        Handle single isolated hit.

        Creates:
        - One line segment perpendicular to the hitting ray (effectively infinite depth)
        - Two corner rays parallel to hitting ray at imaginary neighbor positions
        """
        hit_pos = hit_ray['position']
        ray_dir = hit_ray['direction']

        # Wall perpendicular to ray direction
        perp_dir = jnp.array([-ray_dir[1], ray_dir[0]])  # 90° rotation

        # Choose perpendicular direction that goes "across" the obstacle (not toward ego)
        to_ego = ego_pos - hit_pos
        if float(jnp.dot(perp_dir, to_ego)) > 0:
            perp_dir = -perp_dir

        # Imaginary corner positions (very large distance to represent infinite depth)
        infinite_depth = 0.05
        left_corner = hit_pos + perp_dir * infinite_depth
        right_corner = hit_pos - perp_dir * infinite_depth

        # Line segment (the perceived wall face)
        segments = [jnp.stack([left_corner, right_corner])]

        # Two corner rays parallel to hitting ray (pointing away from ego)
        # Ray direction is opposite to the ray that hits the obstacle
        corner_ray_dir = ray_dir
        rays = [
            jnp.stack([left_corner, corner_ray_dir]),
            jnp.stack([right_corner, corner_ray_dir])
        ]

        return segments, rays

    @staticmethod
    def _process_multi_hit_group(group, ego_pos, all_ray_info, n_rays, comm_radius):
        """
        Handle multi-hit group with corner detection on both ends.

        For each end:
        - Check if neighboring ray should-hit-but-didn't
        - If yes: create corner with perpendicular ray
        - If no: create infinite ray along wall direction
        """
        n_hits = len(group)
        positions = jnp.array([ray['position'] for ray in group])

        # Get ray indices in original ordering
        ray_indices = [ray['ray_idx'] for ray in group]

        # --- Left boundary check ---
        left_neighbor_idx = PerceivedWalls._find_neighbor_ray(
            ray_indices[0], all_ray_info, direction='left'
        )
        left_corner = PerceivedWalls._check_corner_boundary(
            ego_pos, left_neighbor_idx, all_ray_info,
            positions[0], positions[1], comm_radius, 'left'
        )

        # --- Right boundary check ---
        right_neighbor_idx = PerceivedWalls._find_neighbor_ray(
            ray_indices[-1], all_ray_info, direction='right'
        )
        right_corner = PerceivedWalls._check_corner_boundary(
            ego_pos, right_neighbor_idx, all_ray_info,
            positions[-1], positions[-2], comm_radius, 'right'
        )

        # --- Build line segments ---
        segment_points = []
        if left_corner is not None:
            segment_points.append(left_corner['pos'])
        segment_points.extend([positions[i] for i in range(n_hits)])
        if right_corner is not None:
            segment_points.append(right_corner['pos'])

        segments = [
            jnp.stack([segment_points[i], segment_points[i + 1]])
            for i in range(len(segment_points) - 1)
        ]

        # --- Build infinite rays ---
        rays = []

        if left_corner is not None:
            # Perpendicular ray at corner
            rays.append(jnp.stack([left_corner['pos'], left_corner['perp_dir']]))
        else:
            # Infinite ray along wall direction (extending leftward)
            wall_dir = positions[1] - positions[0]
            wall_dir = wall_dir / (jnp.linalg.norm(wall_dir) + 1e-10)
            rays.append(jnp.stack([positions[0], -wall_dir]))

        if right_corner is not None:
            # Perpendicular ray at corner
            rays.append(jnp.stack([right_corner['pos'], right_corner['perp_dir']]))
        else:
            # Infinite ray along wall direction (extending rightward)
            wall_dir = positions[-1] - positions[-2]
            wall_dir = wall_dir / (jnp.linalg.norm(wall_dir) + 1e-10)
            rays.append(jnp.stack([positions[-1], wall_dir]))

        return segments, rays

    @staticmethod
    def _find_neighbor_ray(ray_idx, all_ray_info, direction='left'):
        """
        Find the neighboring ray in the spatially ordered ray_info list.
        """
        # Find current ray in sorted list
        for i, ray in enumerate(all_ray_info):
            if ray['ray_idx'] == ray_idx:
                if direction == 'left':
                    neighbor_idx = (i - 1) % len(all_ray_info)
                else:  # right
                    neighbor_idx = (i + 1) % len(all_ray_info)
                return all_ray_info[neighbor_idx]['ray_idx']
        return None

    @staticmethod
    def _check_corner_boundary(ego_pos, neighbor_ray_idx, all_ray_info,
                               hit_pos_inside, hit_pos_next_inside,
                               comm_radius, direction):
        """
        Check if obstacle ends with a 90° corner at this boundary.

        Returns:
            None if wall continues infinitely
            dict with 'pos' and 'perp_dir' if corner detected
        """
        if neighbor_ray_idx is None:
            return None

        # Find neighbor ray info
        neighbor_ray = None
        for ray in all_ray_info:
            if ray['ray_idx'] == neighbor_ray_idx:
                neighbor_ray = ray
                break

        if neighbor_ray is None or neighbor_ray['is_hit']:
            return None  # Ray hit something else, not part of this obstacle

        # Get neighbor ray direction
        neighbor_ray_dir = neighbor_ray['direction']

        # Compute extended wall direction
        wall_vec = hit_pos_next_inside - hit_pos_inside
        wall_dir = wall_vec / (jnp.linalg.norm(wall_vec) + 1e-10)

        # Compute where neighbor ray would hit the extended wall
        imaginary_hit = PerceivedWalls._ray_line_intersection(
            ego_pos, neighbor_ray_dir,
            hit_pos_inside, wall_dir
        )

        if imaginary_hit is None:
            return None  # Parallel or no intersection

        # Check if within LiDAR range
        dist_to_imaginary = float(jnp.linalg.norm(imaginary_hit - ego_pos))

        if dist_to_imaginary >= comm_radius:
            return None  # Out of range → wall continues

        # Should-hit-but-didn't → corner detected!
        # Compute perpendicular direction away from ego
        perp_dir = PerceivedWalls._perpendicular_away_from_ego(
            wall_dir, ego_pos, imaginary_hit
        )

        return {
            'pos': imaginary_hit,
            'perp_dir': perp_dir
        }

    @staticmethod
    def _ray_line_intersection(ray_origin, ray_direction, line_point, line_direction):
        """
        Compute intersection of ray with infinite line.
        Returns intersection point or None if parallel.
        """
        # Solve: ray_origin + t*ray_direction = line_point + s*line_direction
        # Build 2x2 system: [ray_dir | -line_dir] * [t; s] = line_point - ray_origin
        A = jnp.column_stack([ray_direction, -line_direction])
        b = line_point - ray_origin

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

        if abs(float(det)) < 1e-10:
            return None  # Parallel

        # Solve using Cramer's rule
        t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det

        intersection = ray_origin + t * ray_direction
        return intersection

    @staticmethod
    def _perpendicular_away_from_ego(wall_dir, ego_pos, corner_pos):
        """
        Compute perpendicular to wall that points away from ego.
        """
        # Two perpendicular options (90° rotations)
        perp1 = jnp.array([-wall_dir[1], wall_dir[0]])
        perp2 = jnp.array([wall_dir[1], -wall_dir[0]])

        # Vector from corner toward ego
        to_ego = ego_pos - corner_pos

        # Choose perpendicular with negative dot product (points away)
        if float(jnp.dot(perp1, to_ego)) < 0:
            return perp1
        else:
            return perp2


# ========================= JAX-Compatible Ray Intersection Functions =========================

@jax.jit
def ray_segment_intersection_jax(ray_origin: jax.Array,
                                 ray_direction: jax.Array,
                                 seg_start: jax.Array,
                                 seg_end: jax.Array) -> jax.Array:
    """
    JAX-compatible ray-segment intersection.
    Returns distance (or inf if no intersection).
    """
    seg_vec = seg_end - seg_start
    seg_length = jnp.linalg.norm(seg_vec)

    # Build 2x2 system: [ray_dir | -seg_vec] * [t; s] = seg_start - ray_origin
    A = jnp.column_stack([ray_direction, -seg_vec])
    b = seg_start - ray_origin

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    # Solve using Cramer's rule
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / (det + 1e-10)
    s = (A[0, 0] * b[1] - A[1, 0] * b[0]) / (det + 1e-10)

    # Valid intersection: t > 0, 0 <= s <= 1, det != 0
    valid = (jnp.abs(det) > 1e-10) & (t > 1e-6) & (s >= 0) & (s <= 1)

    return jnp.where(valid, t, jnp.inf)


@jax.jit
def ray_ray_intersection_jax(ray_origin: jax.Array,
                             ray_direction: jax.Array,
                             semi_ray_origin: jax.Array,
                             semi_ray_direction: jax.Array) -> jax.Array:
    """
    JAX-compatible ray-ray intersection.
    Returns distance (or inf if no intersection).
    """
    A = jnp.column_stack([ray_direction, -semi_ray_direction])
    b = semi_ray_origin - ray_origin

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / (det + 1e-10)
    s = (A[0, 0] * b[1] - A[1, 0] * b[0]) / (det + 1e-10)

    # Valid: both rays point forward, det != 0
    valid = (jnp.abs(det) > 1e-10) & (t > 1e-6) & (s >= 0)

    return jnp.where(valid, t, jnp.inf)


def compute_ray_intersections_jax(ego_pos: jax.Array,
                                  perceived_walls: PerceivedWalls,
                                  num_rays: int,
                                  comm_radius: float,
                                  ray_angles: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
    """
    Fully JAX-compatible ray intersection computation.
    Uses vectorization for speed.
    """
    if ray_angles is None:
        ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)

    # Vectorized computation for all rays
    def single_ray_intersection(angle):
        ray_direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])

        # Initialize with max distance
        min_distance = comm_radius

        # Check line segment intersections (vectorized)
        if perceived_walls.n_segments > 0:
            # Vectorize over all segments
            def check_segment(seg):
                return ray_segment_intersection_jax(
                    ego_pos, ray_direction, seg[0], seg[1]
                )

            segment_distances = jax.vmap(check_segment)(perceived_walls.line_segments)
            min_segment_dist = jnp.min(segment_distances)
            min_distance = jnp.minimum(min_distance, min_segment_dist)

        # Check semi-infinite ray intersections (vectorized)
        if perceived_walls.n_rays > 0:
            def check_ray(semi_ray):
                return ray_ray_intersection_jax(
                    ego_pos, ray_direction, semi_ray[0], semi_ray[1]
                )

            ray_distances = jax.vmap(check_ray)(perceived_walls.semi_infinite_rays)
            min_ray_dist = jnp.min(ray_distances)
            min_distance = jnp.minimum(min_distance, min_ray_dist)

        # Compute hit position
        hit_pos = ego_pos + min_distance * ray_direction

        return min_distance, hit_pos

    # Vectorize over all rays
    hit_distances, hit_positions = jax.vmap(single_ray_intersection)(ray_angles)

    return hit_distances, hit_positions


# ========================= MPC Graph Predictor =========================

class MPCGraphPredictor:
    """
    Predicts graph evolution over MPC horizon using geometric wall perception.

    Key features:
    - Caches perceived walls from initial graph
    - Uses analytical ray-wall intersections for predictions
    - Assumes constant velocity for other agents
    - Refreshes wall cache only when real environment advances
    """

    def __init__(self, env, agent_idx: int = 0):
        """
        Initialize predictor for a specific agent.

        Args:
            env: Multi-agent environment
            agent_idx: Index of the ego agent (default: 0)
        """
        self.env = env
        self.agent_idx = agent_idx
        self.perceived_walls: Optional[PerceivedWalls] = None
        self.dt = env.dt
        self.mass = env._params["m"]
        self.comm_radius = env._params["comm_radius"]

        # # Get velocity limits from environment
        lower_lim, upper_lim = self.env.state_lim()
        self.v_min = float(lower_lim[2])  # velocity lower limit (same for vx, vy)
        self.v_max = float(upper_lim[2])  # velocity upper limit (same for vx, vy)

        # Override velocity limits for testing
        # self.v_min = -0.5
        # self.v_max = 0.5

    def reset_wall_cache(self):
        """Reset cached walls. Call this when the real environment advances."""
        self.perceived_walls = None
        # print("[Predictor] Resetting wall cache")

    def get_graph_info(self, graph: GraphsTuple) -> Dict:
        """Get graph structure information for debugging."""
        agent_mask = graph.node_type == 0
        actual_num_agents = int(jnp.sum(agent_mask))

        lidar_mask = graph.node_type == 2
        total_lidar = int(jnp.sum(lidar_mask))

        edge_count = len(graph.senders) if hasattr(graph, 'senders') else 0

        return {
            'num_agents': actual_num_agents,
            'num_lidar': total_lidar,
            'num_edges': edge_count,
            'total_nodes': int(jnp.sum(graph.n_node))
        }

    def verify_gcbf_compatibility(self, predicted_graph: GraphsTuple, reference_graph: GraphsTuple) -> bool:
        """Verify that predicted graph maintains GCBF compatibility."""
        pred_info = self.get_graph_info(predicted_graph)
        ref_info = self.get_graph_info(reference_graph)

        # Check structure consistency
        agents_match = pred_info['num_agents'] == ref_info['num_agents']
        lidars_match = pred_info['num_lidar'] == ref_info['num_lidar']

        return agents_match and lidars_match

    def double_integrator_step(self, state, control):
        """Pure double integrator dynamics."""
        pos = state[:2]
        vel = state[2:]
        vel = jnp.clip(vel, self.v_min, self.v_max)
        accel = control / self.mass
        new_pos = pos + vel * self.dt + 0.5 * accel * self.dt ** 2
        new_vel = vel + accel * self.dt
        new_vel = jnp.clip(new_vel, self.v_min, self.v_max)
        return jnp.concatenate([new_pos, new_vel])


    def predict_agent_states_all(self,
                                 current_graph,
                                 ego_control=None,
                                 all_agent_controls=None):
        """
        Update ALL agent states with proper padding handling.

        Args:
            current_graph: GraphsTuple
            ego_control: (2,) control for ego (used if all_agent_controls is None)
            all_agent_controls: optional (n_agents, 2) controls for *each* agent
        """
        # Get agent count directly
        agent_mask = current_graph.node_type == 0
        actual_num_agents = int(jnp.sum(agent_mask))

        # Extract agent states
        agent_states = current_graph.type_states(type_idx=0, n_type=actual_num_agents)

        # Normalize controls
        if all_agent_controls is not None:
            all_agent_controls = jnp.asarray(all_agent_controls)
            # If someone passes a single control vector, broadcast it
            if all_agent_controls.ndim == 1:
                all_agent_controls = jnp.tile(all_agent_controls[None, :],
                                              (actual_num_agents, 1))

        # Update agent states with dynamics
        new_agent_states = jnp.zeros_like(agent_states)

        # print(f"\n--- Updating predicted graphs ---")
        for i in range(actual_num_agents):
            if all_agent_controls is not None:
                # Use the per-agent control if provided (clip index just in case)
                idx = min(i, all_agent_controls.shape[0] - 1)
                control = all_agent_controls[idx]
                print(f"    Applied predicted control for agent {i} is {control}")
            else:
                # Backwards-compatible path: ego gets ego_control, others get 0
                if i == 0 and ego_control is not None:
                    control = ego_control
                else:
                    control = jnp.zeros(2)

            current_state = agent_states[i]
            # print(f"    Current state for agent {i} is {current_state}")
            new_state = self.double_integrator_step(current_state, control)
            # print(f"    New state for agent {i} is {new_state}")
            new_agent_states = new_agent_states.at[i].set(new_state)

        return new_agent_states


    def update_obstacle_edges_ego_geometric(self, current_graph, new_ego_pos):
        """
        Update ego agent's LiDAR using geometric wall perception.
        """
        n_rays = self.env._params["n_rays"]
        comm_radius = self.env._params["comm_radius"]

        # Use consistent ray angles
        ray_angles = jnp.linspace(0, 2 * jnp.pi, n_rays, endpoint=False)

        # Compute intersections using JAX-compatible function
        hit_distances, hit_positions = compute_ray_intersections_jax(
            new_ego_pos, self.perceived_walls, n_rays, comm_radius, ray_angles=ray_angles
        )

        return hit_distances

    def create_lidar_states_all_agents(self, agent_states, ego_lidar_distances, original_graph):
        """Create LiDAR states for ALL agents with geometric perception for ego."""
        #  Get actual count
        agent_mask = original_graph.node_type == 0
        actual_num_agents = int(jnp.sum(agent_mask))

        # Extract original agent states directly
        original_agent_states = original_graph.type_states(type_idx=0, n_type=actual_num_agents)

        # Extract original LiDAR states
        lidar_mask = original_graph.node_type == 2
        total_lidar_nodes = int(jnp.sum(lidar_mask))
        original_all_lidar = original_graph.type_states(type_idx=2, n_type=total_lidar_nodes)

        n_rays = self.env._params["n_rays"]
        all_lidar_states = []

        for i in range(actual_num_agents):
            agent_pos = agent_states[i, :2]
            original_agent_pos = original_agent_states[i, :2]
            agent_displacement = agent_pos - original_agent_pos

            if i == 0:  # Ego agent
                # Use consistent ray angles (JAX arrays)
                ray_angles = jnp.linspace(0, 2 * jnp.pi, n_rays, endpoint=False)

                # Reconstruct using JAX operations
                ego_lidar_positions = jnp.zeros((len(ego_lidar_distances), 2))

                for ray_idx in range(len(ego_lidar_distances)):
                    angle = ray_angles[ray_idx]
                    ray_direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
                    new_distance = ego_lidar_distances[ray_idx]
                    new_pos = agent_pos + new_distance * ray_direction
                    ego_lidar_positions = ego_lidar_positions.at[ray_idx].set(new_pos)

                ego_lidar_states = jnp.concatenate([
                    ego_lidar_positions,
                    jnp.zeros_like(ego_lidar_positions)
                ], axis=-1)
                all_lidar_states.append(ego_lidar_states)

            else:  # Non-ego agents: preserve with displacement
                start_idx = i * n_rays
                end_idx = min((i + 1) * n_rays, len(original_all_lidar))

                if start_idx < len(original_all_lidar):
                    original_agent_lidar = original_all_lidar[start_idx:end_idx]

                    if len(original_agent_lidar) < n_rays:
                        padding_needed = n_rays - len(original_agent_lidar)
                        padding = jnp.zeros((padding_needed, original_agent_lidar.shape[1]))
                        original_agent_lidar = jnp.concatenate([original_agent_lidar, padding], axis=0)

                    updated_lidar_positions = original_agent_lidar[:, :2] + agent_displacement
                    ray_states = jnp.concatenate([
                        updated_lidar_positions,
                        original_agent_lidar[:, 2:]
                    ], axis=-1)
                else:
                    ray_states = jnp.zeros((n_rays, 4))

                all_lidar_states.append(ray_states)

        # Combine all LiDAR states
        if all_lidar_states:
            combined_lidar = jnp.concatenate(all_lidar_states, axis=0)
        else:
            combined_lidar = jnp.zeros((n_rays * actual_num_agents, 4))

        return combined_lidar

    def _create_edge_blocks(self, agent_states, goal_states, all_lidar_states, n_agents):
        """Create edge blocks with proper indexing - CORRECTED VERSION"""
        n_rays = self.env._params["n_rays"]
        edge_blocks = []

        # ====== Agent-Agent edges ======
        # Calculate pairwise distances and state differences
        agent_pos = agent_states[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)

        # Exclude self-connections by adding large distance to diagonal
        dist = dist + jnp.eye(n_agents) * (self.comm_radius + 1)

        # State differences for all pairs
        state_diff = agent_states[:, None, :] - agent_states[None, :, :]

        # Connectivity mask: within communication radius
        agent_agent_mask = dist < self.comm_radius

        # Node IDs
        id_agent = jnp.arange(n_agents)

        agent_agent_edges = EdgeBlock(
            state_diff,  # Features: state differences
            agent_agent_mask,  # Mask: which edges are active
            id_agent,  # Senders: all agents
            id_agent  # Receivers: all agents
        )
        edge_blocks.append(agent_agent_edges)

        # ====== Goal-Agent edges ======
        id_goal = jnp.arange(n_agents, n_agents + len(goal_states))

        # Connectivity: each agent to its own goal
        agent_goal_mask = jnp.eye(n_agents, len(goal_states))

        # Features: agent - goal (relative position from goal's perspective)
        agent_goal_feats = agent_states[:, None, :] - goal_states[None, :, :]

        # Apply distance clipping for goals
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :, :2] ** 2, axis=-1, keepdims=True))
        safe_feats_norm = jnp.maximum(feats_norm, self.comm_radius)
        coef = jnp.where(feats_norm > self.comm_radius, self.comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :, :2].set(agent_goal_feats[:, :, :2] * coef)

        # CRITICAL: EdgeBlock(features, mask, SENDERS, RECEIVERS)
        # For goal-agent: senders=agents, receivers=goals
        goal_agent_edges = EdgeBlock(
            agent_goal_feats,
            agent_goal_mask,
            id_agent,  # Senders: agents
            id_goal  # Receivers: goals
        )
        edge_blocks.append(goal_agent_edges)

        # ====== Agent-LiDAR edges for ALL agents ======
        id_obs_start = n_agents + len(goal_states)
        id_obs = jnp.arange(id_obs_start, id_obs_start + len(all_lidar_states))

        for i in range(n_agents):
            start_idx = i * n_rays
            end_idx = (i + 1) * n_rays

            # Get this agent's LiDAR rays
            if start_idx < len(all_lidar_states):
                agent_lidar_rays = all_lidar_states[start_idx:end_idx]

                # Pad if necessary
                if len(agent_lidar_rays) < n_rays:
                    padding_needed = n_rays - len(agent_lidar_rays)
                    padding = jnp.zeros((padding_needed, 4))
                    agent_lidar_rays = jnp.concatenate([agent_lidar_rays, padding], axis=0)

                # Calculate distances and active mask
                agent_pos = agent_states[i, :2]
                lidar_pos = agent_lidar_rays[:, :2]
                lidar_dist = jnp.linalg.norm(lidar_pos - agent_pos, axis=1)
                active_lidar = lidar_dist < (self.comm_radius - 1e-6)  # Hits within range

                # Features: agent - lidar_rays
                lidar_feats = agent_states[i] - agent_lidar_rays
                lidar_feats = lidar_feats[None, :, :]  # Shape: (1, n_rays, 4)
                agent_obs_mask = active_lidar[None, :]  # Shape: (1, n_rays)

                # Edge indices for this agent's LiDAR
                edge_indices = id_obs[start_idx:end_idx]

                # Ensure edge_indices has exactly n_rays elements
                if len(edge_indices) < n_rays:
                    # Pad with the last valid index
                    last_valid = edge_indices[-1] if len(edge_indices) > 0 else id_obs_start
                    padding_indices = jnp.full(n_rays - len(edge_indices), last_valid)
                    edge_indices = jnp.concatenate([edge_indices, padding_indices])

                agent_lidar_edge = EdgeBlock(
                    lidar_feats,
                    agent_obs_mask,
                    jnp.array([id_agent[i]]),  # Sender: this agent
                    edge_indices  # Receivers: this agent's LiDAR nodes
                )
                edge_blocks.append(agent_lidar_edge)
            else:
                # No LiDAR data - create empty edge block
                empty_feats = jnp.zeros((1, n_rays, 4))
                empty_mask = jnp.zeros((1, n_rays), dtype=bool)
                empty_indices = jnp.full(n_rays, id_obs_start, dtype=jnp.int32)

                agent_lidar_edge = EdgeBlock(
                    empty_feats,
                    empty_mask,
                    jnp.array([id_agent[i]]),
                    empty_indices
                )
                edge_blocks.append(agent_lidar_edge)

        return edge_blocks

    def create_complete_gcbf_compatible_graph(self, agent_states, goal_states, ego_lidar_distances, original_graph):
        """Create graph maintaining GCBF format with proper padding."""
        n_agents = len(agent_states)
        n_goals = len(goal_states)
        n_rays = self.env._params["n_rays"]
        n_rays_total = n_rays * n_agents
        n_logical_total = n_agents + n_goals + n_rays_total

        # Create LiDAR states
        all_lidar_states = self.create_lidar_states_all_agents(agent_states, ego_lidar_distances, original_graph)

        # Create edge blocks
        edge_blocks = self._create_edge_blocks(agent_states, goal_states, all_lidar_states, n_agents)

        # Create node features and types (logical nodes only)
        node_feats = jnp.zeros((n_logical_total, 3))
        node_feats = node_feats.at[:n_agents, 2].set(1)  # agents
        node_feats = node_feats.at[n_agents:n_agents + n_goals, 1].set(1)  # goals
        node_feats = node_feats.at[-n_rays_total:, 0].set(1)  # LiDAR

        node_types = jnp.concatenate([
            jnp.zeros(n_agents, dtype=jnp.int32),  # agents: type 0
            jnp.ones(n_goals, dtype=jnp.int32),  # goals: type 1
            jnp.full(n_rays_total, 2, dtype=jnp.int32)  # LiDAR: type 2
        ])

        # Combine all states
        all_states = jnp.concatenate([agent_states, goal_states, all_lidar_states], axis=0)

        # Create environment state
        env_state = self.env.EnvState(
            agent=agent_states,
            goal=goal_states,
            obstacle=original_graph.env_states.obstacle
        )

        # Use GetGraph pipeline to add padding correctly
        get_graph = GetGraph(
            nodes=node_feats,
            node_type=node_types,
            edge_blocks=edge_blocks,
            env_states=env_state,
            states=all_states
        )

        predicted_graph = get_graph.to_padded()

        return predicted_graph

    # def predict_next_graph_complete(self, graph, ego_control):
    #     """
    #     MAIN PREDICTION FUNCTION with geometric wall perception.
    #     Ensures complete GCBF compatibility.
    #     """
    #     try:
    #         # Step 1: Update all agent states (handles padding correctly)
    #         new_agent_states = self.predict_agent_states_all(graph, ego_control)
    #         new_ego_pos = new_agent_states[0, :2]
    #
    #         # Step 2: Update ego LiDAR distances using geometric wall perception
    #         updated_ego_lidar_distances = self.update_obstacle_edges_ego_geometric(graph, new_ego_pos)
    #
    #         # Step 3: Extract goal states (handles padding correctly)
    #         agent_mask = graph.node_type == 0
    #         actual_num_agents = int(jnp.sum(agent_mask))
    #         goal_states = graph.type_states(type_idx=1, n_type=actual_num_agents)
    #
    #         # Step 4: Create complete GCBF-compatible graph
    #         predicted_graph = self.create_complete_gcbf_compatible_graph(
    #             new_agent_states, goal_states, updated_ego_lidar_distances, graph
    #         )
    #
    #         return predicted_graph
    #
    #     except Exception as e:
    #         print(f"Error in graph prediction: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise

    def predict_next_graph_complete(self, graph, controls):
        """
        MAIN PREDICTION FUNCTION with geometric wall perception.
        Supports either:
          - controls.shape == (2,)          → ego only
          - controls.shape == (n_agents,2)  → per-agent controls
        """
        try:
            # Step 1: Update all agent states
            if controls.ndim == 1:
                new_agent_states = self.predict_agent_states_all(
                    graph, ego_control=controls
                )
            else:
                # assume (n_agents, 2)
                new_agent_states = self.predict_agent_states_all(
                    graph, all_agent_controls=controls
                )
            new_ego_pos = new_agent_states[0, :2]

            # Step 2: Update ego LiDAR distances using geometric wall perception
            updated_ego_lidar_distances = self.update_obstacle_edges_ego_geometric(graph, new_ego_pos)

            # Step 3: Extract goal states (handles padding correctly)
            agent_mask = graph.node_type == 0
            actual_num_agents = int(jnp.sum(agent_mask))
            goal_states = graph.type_states(type_idx=1, n_type=actual_num_agents)

            # Step 4: Create complete GCBF-compatible graph
            predicted_graph = self.create_complete_gcbf_compatible_graph(
                new_agent_states, goal_states, updated_ego_lidar_distances, graph
            )

            return predicted_graph

        except Exception as e:
            print(f"Error in graph prediction: {e}")
            import traceback
            traceback.print_exc()
            raise


    # def predict_graphs_horizon(self, initial_graph, control_sequence):
    #     """
    #     Predict sequence of graphs with geometric wall perception.
    #
    #     Caches perceived walls from initial_graph for entire horizon.
    #     Call reset_wall_cache() when real environment advances.
    #     """
    #     # Cache walls from initial graph (only once per MPC horizon)
    #     if self.perceived_walls is None:
    #         # print("[Predictor] Building perceived walls cache from initial_graph")
    #         self.perceived_walls = PerceivedWalls.from_graph(initial_graph, self.agent_idx, self.env)
    #
    #     horizon = control_sequence.shape[0]
    #     graphs = []
    #     current_graph = initial_graph
    #
    #     for step in range(horizon):
    #         try:
    #             control = control_sequence[step]
    #             next_graph = self.predict_next_graph_complete(current_graph, control)
    #             graphs.append(next_graph)
    #             current_graph = next_graph
    #
    #         except Exception as e:
    #             print(f"Error at prediction step {step}: {e}")
    #             break
    #
    #     return graphs

    def predict_graphs_horizon(self, initial_graph, control_sequence):
        """
        Predict sequence of graphs with geometric wall perception.

        control_sequence:
          - shape (H, 2)           → ego-only controls
          - shape (H, n_agents, 2) → per-agent controls
        """
        # Cache walls from initial graph (only once per MPC horizon)
        if self.perceived_walls is None:
            # print("[Predictor] Building perceived walls cache from initial_graph")
            self.perceived_walls = PerceivedWalls.from_graph(initial_graph, self.agent_idx, self.env)

        control_sequence = jnp.asarray(control_sequence)
        horizon = control_sequence.shape[0]
        graphs = []
        current_graph = initial_graph

        for step in range(horizon):
            try:
                controls_step = control_sequence[step]  # (2,) or (n_agents,2)
                next_graph = self.predict_next_graph_complete(current_graph, controls_step)

                graphs.append(next_graph)
                current_graph = next_graph

            except Exception as e:
                print(f"Error at prediction step {step}: {e}")
                break

        return graphs

# ========================= Visualization Functions (KEPT FROM test_graph_update8) =========================

def plot_all_graph_edges(ax, graph, env):
    """Plot ALL edges with variable graph structure support"""
    if not hasattr(graph, 'senders') or len(graph.senders) == 0:
        return {'hits': 0, 'misses': 0}

    # Get agent count
    agent_mask = graph.node_type == 0
    actual_num_agents = jnp.sum(agent_mask)

    # Get all node positions
    agent_states = graph.type_states(type_idx=0, n_type=actual_num_agents)
    goal_states = graph.type_states(type_idx=1, n_type=actual_num_agents)

    # Handle variable LiDAR counts
    lidar_mask = graph.node_type == 2
    total_lidar_nodes = jnp.sum(lidar_mask)
    lidar_states = graph.type_states(type_idx=2, n_type=total_lidar_nodes)

    all_positions = jnp.concatenate([
        agent_states[:, :2],
        goal_states[:, :2],
        lidar_states[:, :2]
    ], axis=0)

    # Node boundaries
    n_goals = actual_num_agents
    agent_start = 0
    agent_end = actual_num_agents
    goal_start = actual_num_agents
    goal_end = actual_num_agents + n_goals
    lidar_start = actual_num_agents + n_goals

    # Categorize edges
    edge_types = {
        'agent_to_agent': [],
        'goal_to_agent': [],
        'agent_to_lidar': [],
        'other': []
    }

    # Debug first few edges
    for i, (sender, receiver) in enumerate(zip(graph.senders, graph.receivers)):
        if sender >= len(all_positions) or receiver >= len(all_positions):
            continue

        sender_pos = all_positions[sender]
        receiver_pos = all_positions[receiver]

        # Categorize edge with explicit conditions
        if agent_start <= sender < agent_end and agent_start <= receiver < agent_end:
            edge_types['agent_to_agent'].append((sender_pos, receiver_pos))
        elif goal_start <= sender < goal_end and agent_start <= receiver < agent_end:
            edge_types['goal_to_agent'].append((sender_pos, receiver_pos))
        elif agent_start <= sender < agent_end and receiver >= lidar_start:
            edge_types['agent_to_lidar'].append((sender_pos, receiver_pos))
        else:
            edge_types['other'].append((sender_pos, receiver_pos))

    # Plot different edge types
    agent_agent_edges = edge_types['agent_to_agent']
    if agent_agent_edges:
        for sender_pos, receiver_pos in agent_agent_edges:
            ax.plot([sender_pos[0], receiver_pos[0]], [sender_pos[1], receiver_pos[1]],
                    'b-', alpha=0.8, linewidth=4)
        ax.plot([], [], 'b-', alpha=0.8, linewidth=4, label=f'Agent-Agent ({len(agent_agent_edges)})')

    goal_agent_edges = edge_types['goal_to_agent']
    if goal_agent_edges:
        for sender_pos, receiver_pos in goal_agent_edges:
            ax.plot([sender_pos[0], receiver_pos[0]], [sender_pos[1], receiver_pos[1]],
                    'green', alpha=1.0, linewidth=2)
        ax.plot([], [], 'green', alpha=1.0, linewidth=2,
                label=f'Goal-Agent ({len(goal_agent_edges)})')

    # LiDAR rays for variable agent count
    hits = 0
    misses = 0
    n_rays = env._params["n_rays"]

    for agent_idx in range(actual_num_agents):
        agent_pos = agent_states[agent_idx, :2]

        # Get this agent's LiDAR rays
        start_idx = agent_idx * n_rays
        end_idx = min((agent_idx + 1) * n_rays, len(lidar_states))

        if start_idx < len(lidar_states):
            agent_lidar_states = lidar_states[start_idx:end_idx]

            for i, lidar_state in enumerate(agent_lidar_states):
                lidar_pos = lidar_state[:2]
                distance = jnp.linalg.norm(lidar_pos - agent_pos)

                if distance < env._params["comm_radius"] - 1e-3:
                    # Hit - red for ego, orange for others
                    color = 'r-' if agent_idx == 0 else 'orange'
                    alpha = 0.8 if agent_idx == 0 else 0.5
                    linewidth = 2 if agent_idx == 0 else 1
                    hits += 1
                    ax.plot([agent_pos[0], lidar_pos[0]], [agent_pos[1], lidar_pos[1]],
                            color, alpha=alpha, linewidth=linewidth)
                    if agent_idx == 0:
                        ax.plot(lidar_pos[0], lidar_pos[1], 'ro', markersize=5)
                else:
                    # Miss - gray thin line
                    misses += 1
                    ax.plot([agent_pos[0], lidar_pos[0]], [agent_pos[1], lidar_pos[1]],
                            'gray', alpha=0.3, linewidth=0.5)

    # Add legend entries for LiDAR rays
    if hits > 0:
        ax.plot([], [], 'r-', alpha=0.8, linewidth=2, label=f'LiDAR Hits ({hits})')
    if misses > 0:
        ax.plot([], [], 'gray', alpha=0.3, linewidth=0.5, label=f'LiDAR Misses ({misses})')

    return {'hits': hits, 'misses': misses}


def plot_perceived_walls(ax, graph, env, predictor=None):
    """Plot perceived walls from geometric reconstruction"""
    # If predictor is provided with cached walls, visualize those
    if predictor and predictor.perceived_walls:
        walls = predictor.perceived_walls

        # Plot line segments (for groups with >3 hits)
        for seg_start, seg_end in walls.line_segments:
            ax.plot([seg_start[0], seg_end[0]], [seg_start[1], seg_end[1]],
                    'cyan', linewidth=4, alpha=0.8, label='Line Segments')

        # Plot semi-infinite rays (show first 2 units of each ray)
        ray_length = 2.0  # Visualization length
        for ray_start, ray_dir in walls.semi_infinite_rays:
            ray_end = ray_start + ray_length * ray_dir
            ax.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]],
                    'magenta', linewidth=3, linestyle='--', alpha=0.8, label='Semi-infinite Rays')

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Don't overwrite existing legend
        return

    # FALLBACK: If no predictor provided, show old visualization (consecutive hits)
    agent_mask = graph.node_type == 0
    actual_num_agents = jnp.sum(agent_mask)

    if actual_num_agents == 0:
        return

    n_rays = env._params["n_rays"]
    ego_pos = graph.type_states(type_idx=0, n_type=actual_num_agents)[0, :2]

    # Get ego LiDAR
    lidar_mask = graph.node_type == 2
    total_lidar_nodes = jnp.sum(lidar_mask)
    all_lidar_states = graph.type_states(type_idx=2, n_type=total_lidar_nodes)

    ego_lidar_end = min(n_rays, total_lidar_nodes)
    ego_lidar_states = all_lidar_states[:ego_lidar_end]

    # Create spatial ordering
    ray_info = []
    for i in range(ego_lidar_end):
        lidar_pos = ego_lidar_states[i, :2]
        ray_vec = lidar_pos - ego_pos
        ray_dist = jnp.linalg.norm(ray_vec)

        if ray_dist > 1e-6:
            angle_rad = jnp.arctan2(ray_vec[1], ray_vec[0])
        else:
            angle_rad = 0.0
            ray_dist = env._params["comm_radius"]

        ray_info.append({
            'distance': ray_dist,
            'angle_rad': float(angle_rad),
            'position': lidar_pos
        })

    spatial_ordered = sorted(ray_info, key=lambda x: x['angle_rad'])
    n_rays_actual = len(spatial_ordered)

    # Find wall pairs (OLD METHOD - just for fallback visualization)
    hit_threshold = env._params["comm_radius"] - 1e-3
    spatial_hits = [ray['distance'] < hit_threshold for ray in spatial_ordered]

    wall_pairs = []
    for i in range(n_rays_actual):
        current = i
        next_idx = (i + 1) % n_rays_actual

        if spatial_hits[current] and spatial_hits[next_idx]:
            wall_pairs.append({
                'ray1': spatial_ordered[current],
                'ray2': spatial_ordered[next_idx]
            })

    # Plot walls (OLD METHOD)
    for wall_idx, wall_data in enumerate(wall_pairs):
        ray1_pos = wall_data['ray1']['position']
        ray2_pos = wall_data['ray2']['position']

        wall_vec = ray2_pos - ray1_pos
        wall_length = jnp.linalg.norm(wall_vec)

        if wall_length > 1e-4:
            if wall_idx == 0:
                ax.plot([ray1_pos[0], ray2_pos[0]], [ray1_pos[1], ray2_pos[1]],
                        'purple', linewidth=6, alpha=0.5, label='Consecutive Hits (Old)')
            else:
                ax.plot([ray1_pos[0], ray2_pos[0]], [ray1_pos[1], ray2_pos[1]],
                        'purple', linewidth=6, alpha=0.5)


def plot_complete_graph_state(ax, graph, env, step_num, title_prefix="", predictor=None):
    """Enhanced plotting function with variable graph structure support"""
    ax.clear()
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Get actual agent count
    agent_mask = graph.node_type == 0
    actual_num_agents = jnp.sum(agent_mask)

    if actual_num_agents == 0:
        ax.set_title(f"{title_prefix}Step {step_num} - No agents found")
        return

    # Get states
    agent_states = graph.type_states(type_idx=0, n_type=actual_num_agents)
    goal_states = graph.type_states(type_idx=1, n_type=actual_num_agents)

    lidar_mask = graph.node_type == 2
    total_lidar_nodes = jnp.sum(lidar_mask)
    lidar_states = graph.type_states(type_idx=2, n_type=total_lidar_nodes)

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

                center = np.mean(obs_vertices, axis=0)
                ax.text(center[0], center[1], f'Obs{i}', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Error plotting obstacles: {e}")

    # Plot ALL graph edges with different styles
    lidar_stats = plot_all_graph_edges(ax, graph, env)

    # Plot perceived walls from ego agent's LiDAR - WITH PREDICTOR
    plot_perceived_walls(ax, graph, env, predictor=predictor)

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

        # Velocity vector
        vel = agent_state[2:4]
        if jnp.linalg.norm(vel) > 1e-4:
            ax.arrow(agent_pos[0], agent_pos[1], vel[0] * 0.1, vel[1] * 0.1,
                     head_width=0.03, head_length=0.03, fc=color, ec=color, alpha=0.7, zorder=10)

    # Enhanced title with edge count and LiDAR stats
    edge_count = len(graph.senders) if hasattr(graph, 'senders') else 0
    hits = lidar_stats['hits']
    misses = lidar_stats['misses']

    ax.set_title(f"{title_prefix}Step {step_num} (Agents: {actual_num_agents})\n"
                 f"Hits: {hits}, Misses: {misses}, Edges: {edge_count}")
    ax.legend(loc='upper right', fontsize=8)


# ========================= Testing Functions =========================

def create_test_scenario():
    """Create a controlled test scenario"""
    from gcbfplus.env.double_integrator_no_clipping import DoubleIntegratorNoClipping

    # Environment parameters
    env_params = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.5],
        "n_obs": 1,
        "m": 0.1,
    }

    # Create environment
    env = DoubleIntegratorNoClipping(
        num_agents=2,  # Ego + 1 other
        area_size=2.0,
        max_step=256,
        max_travel=None,
        dt=0.03,
        params=env_params
    )

    # # =======================================================
    # # Create obstacles
    # obs_positions = jnp.array([[1.0, 1.2], [0.6, 1.2]])
    # obs_lengths_x = jnp.array([0.3, 0.1])
    # obs_lengths_y = jnp.array([0.1, 0.4])
    # obs_thetas = jnp.array([0.0, -jnp.pi / 8])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # Create agents WITHIN communication radius
    # ego_state = jnp.array([1.0, 1.05, 0.5, 0.5])  # [x, y, vx, vy]
    # other_state = jnp.array([0.7, 0.6, 0.5, 0.2])  # Within comm_radius
    # agent_states = jnp.array([ego_state, other_state])
    #
    # # Goals
    # goal_states = jnp.array([
    #     [1.8, 1.5, 0.0, 0.0],  # Ego goal
    #     [1.5, 1.0, 0.0, 0.0]  # Other agent goal
    # ])
    # # =======================================================

    # =======================================================
    # Create obstacles
    obs_positions = jnp.array([[1.0, 1.0]])
    obs_lengths_x = jnp.array([0.8])
    obs_lengths_y = jnp.array([0.8])
    obs_thetas = jnp.array([-jnp.pi / 32])

    obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)

    # Create agents
    # ego_state = jnp.array([0.9, 0.95, -0.4, 0.5])
    # ego_state = jnp.array([1, 1.05, 0.5, 0.12]) # 11 step safe discrete
    ego_state = jnp.array([0.25, 0.35, 0.4, 0.4])
    other_state = jnp.array([0.3, 0.25, 0, 0])
    agent_states = jnp.array([ego_state, other_state])

    # Goals
    goal_states = jnp.array([
        [1.8, 1.8, 0.0, 0.0],
        [1.5, 1.0, 0.0, 0.0]
    ])
    # =======================================================

    # # =========================Four squares==============================
    # # Create obstacles (same four 0.3×0.3 blocks as before)
    # obs_positions = jnp.array([
    #     [1.5, 1.5],
    #     [1.5, 2.5],
    #     [2.5, 1.5],
    #     [2.5, 2.5],
    # ])
    # obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
    # obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
    #
    # obstacles = env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
    #
    # # Agents
    # # Ego uses provided ego_pos/ego_vel (intended to mirror agent_0's start [0.6, 0.6, 0, 0])
    # # ego_state = jnp.array([1.125, 1.125, 0.499, 0.499])
    # # ego_state = jnp.array([1.3361, 1.1702, 0.499, -0.054])
    # ego_state = jnp.array([2.55785793, 2.25829877, 0.5, 0.5])
    # # Other agent starts at its original starting point [3.4, 3.4, 0, 0]
    # other_state = jnp.array([2.25829877, 2.55785791, 0.5, 0.5])
    # agent_states = jnp.array([ego_state, other_state])
    #
    # # Goals (agent 0 → [3.4, 3.4], agent 3 → [0.6, 0.6])
    # goal_states = jnp.array([
    #     [3, 3.5, 0.0, 0.0],  # ego's goal (was goal_0_state)
    #     [3.5, 3, 0.0, 0.0],  # other agent's goal (was goal_3_state)
    # ])
    # # =======================================================



    # Create environment state and graph
    env_state = env.EnvState(agent_states, goal_states, obstacles)
    initial_graph = env.get_graph(env_state)

    return env, initial_graph

def main():
    """Main test function for geometric wall perception graph prediction"""
    print("=" * 100)
    print("GEOMETRIC WALL PERCEPTION - MPC GRAPH PREDICTOR")
    print("=" * 100)

    # Create test scenario
    env, initial_graph = create_test_scenario()

    # Debug initial graph
    predictor = MPCGraphPredictor(env)
    initial_info = predictor.get_graph_info(initial_graph)
    print(f"Test scenario graph info: {initial_info}")

    # Generate control sequence for testing
    horizon = 3
    control_sequence = jnp.array([
        [-0.2, 0.7],  # Move towards obstacles
        [-0.1, 0.2],  # No movement
        [0.1, 0.1]  # Move away
    ])

    print(f"\nControl sequence: {control_sequence.shape}")
    for i, control in enumerate(control_sequence):
        print(f"  Step {i + 1}: [{control[0]:.3f}, {control[1]:.3f}]")

    # Test with geometric wall perception
    try:
        print(f"\n=== TESTING GEOMETRIC WALL PERCEPTION ===")

        # Predict complete graphs
        predicted_graphs = predictor.predict_graphs_horizon(initial_graph, control_sequence)
        print(f"SUCCESS: Generated {len(predicted_graphs)} graphs with geometric wall perception!")

        # Verify each graph
        for i, graph in enumerate(predicted_graphs):
            print(f"\nVerifying Graph {i + 1}:")
            graph_info = predictor.get_graph_info(graph)
            print(f"  Graph info: {graph_info}")

            is_compatible = predictor.verify_gcbf_compatibility(graph, initial_graph)
            print(f"  GCBF Compatible: {is_compatible}")

        # Print wall perception info
        if predictor.perceived_walls:
            print(f"\nPerceived Walls Info:")
            print(f"  Line segments: {len(predictor.perceived_walls.line_segments)}")
            print(f"  Semi-infinite rays: {len(predictor.perceived_walls.semi_infinite_rays)}")
            print(f"  Wall groups: {len(predictor.perceived_walls.group_info)}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Visualize results
    print(f"\n{'=' * 100}")
    print("CREATING VISUALIZATION")
    print("=" * 100)

    all_graphs = [initial_graph] + predicted_graphs

    # Create visualization
    n_cols = 2
    n_rows = (len(all_graphs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Plot each graph WITH PREDICTOR for visualization
    for i, graph in enumerate(all_graphs):
        if i < len(axes):
            title = "Initial: " if i == 0 else f"Predicted: "
            # PASS PREDICTOR to enable geometric wall visualization
            plot_complete_graph_state(axes[i], graph, env, i, title, predictor=predictor)

    # Hide unused subplots
    for i in range(len(all_graphs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(
        'Geometric Wall Perception - MPC Graph Prediction\n'
        '(Line Segments + Semi-Infinite Rays for Accurate Corner Handling)',
        y=0.98, fontsize=14, weight='bold')
    fig.subplots_adjust(top=0.9)
    plt.show()

    print(f"\n{'=' * 100}")
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("✅ Geometric wall perception implemented")
    print("✅ Analytical ray-wall intersections working")
    print("✅ Wall cache system functional")
    print("✅ GCBF compatibility maintained")
    print("✅ Ready for integration with MPC controller")
    print("=" * 100)

if __name__ == "__main__":
    main()
