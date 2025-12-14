#!/usr/bin/env python3
"""
CBF Field Visualizer

Creates a comprehensive visualization of the CBF field across the environment space:
1. Creates a mesh grid over the environment
2. At each mesh point, generates a graph with an agent at that position
3. Evaluates CBF value and Jacobian using the trained model
4. Plots CBF contour lines and velocity gradient vector field

This helps understand the safety landscape and optimal velocity directions.
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
from matplotlib.colors import ListedColormap
import pathlib
import sys
from typing import Tuple, List

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.utils.graph import GraphsTuple
from graph_evaluator import CBFEvaluator
from graph_predictor import MPCGraphPredictor


class CBFFieldVisualizer:
    """
    Visualizes CBF value field and velocity gradients across environment space.
    """

    def __init__(self, model_path: str, ego_agent_idx: int = 0):
        """
        Initialize CBF field visualizer.

        Args:
            model_path: Path to trained GCBF model
            ego_agent_idx: Index of ego agent (usually 0)
        """
        self.model_path = pathlib.Path(model_path)
        self.ego_agent_idx = ego_agent_idx

        print(f"=== INITIALIZING CBF FIELD VISUALIZER ===")
        print(f"Model path: {model_path}")

        # Create environment for field visualization
        self.env = self._create_field_environment()

        # Initialize graph predictor and CBF evaluator
        self.graph_predictor = MPCGraphPredictor(self.env)
        self.cbf_evaluator = CBFEvaluator(str(model_path), ego_agent_idx)

        print("✓ CBF Field Visualizer initialized successfully")

    def _create_field_environment(self):
        """Create controlled environment for field visualization."""
        env_params = {
            "car_radius": 0.05,
            "comm_radius": 0.5,
            "n_rays": 32,
            "obs_len_range": [0.1, 0.5],
            "n_obs": 2,  # Fixed number of obstacles
            "m": 0.1,
        }

        env = DoubleIntegrator(
            num_agents=2,  # Ego + one other agent
            area_size=2.0,
            max_step=256,
            max_travel=None,
            dt=0.03,
            params=env_params
        )

        return env

    def create_field_scenario(self, ego_pos: np.ndarray, ego_vel: np.ndarray = np.array([0, 0])):
        """
        Create a scenario with ego agent at specified position/velocity.

        Args:
            ego_pos: Ego agent position [x, y]
            ego_vel: Ego agent velocity [vx, vy]

        Returns:
            GraphsTuple for the scenario
        """

        # # =======================================================
        # # Create fixed obstacles in the environment
        # obs_positions = jnp.array([[2.0, 2.0]])
        # obs_lengths_x = jnp.array([0.8])
        # obs_lengths_y = jnp.array([0.8])
        # obs_thetas = jnp.array([0.0])
        # obstacles = self.env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
        #
        # # Create agent states: ego at specified position, other agent at fixed location
        # ego_state = jnp.array([ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1]])
        # other_state = jnp.array([-2, -2, 0.0, 0.0])  # Fixed other agent
        # agent_states = jnp.array([ego_state, other_state])
        #
        # # Create goal states
        # goal_states = jnp.array([
        #     [0.6, 0.5, 0.0, 0.0],  # Ego goal
        #     [3.4, 3.5, 0.0, 0.0]  # Other agent goal
        # ])
        # # =======================================================

        # # ======================Leaning square=================================
        # # Create obstacles
        # obs_positions = jnp.array([[1.0, 1.0]])
        # obs_lengths_x = jnp.array([0.8])
        # obs_lengths_y = jnp.array([0.8])
        # obs_thetas = jnp.array([-jnp.pi / 32])
        #
        # obstacles = self.env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
        #
        # # Create agents
        # # ego_state = jnp.array([0.9, 0.95, -0.4, 0.5])
        # # ego_state = jnp.array([1, 1.05, 0.5, 0.12]) # 11 step safe discrete
        # ego_state = jnp.array([ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1]])
        # other_state = jnp.array([0.3, 0.25, 0, 0])
        # agent_states = jnp.array([ego_state, other_state])
        #
        # # Goals
        # goal_states = jnp.array([
        #     [1.8, 1.8, 0.0, 0.0],
        #     [1.5, 1.0, 0.0, 0.0]
        # ])
        # # =======================================================

        # =========================Four squares==============================
        # Create obstacles (same four 0.3×0.3 blocks as before)
        obs_positions = jnp.array([
            [1.5, 1.5],
            [1.5, 2.5],
            [2.5, 1.5],
            [2.5, 2.5],
        ])
        obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
        obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
        obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])

        obstacles = self.env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)

        # Agents
        # Ego uses provided ego_pos/ego_vel (intended to mirror agent_0's start [0.6, 0.6, 0, 0])
        ego_state = jnp.array([ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1]])
        # Other agent starts at its original starting point [3.4, 3.4, 0, 0]
        other_state = jnp.array([3.4, 3.4, 0.0, 0.0])
        agent_states = jnp.array([ego_state, other_state])

        # Goals (agent 0 → [3.4, 3.4], agent 3 → [0.6, 0.6])
        goal_states = jnp.array([
            [3.4, 3.4, 0.0, 0.0],  # ego's goal (was goal_0_state)
            [0.6, 0.6, 0.0, 0.0],  # other agent's goal (was goal_3_state)
        ])
        # =======================================================

        # # =======================================================
        # # Create fixed obstacles in the environment
        # obs_positions = jnp.array([[1.0, 1.2], [0.6, 1.2]])
        # obs_lengths_x = jnp.array([0.6, 0.1])
        # obs_lengths_y = jnp.array([0.1, 0.4])
        # obs_thetas = jnp.array([0.0, -jnp.pi / 8])
        # obstacles = self.env.create_obstacles(obs_positions, obs_lengths_x, obs_lengths_y, obs_thetas)
        #
        # # Create agent states: ego at specified position, other agent at fixed location
        # ego_state = jnp.array([ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1]])
        # other_state = jnp.array([0.2, 0.2, 0.0, 0.0])  # Fixed other agent
        # agent_states = jnp.array([ego_state, other_state])
        #
        # # Create goal states
        # goal_states = jnp.array([
        #     [1.8, 1.5, 0.0, 0.0],  # Ego goal
        #     [1.5, 1.5, 0.0, 0.0]  # Other agent goal
        # ])
        # # =======================================================


        # Create environment state and graph
        env_state = self.env.EnvState(agent_states, goal_states, obstacles)
        graph = self.env.get_graph(env_state)

        return graph

    def evaluate_field_point(self, pos: np.ndarray, vel: np.ndarray = np.array([0.4, 0.4])) -> Tuple[float, np.ndarray]:
        """
        Evaluate CBF value and Jacobian at a single field point.

        Args:
            pos: Position [x, y]
            vel: Velocity [vx, vy]

        Returns:
            Tuple of (h_value, jacobian)
        """
        try:
            # Create graph for this field point
            graph = self.create_field_scenario(pos, vel)

            # Evaluate CBF and Jacobian
            h_val, jacobian = self.cbf_evaluator.evaluate_h_and_jacobian(graph)

            return h_val, jacobian

        except Exception as e:
            print(f"Error at position {pos}: {e}")
            return 0.0, np.zeros(4)

    def create_field_mesh(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                          resolution: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create mesh grid and evaluate CBF field over the entire space.

        Args:
            x_range: (x_min, x_max) for the field
            y_range: (y_min, y_max) for the field
            resolution: Number of points per dimension

        Returns:
            Tuple of (X, Y, H, grad_field) where:
            - X, Y: mesh grid coordinates
            - H: CBF values at each mesh point
            - grad_field: velocity gradients [resolution, resolution, 2]
        """
        print(f"\n=== CREATING CBF FIELD MESH ===")
        print(f"X range: {x_range}, Y range: {y_range}")
        print(f"Resolution: {resolution}x{resolution} = {resolution ** 2} points")

        # Create mesh grid
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Initialize result arrays
        H = np.zeros_like(X)
        grad_field = np.zeros((resolution, resolution, 2))

        # Evaluate field at each mesh point
        total_points = resolution * resolution
        completed = 0

        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                h_val, jacobian = self.evaluate_field_point(pos)

                H[i, j] = h_val
                grad_field[i, j, :] = jacobian[2:4]  # Velocity gradient components

                completed += 1
                if completed % 20 == 0:
                    print(f"Progress: {completed}/{total_points} ({100 * completed / total_points:.1f}%)")

        print(f"✓ Field evaluation complete!")
        print(f"H value range: [{np.min(H):.3f}, {np.max(H):.3f}]")

        return X, Y, H, grad_field

    def plot_cbf_field(self, X, Y, H, grad_field,
                       save_path: str = None,
                       highlight_points: List[Tuple[float, float]] = None,
                       max_speed: float = 1.0,  # Changed from disk_radius
                       disk_normalize: bool = True,
                       use_exact_disks: bool = True):  # Can remove this now

        """
        Create comprehensive CBF field visualization.

        Args:
            X, Y: Mesh grid coordinates
            H: CBF values
            grad_field: Velocity gradient field
            save_path: Optional path to save the plot
        """
        print(f"\n=== CREATING CBF FIELD VISUALIZATION ===")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # === LEFT PLOT: CBF CONTOUR + VELOCITY GRADIENT ARROWS ===

        # Plot CBF contour
        safety_levels = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
        contour = ax1.contour(X, Y, H, levels=safety_levels, colors='black', alpha=0.6, linewidths=1)
        contour_filled = ax1.contourf(X, Y, H, levels=safety_levels, alpha=0.3, cmap='RdYlGn')
        ax1.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

        # Add zero-level contour (safety boundary) in bold
        zero_contour = ax1.contour(X, Y, H, levels=[0.0], colors='red', linewidths=3)

        # Plot velocity gradient vectors (subsampled for clarity)
        step = max(1, len(X) // 20)  # Adjust arrow density
        X_sub = X[::step, ::step]
        Y_sub = Y[::step, ::step]
        grad_field_sub = grad_field[::step, ::step, :]

        # Normalize gradients for better visualization
        grad_magnitude = np.linalg.norm(grad_field_sub, axis=2)
        mask = grad_magnitude > 1e-6

        # Create normalized arrows
        U = np.zeros_like(X_sub)
        V = np.zeros_like(Y_sub)
        U[mask] = grad_field_sub[:, :, 0][mask] / grad_magnitude[mask]
        V[mask] = grad_field_sub[:, :, 1][mask] / grad_magnitude[mask]

        # Color arrows by gradient magnitude
        colors = grad_magnitude.flatten()

        quiver = ax1.quiver(X_sub, Y_sub, U, V, colors,
                            scale=40, scale_units='width', width=0.002, alpha=0.7, cmap='plasma')

        # Plot obstacles
        self._plot_obstacles(ax1)

        # === Highlight requested points with directional-derivative disks ===
        if highlight_points:
            for k, (px, py) in enumerate(highlight_points):
                self._overlay_directional_disk(
                    ax1, X, Y, grad_field, px, py,
                    max_speed=max_speed,
                    normalize=disk_normalize,
                    cmap="RdBu_r",
                    alpha=0.85
                )
                ax1.text(px, py + max_speed * 0.15 + 0.02, "Velocity-h Map",
                         ha="center", va="bottom",
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                         zorder=8)

        # Formatting
        ax1.set_xlim(np.min(X), np.max(X))
        ax1.set_ylim(np.min(Y), np.max(Y))
        ax1.set_aspect('equal')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('CBF Field with Velocity Gradients\n(Arrows point toward safety-increasing velocities)')
        ax1.grid(True, alpha=0.3)

        # Add colorbar for CBF values
        cbar1 = plt.colorbar(contour_filled, ax=ax1, shrink=0.8)
        cbar1.set_label('CBF Value h(x)')

        # Add colorbar for gradient magnitudes
        cbar2 = plt.colorbar(quiver, ax=ax1, shrink=0.6, pad=0.12)
        cbar2.set_label('Gradient Magnitude |∇_v h|')

        # === RIGHT PLOT: GRADIENT MAGNITUDE HEATMAP ===

        grad_magnitude_full = np.linalg.norm(grad_field, axis=2)

        heatmap = ax2.imshow(grad_magnitude_full, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)],
                             origin='lower', aspect='equal', cmap='hot')

        # Overlay contour lines
        ax2.contour(X, Y, H, levels=[0.0], colors='cyan', linewidths=2, linestyles='--')
        ax2.contour(X, Y, H, levels=safety_levels, colors='white', alpha=0.4, linewidths=0.5)

        # Plot obstacles
        self._plot_obstacles(ax2)

        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Velocity Gradient Magnitude Field\n(How sensitive CBF is to velocity changes)')

        # Add colorbar for gradient magnitude
        cbar3 = plt.colorbar(heatmap, ax=ax2, shrink=0.8)
        cbar3.set_label('|∇_v h|')

        # === OVERALL FORMATTING ===

        plt.suptitle(f'CBF Field Analysis\nModel: {self.model_path.name}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Add text annotations
        fig.text(0.02, 0.02,
                 f'Red contour: Safety boundary (h=0)\n'
                 f'Green regions: Safe (h>0)\n'
                 f'Red regions: Unsafe (h<0)\n'
                 f'Arrows: Velocity directions that increase safety',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def _plot_obstacles(self, ax):
        """Plot obstacles on the given axes."""
        try:
            # Create a dummy graph to get obstacle information
            dummy_graph = self.create_field_scenario(np.array([1.0, 1.0]))
            obstacles = dummy_graph.env_states.obstacle

            if len(obstacles) >= 6:
                vertices = obstacles[5]
                n_obstacles = vertices.shape[0]

                for i in range(n_obstacles):
                    obs_vertices = vertices[i]
                    polygon = patches.Polygon(
                        obs_vertices,
                        linewidth=2,
                        edgecolor='black',
                        facecolor='gray',
                        alpha=0.8,
                        closed=True
                    )
                    ax.add_patch(polygon)

                    # Add obstacle label
                    center = np.mean(obs_vertices, axis=0)
                    ax.text(center[0], center[1], f'Obs{i + 1}', fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        except Exception as e:
            print(f"Error plotting obstacles: {e}")

    def analyze_field_statistics(self, X: np.ndarray, Y: np.ndarray, H: np.ndarray, grad_field: np.ndarray):
        """Analyze and print field statistics."""
        print(f"\n{'=' * 60}")
        print("CBF FIELD ANALYSIS")
        print("=" * 60)

        # Basic statistics
        h_min, h_max = np.min(H), np.max(H)
        h_mean, h_std = np.mean(H), np.std(H)
        safe_ratio = np.mean(H > 0)

        print(f"CBF Value Statistics:")
        print(f"  Range: [{h_min:.3f}, {h_max:.3f}]")
        print(f"  Mean: {h_mean:.3f} ± {h_std:.3f}")
        print(f"  Safe region ratio: {safe_ratio:.1%}")

        # Gradient statistics
        grad_magnitude = np.linalg.norm(grad_field, axis=2)
        grad_min, grad_max = np.min(grad_magnitude), np.max(grad_magnitude)
        grad_mean, grad_std = np.mean(grad_magnitude), np.std(grad_magnitude)

        print(f"\nVelocity Gradient Statistics:")
        print(f"  Magnitude range: [{grad_min:.3f}, {grad_max:.3f}]")
        print(f"  Mean magnitude: {grad_mean:.3f} ± {grad_std:.3f}")

        # Find critical points
        safety_boundary_mask = np.abs(H) < 0.05
        if np.any(safety_boundary_mask):
            boundary_grads = grad_magnitude[safety_boundary_mask]
            print(f"  Average gradient at safety boundary: {np.mean(boundary_grads):.3f}")

    def _nearest_grid_index(self, X: np.ndarray, Y: np.ndarray, x0: float, y0: float) -> Tuple[int, int]:
        """Return (i, j) index of the nearest grid node to (x0, y0)."""
        # Assumes rectilinear mesh (your case)
        i = np.argmin(np.abs(Y[:, 0] - y0))
        j = np.argmin(np.abs(X[0, :] - x0))
        return int(i), int(j)

    def _grad_v_at_point(self, pos_xy: Tuple[float, float], vel_vxvy: Tuple[float, float] = (0.0, 0.0)) -> Tuple[
        float, float]:
        """
        Return grad_v h = (∂h/∂vx, ∂h/∂vy) at an exact (x,y) and velocity (vx,vy),
        by calling the model once at that state.
        """
        pos = np.array([pos_xy[0], pos_xy[1]])
        vel = np.array([vel_vxvy[0], vel_vxvy[1]])
        _, jac = self.evaluate_field_point(pos, vel)  # jac is ∂h/∂[x,y,vx,vy]
        gx, gy = float(jac[2]), float(jac[3])  # grad wrt velocity components
        return gx, gy

    def _overlay_directional_disk(self,
                                  ax: plt.Axes,
                                  X: np.ndarray,
                                  Y: np.ndarray,
                                  grad_field: np.ndarray,
                                  x0: float, y0: float,
                                  max_speed: float = 0.8,  # Max velocity to visualize
                                  N: int = 40,
                                  normalize: bool = False,
                                  cmap: str = "RdBu_r",
                                  alpha: float = 0.85):
        """
        Paint a disk at (x0, y0) showing h(x0, y0, v) across velocity space.

        Args:
            x0, y0: Position where we're evaluating
            max_speed: Maximum velocity magnitude (disk radius in velocity space)
            N: Resolution of the disk
            normalize: If True, normalize h values to [-1, 1]

        The disk represents velocity space:
        - Distance from center = velocity magnitude
        - Angle = velocity direction
        - Color = CBF value h(x0, y0, v) at that velocity
        """

        # Create velocity grid (in velocity space, not position space!)
        vx_vals = np.linspace(-max_speed, max_speed, N)
        vy_vals = np.linspace(-max_speed, max_speed, N)
        VX, VY = np.meshgrid(vx_vals, vy_vals)

        # Velocity magnitude at each grid point
        v_mag = np.sqrt(VX ** 2 + VY ** 2)

        # Initialize h values
        h_values = np.full_like(VX, np.nan)

        # Query model at each velocity
        print(f"  Querying {N}x{N} velocities at position ({x0:.2f}, {y0:.2f})...")
        for i in range(N):
            for j in range(N):
                if v_mag[i, j] <= max_speed:  # Only inside the disk
                    vx, vy = VX[i, j], VY[i, j]

                    # Query model at fixed position (x0, y0) with varying velocity
                    pos = np.array([x0, y0])
                    vel = np.array([vx, vy])
                    h_val, _ = self.evaluate_field_point(pos, vel)
                    h_values[i, j] = h_val

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{N} rows completed")

        # Normalize if requested
        if normalize:
            h_min, h_max = np.nanmin(h_values), np.nanmax(h_values)
            if h_max > h_min:
                h_values = (h_values - h_min) / (h_max - h_min) * 2 - 1  # Map to [-1, 1]

        print(f"    h value range in disk: [{np.nanmin(h_values):.3f}, {np.nanmax(h_values):.3f}]")

        # Plot the velocity disk
        from matplotlib import cm
        cmap_obj = cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)  # Transparent outside disk

        vmax = np.nanmax(np.abs(h_values)) or 1.0

        # Convert to position space for plotting (scale by some factor for visibility)
        # This scaling is just for visualization - doesn't affect the data
        plot_scale = 0.15  # Adjust this to change disk size on plot
        extent = [x0 - max_speed * plot_scale, x0 + max_speed * plot_scale,
                  y0 - max_speed * plot_scale, y0 + max_speed * plot_scale]

        ax.imshow(
            h_values,
            origin="lower",
            extent=extent,
            cmap=cmap_obj,
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
            alpha=alpha,
            zorder=5,
        )

        # Add circle border
        plot_radius = max_speed * plot_scale
        circ = patches.Circle((x0, y0), radius=plot_radius, fill=False,
                              linewidth=1.8, edgecolor="k", zorder=6)
        ax.add_patch(circ)

        # Mark center point
        ax.plot([x0], [y0], marker="o", markersize=3.5, color="k", zorder=7)

        # Add velocity axes for reference
        ax.plot([x0 - plot_radius, x0 + plot_radius], [y0, y0],
                'k--', alpha=0.3, linewidth=0.8, zorder=6)
        ax.plot([x0, x0], [y0 - plot_radius, y0 + plot_radius],
                'k--', alpha=0.3, linewidth=0.8, zorder=6)


def main():
    """Main function to create CBF field visualization."""
    print("=" * 80)
    print("CBF FIELD VISUALIZER")
    print("=" * 80)

    # Model path - update this to your trained model
    # model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorMPC/gcbf+/seed0_20250626023916"
    model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulation/logs/DoubleIntegrator/gcbf+/seed0_20250605034319"

    # Verify model exists
    if not pathlib.Path(model_path).exists():
        print(f"✗ Model path does not exist: {model_path}")
        print("Please update the model_path variable to point to your trained model")
        return

    try:
        # Initialize visualizer
        visualizer = CBFFieldVisualizer(model_path, ego_agent_idx=0)

        # Define field boundaries (adjust based on your environment)
        # x_range = (1.4, 1.7)  # Avoid exact boundaries to prevent edge cases
        # y_range = (2.3, 2.6)

        x_range = (0, 4)  # Avoid exact boundaries to prevent edge cases
        y_range = (0, 4)

        resolution = 40  # 25x25 = 625 points (adjust based on computational capacity)

        print(f"\nField parameters:")
        print(f"  X range: {x_range}")
        print(f"  Y range: {y_range}")
        print(f"  Resolution: {resolution}x{resolution}")
        print(f"  Total evaluations: {resolution ** 2}")

        # Create field mesh and evaluate CBF
        X, Y, H, grad_field = visualizer.create_field_mesh(x_range, y_range, resolution)

        # Analyze field statistics
        visualizer.analyze_field_statistics(X, Y, H, grad_field)

        # Create visualization
        visualizer.plot_cbf_field(
            X, Y, H, grad_field,
            save_path="cbf_field_visualization.png",
            highlight_points=[(0.8, 2), (0.6, 3)],
            max_speed=1.0,  # Max velocity to visualize
            disk_normalize=True,  # Normalize colors
        )

        print(f"\n{'=' * 80}")
        print("✓ CBF FIELD VISUALIZATION COMPLETE!")
        print("✓ Contour plot shows safety landscape")
        print("✓ Vector field shows optimal velocity directions")
        print("✓ Heatmap shows gradient sensitivity")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error creating field visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()