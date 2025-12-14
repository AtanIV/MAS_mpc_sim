#!/usr/bin/env python3
"""
CBF Field Visualizer with Cost Function Visualization

Creates a comprehensive visualization of the CBF field across the environment space:
1. Creates a mesh grid over the environment
2. At each mesh point, generates a graph with an agent at that position
3. Evaluates CBF value and Jacobian using the trained model
4. Plots CBF contour lines, cost function values, and velocity gradient vector field

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
from typing import Tuple, List, Dict

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcbfplus.env.double_integrator_no_clipping import DoubleIntegratorNoClipping
from gcbfplus.utils.graph import GraphsTuple
from pipeline.graph_evaluator import CBFEvaluator
from pipeline.graph_predictor import MPCGraphPredictor


class CBFFieldVisualizer:
    """
    Visualizes CBF value field, cost function, and velocity gradients across environment space.
    """

    def __init__(self, model_path: str, ego_agent_idx: int = 0,
                 mpc_params: Dict = None):
        """
        Initialize CBF field visualizer.

        Args:
            model_path: Path to trained GCBF model
            ego_agent_idx: Index of ego agent (usually 0)
            mpc_params: Dictionary of MPC parameters for cost function
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

        # MPC cost function parameters
        self.mpc_params = mpc_params or {}
        self.ref_weight = self.mpc_params.get('ref_weight', 1.0)
        self.vel_weight = self.mpc_params.get('vel_weight', 0.0)  # Set to 0
        self.control_weight = self.mpc_params.get('control_weight', 0.0)  # Set to 0
        self.term_pos_w = self.mpc_params.get('term_pos_w', 10.0)
        self.regularization_weight = self.mpc_params.get('regularization_weight', 1e-3)

        # Velocity sigmoid parameters (even though vel_weight=0, keep for completeness)
        self.vel_sigmoid_k = self.mpc_params.get('vel_sigmoid_k', 1.0)
        self.vel_gate_radius = self.mpc_params.get('vel_gate_radius', 0.2)
        self.vel_gate_slope = self.mpc_params.get('vel_gate_slope', 2.0)

        self.dt = self.mpc_params.get('dt', 0.03)
        self.horizon = self.mpc_params.get('horizon', 4)

        # Store ego goal (will be set when creating scenarios)
        self.ego_goal = None

        print("✓ CBF Field Visualizer initialized successfully")

    def _create_field_environment(self):
        """Create controlled environment for field visualization."""
        env_params = {
            "car_radius": 0.05,
            "comm_radius": 0.5,
            "n_rays": 32,
            "obs_len_range": [0.1, 0.5],
            "n_obs": 2,
            "m": 0.1,
        }

        env = DoubleIntegratorNoClipping(
            num_agents=2,
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
        # Four squares scenario
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
        ego_state = jnp.array([ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1]])
        other_state = jnp.array([3.4, 3.4, 0.0, 0.0])
        agent_states = jnp.array([ego_state, other_state])

        # Goals
        goal_states = jnp.array([
            [3.4, 3.4, 0.0, 0.0],  # ego's goal
            [0.6, 0.6, 0.0, 0.0],  # other agent's goal
        ])

        # Store ego goal for cost computation
        self.ego_goal = goal_states[0, :2]

        # Create environment state and graph
        env_state = self.env.EnvState(agent_states, goal_states, obstacles)
        graph = self.env.get_graph(env_state)

        return graph

    def compute_cost_at_point(self, pos: np.ndarray, vel: np.ndarray = np.array([0.0, 0.0])) -> float:
        """
        Compute MPC cost function value at a given position and velocity.
        Assumes zero control sequence (just evaluating current state cost).

        Args:
            pos: Position [x, y]
            vel: Velocity [vx, vy]

        Returns:
            Cost value
        """
        if self.ego_goal is None:
            return 0.0

        # 1) Tracking cost (position error)
        tracking_cost = np.sum((pos - self.ego_goal) ** 2)

        # 2) Terminal position cost (same as tracking in single-point eval)
        term_pos_cost = tracking_cost

        # 3) Velocity cost (near-goal sigmoid weighting)
        d = np.linalg.norm(pos - self.ego_goal)
        r0 = self.vel_gate_radius
        alpha = self.vel_gate_slope
        kmax = self.vel_sigmoid_k

        # Sigmoid: C(d) = k * sigmoid(alpha * (r0 - d))
        sigmoid_val = 1.0 / (1.0 + np.exp(-alpha * (r0 - d)))
        C = kmax * sigmoid_val

        vel_cost = C * np.sum(vel ** 2)

        # Total cost (control cost is 0 since we're not applying controls)
        total_cost = (
                self.ref_weight * tracking_cost +
                self.vel_weight * vel_cost +
                self.term_pos_w * term_pos_cost
        )

        return total_cost

    def evaluate_field_point(self, pos: np.ndarray, vel: np.ndarray = np.array([0.5, 0.5])) -> Tuple[
        float, np.ndarray, float]:
        """
        Evaluate CBF value, Jacobian, and cost at a single field point.

        Args:
            pos: Position [x, y]
            vel: Velocity [vx, vy]

        Returns:
            Tuple of (h_value, jacobian, cost_value)
        """
        try:
            # Create graph for this field point
            graph = self.create_field_scenario(pos, vel)

            # Evaluate CBF and Jacobian
            h_val, jacobian = self.cbf_evaluator.evaluate_h_and_jacobian(graph)

            # Compute cost
            cost_val = self.compute_cost_at_point(pos, vel)

            return h_val, jacobian, cost_val

        except Exception as e:
            print(f"Error at position {pos}: {e}")
            return 0.0, np.zeros(4), 0.0

    def create_field_mesh(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                          resolution: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create mesh grid and evaluate CBF field and cost over the entire space.

        Args:
            x_range: (x_min, x_max) for the field
            y_range: (y_min, y_max) for the field
            resolution: Number of points per dimension

        Returns:
            Tuple of (X, Y, H, Cost, grad_field) where:
            - X, Y: mesh grid coordinates
            - H: CBF values at each mesh point
            - Cost: Cost function values at each mesh point
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
        Cost = np.zeros_like(X)
        grad_field = np.zeros((resolution, resolution, 2))

        # Evaluate field at each mesh point
        total_points = resolution * resolution
        completed = 0

        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                h_val, jacobian, cost_val = self.evaluate_field_point(pos)

                H[i, j] = h_val
                Cost[i, j] = cost_val
                grad_field[i, j, :] = jacobian[2:4]  # Velocity gradient components

                completed += 1
                if completed % 20 == 0:
                    print(f"Progress: {completed}/{total_points} ({100 * completed / total_points:.1f}%)")

        print(f"✓ Field evaluation complete!")
        print(f"H value range: [{np.min(H):.3f}, {np.max(H):.3f}]")
        print(f"Cost value range: [{np.min(Cost):.3f}, {np.max(Cost):.3f}]")

        return X, Y, H, Cost, grad_field

    def plot_cbf_field(self, X, Y, H, Cost, grad_field,
                       save_path: str = None,
                       highlight_points: List[Tuple[float, float]] = None,
                       max_speed: float = 1.0,
                       disk_plot_scale: float = 0.15,
                       disk_normalize: bool = True,
                       # Plot toggles
                       show_cbf_plot: bool = True,
                       show_cost_plot: bool = True,
                       show_gradient_plot: bool = True,
                       show_velocity_disks: bool = True):
        """
        Create comprehensive CBF field visualization.

        Args:
            X, Y: Mesh grid coordinates
            H: CBF values
            Cost: Cost function values
            grad_field: Velocity gradient field
            save_path: Optional path to save the plot
            highlight_points: List of (x, y) positions to highlight with velocity disks
            max_speed: Maximum velocity magnitude for disk visualization
            disk_plot_scale: Visual scale for disks
            disk_normalize: Whether to normalize disk colors
            show_cbf_plot: Toggle CBF contour plot
            show_cost_plot: Toggle cost function plot
            show_gradient_plot: Toggle gradient magnitude plot
            show_velocity_disks: Toggle velocity disk overlays
        """
        print(f"\n=== CREATING CBF FIELD VISUALIZATION ===")

        # Count active plots
        active_plots = sum([show_cbf_plot, show_cost_plot, show_gradient_plot])
        if active_plots == 0:
            print("No plots enabled!")
            return

        fig, axes = plt.subplots(1, active_plots, figsize=(active_plots * 10, 9))

        # Make axes always iterable
        if active_plots == 1:
            axes = [axes]

        plot_idx = 0

        # === PLOT 1: CBF CONTOUR + VELOCITY GRADIENT ARROWS ===
        if show_cbf_plot:
            ax1 = axes[plot_idx]
            plot_idx += 1

            # Plot CBF contour
            safety_levels = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
            contour = ax1.contour(X, Y, H, levels=safety_levels, colors='black', alpha=0.6, linewidths=1)
            contour_filled = ax1.contourf(X, Y, H, levels=safety_levels, alpha=0.3, cmap='RdYlGn')
            ax1.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

            # Add zero-level contour (safety boundary) in bold
            zero_contour = ax1.contour(X, Y, H, levels=[0.0], colors='red', linewidths=3)

            # Plot velocity gradient vectors
            step = max(1, len(X) // 20)
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            grad_field_sub = grad_field[::step, ::step, :]

            # Normalize gradients
            grad_magnitude = np.linalg.norm(grad_field_sub, axis=2)
            mask = grad_magnitude > 1e-6

            U = np.zeros_like(X_sub)
            V = np.zeros_like(Y_sub)
            U[mask] = grad_field_sub[:, :, 0][mask] / grad_magnitude[mask]
            V[mask] = grad_field_sub[:, :, 1][mask] / grad_magnitude[mask]

            colors = grad_magnitude.flatten()

            quiver = ax1.quiver(X_sub, Y_sub, U, V, colors,
                                scale=40, scale_units='width', width=0.002, alpha=0.7, cmap='plasma')

            # Plot obstacles
            self._plot_obstacles(ax1)

            # Overlay velocity disks
            if show_velocity_disks and highlight_points:
                for k, (px, py) in enumerate(highlight_points):
                    self._overlay_velocity_disk(
                        ax1, X, Y, grad_field, px, py,
                        max_speed=max_speed,
                        plot_scale=disk_plot_scale,
                        normalize=disk_normalize,
                        cmap="RdBu_r",
                        alpha=0.85
                    )
                    ax1.text(px, py + max_speed * disk_plot_scale + 0.02, "Velocity-h Map",
                             ha="center", va="bottom",
                             fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9), zorder=8)

            # Formatting
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))
            ax1.set_aspect('equal')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title('CBF Field with Velocity Gradients\n(Arrows point toward safety-increasing velocities)')
            ax1.grid(True, alpha=0.3)

            # Colorbars
            cbar1 = plt.colorbar(contour_filled, ax=ax1, shrink=0.8)
            cbar1.set_label('CBF Value h(x)')

            cbar2 = plt.colorbar(quiver, ax=ax1, shrink=0.6, pad=0.12)
            cbar2.set_label('Gradient Magnitude |∇_v h|')

        # === PLOT 2: COST FUNCTION HEATMAP ===
        if show_cost_plot:
            ax2 = axes[plot_idx]
            plot_idx += 1

            # Plot cost heatmap
            cost_heatmap = ax2.imshow(Cost, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)],
                                      origin='lower', aspect='equal', cmap='viridis', alpha=0.8)

            # Overlay CBF safety boundary for reference
            ax2.contour(X, Y, H, levels=[0.0], colors='red', linewidths=2, linestyles='--', alpha=0.7)

            # Add cost contours
            cost_levels = np.linspace(np.min(Cost), np.max(Cost), 10)
            cost_contour = ax2.contour(X, Y, Cost, levels=cost_levels, colors='white', alpha=0.4, linewidths=0.5)
            ax2.clabel(cost_contour, inline=True, fontsize=7, fmt='%.1f')

            # Plot obstacles
            self._plot_obstacles(ax2)

            # Mark goal position
            if self.ego_goal is not None:
                ax2.plot(self.ego_goal[0], self.ego_goal[1], marker='*', markersize=20,
                         color='yellow', markeredgecolor='black', markeredgewidth=2,
                         label='Goal', zorder=10)
                ax2.legend(loc='upper right')

            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.set_title(f'Cost Function Landscape\n(ref_w={self.ref_weight}, vel_w={self.vel_weight}, ' +
                          f'ctrl_w={self.control_weight}, term_w={self.term_pos_w})')

            # Colorbar
            cbar_cost = plt.colorbar(cost_heatmap, ax=ax2, shrink=0.8)
            cbar_cost.set_label('Cost Value J(x)')

        # === PLOT 3: GRADIENT MAGNITUDE HEATMAP ===
        if show_gradient_plot:
            ax3 = axes[plot_idx]
            plot_idx += 1

            grad_magnitude_full = np.linalg.norm(grad_field, axis=2)

            heatmap = ax3.imshow(grad_magnitude_full, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)],
                                 origin='lower', aspect='equal', cmap='hot')

            # Overlay contour lines
            ax3.contour(X, Y, H, levels=[0.0], colors='cyan', linewidths=2, linestyles='--')
            ax3.contour(X, Y, H, levels=safety_levels, colors='white', alpha=0.4, linewidths=0.5)

            # Plot obstacles
            self._plot_obstacles(ax3)

            ax3.set_xlabel('X Position')
            ax3.set_ylabel('Y Position')
            ax3.set_title('Velocity Gradient Magnitude Field\n(How sensitive CBF is to velocity changes)')

            # Colorbar
            cbar3 = plt.colorbar(heatmap, ax=ax3, shrink=0.8)
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
                 f'Cost includes: tracking + terminal position',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def _plot_obstacles(self, ax):
        """Plot obstacles on the given axes."""
        try:
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

                    center = np.mean(obs_vertices, axis=0)
                    ax.text(center[0], center[1], f'Obs{i + 1}', fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        except Exception as e:
            print(f"Error plotting obstacles: {e}")

    def _overlay_velocity_disk(self,
                               ax: plt.Axes,
                               X: np.ndarray,
                               Y: np.ndarray,
                               grad_field: np.ndarray,
                               x0: float, y0: float,
                               max_speed: float = 0.8,
                               plot_scale: float = 0.15,
                               N: int = 51,  # Reduced for speed
                               normalize: bool = False,
                               cmap: str = "RdBu_r",
                               alpha: float = 0.85):
        """
        Paint a disk at (x0, y0) showing h(x0, y0, v) across velocity space.

        The disk represents velocity space:
        - Distance from center = velocity magnitude
        - Angle = velocity direction
        - Color = CBF value h(x0, y0, v) at that velocity
        """

        # Create velocity grid
        vx_vals = np.linspace(-max_speed, max_speed, N)
        vy_vals = np.linspace(-max_speed, max_speed, N)
        VX, VY = np.meshgrid(vx_vals, vy_vals)

        # Velocity magnitude at each grid point
        v_mag = np.sqrt(VX ** 2 + VY ** 2)

        # Initialize h values
        h_values = np.full_like(VX, np.nan)

        print(f"  Querying {N}x{N} velocities at position ({x0:.2f}, {y0:.2f})...")
        for i in range(N):
            for j in range(N):
                if v_mag[i, j] <= max_speed:
                    vx, vy = VX[i, j], VY[i, j]

                    pos = np.array([x0, y0])
                    vel = np.array([vx, vy])
                    h_val, _, _ = self.evaluate_field_point(pos, vel)
                    h_values[i, j] = h_val

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{N} rows completed")

        # Normalize if requested
        if normalize:
            h_min, h_max = np.nanmin(h_values), np.nanmax(h_values)
            if h_max > h_min:
                h_values = (h_values - h_min) / (h_max - h_min) * 2 - 1

        print(f"    h value range in disk: [{np.nanmin(h_values):.3f}, {np.nanmax(h_values):.3f}]")

        # Plot
        from matplotlib import cm
        cmap_obj = cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)

        vmax = np.nanmax(np.abs(h_values)) or 1.0

        # Convert to position space for plotting
        plot_radius = max_speed * plot_scale
        extent = [x0 - plot_radius, x0 + plot_radius,
                  y0 - plot_radius, y0 + plot_radius]

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

    def analyze_field_statistics(self, X: np.ndarray, Y: np.ndarray, H: np.ndarray,
                                 Cost: np.ndarray, grad_field: np.ndarray):
        """Analyze and print field statistics."""
        print(f"\n{'=' * 60}")
        print("CBF FIELD ANALYSIS")
        print("=" * 60)

        # CBF statistics
        h_min, h_max = np.min(H), np.max(H)
        h_mean, h_std = np.mean(H), np.std(H)
        safe_ratio = np.mean(H > 0)

        print(f"CBF Value Statistics:")
        print(f"  Range: [{h_min:.3f}, {h_max:.3f}]")
        print(f"  Mean: {h_mean:.3f} ± {h_std:.3f}")
        print(f"  Safe region ratio: {safe_ratio:.1%}")

        # Cost statistics
        cost_min, cost_max = np.min(Cost), np.max(Cost)
        cost_mean, cost_std = np.mean(Cost), np.std(Cost)

        print(f"\nCost Function Statistics:")
        print(f"  Range: [{cost_min:.3f}, {cost_max:.3f}]")
        print(f"  Mean: {cost_mean:.3f} ± {cost_std:.3f}")

        # Gradient statistics
        grad_magnitude = np.linalg.norm(grad_field, axis=2)
        grad_min, grad_max = np.min(grad_magnitude), np.max(grad_magnitude)
        grad_mean, grad_std = np.mean(grad_magnitude), np.std(grad_magnitude)

        print(f"\nVelocity Gradient Statistics:")
        print(f"  Magnitude range: [{grad_min:.3f}, {grad_max:.3f}]")
        print(f"  Mean magnitude: {grad_mean:.3f} ± {grad_std:.3f}")

        # Critical points
        safety_boundary_mask = np.abs(H) < 0.05
        if np.any(safety_boundary_mask):
            boundary_grads = grad_magnitude[safety_boundary_mask]
            print(f"  Average gradient at safety boundary: {np.mean(boundary_grads):.3f}")


def main():
    """Main function to create CBF field visualization."""
    print("=" * 80)
    print("CBF FIELD VISUALIZER")
    print("=" * 80)

    # Model path
    # model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulation/logs/DoubleIntegrator/gcbf+/seed0_20250605034319"
    # model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegratorFixedObs/FixedObstacleDoubleIntegrator/gcbf+/seed1234_20251023003513"
    # model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegrator/gcbf+/seed1234_20251104141529"
    # model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegrator/gcbf+/seed1234_20251108035351"
    # model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegrator/gcbf+/seed1234_20251108035351-2000"
    model_path = "/home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/logs/DoubleIntegrator/gcbf+/seed1234_20251130013419"


    # Verify model exists
    if not pathlib.Path(model_path).exists():
        print(f"✗ Model path does not exist: {model_path}")
        print("Please update the model_path variable to point to your trained model")
        return

    try:
        # MPC parameters
        mpc_params = {
            'ref_weight': 10.0,
            'vel_weight': 0.0,  # Set to 0 as requested
            'control_weight': 0.0,  # Set to 0 as requested
            'term_pos_w': 100.0,
            'regularization_weight': 1e-3,
            'vel_sigmoid_k': 1.0,
            'vel_gate_radius': 0.2,
            'vel_gate_slope': 2.0,
            'dt': 0.03,
            'horizon': 4
        }

        # Initialize visualizer
        visualizer = CBFFieldVisualizer(model_path, ego_agent_idx=0, mpc_params=mpc_params)

        # Define field boundaries
        # Define field boundaries
        x_range = (0, 4)
        y_range = (0, 4)
        resolution = 200

        print(f"\nField parameters:")
        print(f"  X range: {x_range}")
        print(f"  Y range: {y_range}")
        print(f"  Resolution: {resolution}x{resolution}")
        print(f"  Total evaluations: {resolution ** 2}")

        # Create field mesh and evaluate CBF + Cost
        X, Y, H, Cost, grad_field = visualizer.create_field_mesh(x_range, y_range, resolution)

        # Analyze field statistics
        visualizer.analyze_field_statistics(X, Y, H, Cost, grad_field)

        # Create visualization with toggles
        visualizer.plot_cbf_field(
            X, Y, H, Cost, grad_field,
            save_path="cbf_field_visualization.png",
            highlight_points=[
                              # (0, 0),
                              (2.5, 2.2)
                              ],  # Optional highlight points
            max_speed=0.5,
            disk_plot_scale=0.3,
            disk_normalize=True,
            # Plot toggles - customize as needed
            show_cbf_plot=True,  # Left plot: CBF contours + gradients
            show_cost_plot=True,  # Middle plot: Cost function heatmap
            show_gradient_plot=True,  # Right plot: Gradient magnitude
            show_velocity_disks=True  # Velocity disks (slow! set False for speed)
        )

        print(f"\n{'=' * 80}")
        print("✓ CBF FIELD VISUALIZATION COMPLETE!")
        print("✓ CBF contour plot shows safety landscape")
        print("✓ Cost plot shows tracking + terminal cost")
        print("✓ Gradient heatmap shows sensitivity")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error creating field visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

