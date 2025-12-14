"""Double integrator environment with fixed rectangular obstacles."""

import jax
import jax.numpy as jnp
import jax.random as jr
import functools as ft
from typing import Tuple

from .double_integrator import DoubleIntegrator
from .obstacle import Rectangle
from .utils import get_node_goal_rng
from ..utils.utils import jax_vmap


class FixedObstacleDoubleIntegrator(DoubleIntegrator):
    """
    Double integrator with pre-defined fixed obstacles.
    Inherits all dynamics and graph construction from DoubleIntegrator.
    """

    # Override PARAMS to set n_obs to 0 (we'll use fixed obstacles instead)
    PARAMS = {
        **DoubleIntegrator.PARAMS,
        "n_obs": 0,  # No random obstacles
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params)

        # Define fixed obstacle configuration
        self.fixed_obs_positions = jnp.array([
            [1.5, 1.5],
            [1.5, 2.5],
            [2.5, 1.5],
            [2.5, 2.5]
        ])
        self.fixed_obs_lengths_x = jnp.array([0.3, 0.3, 0.3, 0.3])
        self.fixed_obs_lengths_y = jnp.array([0.3, 0.3, 0.3, 0.3])
        self.fixed_obs_thetas = jnp.array([0.0, 0.0, 0.0, 0.0])
        self.num_fixed_obs = len(self.fixed_obs_positions)

        print(f"\n{'=' * 60}")
        print(f"FixedObstacleDoubleIntegrator Initialized")
        print(f"{'=' * 60}")
        print(f"  Agents: {num_agents}")
        print(f"  Area: {area_size}x{area_size}")
        print(f"  Fixed obstacles: {self.num_fixed_obs}")
        print(f"  Obstacle positions:\n{self.fixed_obs_positions}")
        print(f"  Obstacle sizes: {self.fixed_obs_lengths_x[0]}x{self.fixed_obs_lengths_y[0]}")
        print(f"{'=' * 60}\n")

    def reset(self, key: jax.random.PRNGKey):
        """Reset environment with fixed obstacles."""
        self._t = 0

        # Create fixed obstacles (no random generation)
        obstacles = self.create_obstacles(
            self.fixed_obs_positions,
            self.fixed_obs_lengths_x,
            self.fixed_obs_lengths_y,
            self.fixed_obs_thetas
        )

        # Randomly generate agent and goal positions (avoiding obstacles)
        states, goals = get_node_goal_rng(
            key,
            self.area_size,
            2,  # 2D environment
            obstacles,
            self.num_agents,
            4 * self.params["car_radius"],  # minimum distance
            self.max_travel
        )

        # Add zero velocity to states and goals
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)

        # Create environment state
        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)
