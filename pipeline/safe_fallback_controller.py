"""
Safe fallback controller using QP with CBF constraints.
Mirrors GCBF+ QP behavior for single-agent control with local subgraphs.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.env.base import GraphsTuple


class FallbackConfig:
    """Configuration for fallback QP controller."""

    def __init__(
            self,
            alpha: float = 1.0,
            alpha_scale: float = 0.01,  # GCBF+ uses 0.1 * alpha
            margin: float = 0.0,
            relax_penalty: float = 1e3,
            relax_quad_weight: float = 10.0,
            qp_max_iter: int = 1000,
            verbose: bool = True,
            vel_relax_penalty: float = 200.0,
            vel_relax_quad_weight: float = 50.0,
    ):
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.margin = margin
        self.relax_penalty = relax_penalty
        self.relax_quad_weight = relax_quad_weight
        self.qp_max_iter = qp_max_iter
        self.verbose = verbose
        self.vel_relax_penalty = vel_relax_penalty
        self.vel_relax_quad_weight = vel_relax_quad_weight


class SafeFallbackController:
    """
    QP-based fallback controller using trained CBF model.

    Designed for decentralized control with local subgraphs:
    - Ego agent is always index 0 in local graph
    - May include multiple agents (ego + neighbors in comm range)
    - Controls only ego agent using CBF safety constraints
    """

    def __init__(
            self,
            env,
            mass: float = 0.1,
            dt: float = 0.03,
            cfg: FallbackConfig = None,
            dtype=jnp.float64,
    ):
        """
        Initialize fallback controller.

        Args:
            env: Environment instance (for dynamics and action limits)
            mass: Agent mass (for control-affine dynamics)
            cfg: Fallback configuration
            dtype: JAX dtype for numerical computations
        """
        self.env = env
        self.mass = mass
        self.dt = dt
        self.cfg = cfg if cfg is not None else FallbackConfig()
        self._dtype = dtype

        # Cache action limits
        u_lb, u_ub = self.env.action_lim()
        self._u_lb = jnp.asarray(u_lb, dtype=self._dtype)
        self._u_ub = jnp.asarray(u_ub, dtype=self._dtype)

        # QP solver settings
        self._qp_settings = JaxProxQP.Settings.default()
        self._qp_settings.max_iter = self.cfg.qp_max_iter
        self._qp_settings.verbose = self.cfg.verbose

        # Cache for diagnostics
        self._current_local_graph = None
        self._last_qp_sol = None

    def act_ego(
            self,
            local_graph: GraphsTuple,
            cbf_ego: Callable[[GraphsTuple], jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute safe control action for ego agent using QP.

        Args:
            local_graph: Local subgraph (ego at index 0, may include neighbors)
            cbf_ego: Function to evaluate CBF on graph (returns scalar)

        Returns:
            u_opt: Optimal control (force) for ego agent, shape (2,)
        """
        # Cache for diagnostics
        self._current_local_graph = local_graph

        # Get CBF value and gradient
        h, hx = self._cbf_value_and_grad(local_graph, cbf_ego)
        h = h - 0.2

        # Compute Lie derivatives for CBF constraint
        Lf_h, Lg_h_force = self._lie_derivatives(local_graph, hx)

        # Get nominal reference control
        u_ref_force = self._get_u_ref(local_graph)

        # ego state/vel from local graph
        local_agent_mask = local_graph.node_type == 0
        n_local_agents = int(jnp.sum(local_agent_mask))
        ego_state = local_graph.type_states(type_idx=0, n_type=n_local_agents)[0]
        ego_vel = ego_state[2:]  # (2,)

        # velocity limits from env.state_lim()
        lb, ub = self.env.state_lim()
        v_min = jnp.asarray(lb[2:4], dtype=self._dtype)
        v_max = jnp.asarray(ub[2:4], dtype=self._dtype)

        # velocity limits (override for testing)
        # v_min = jnp.array([-0.5, -0.5], dtype=self._dtype)
        # v_max = jnp.array([0.5, 0.5], dtype=self._dtype)

        # Build and solve QP
        alpha_eff = self.cfg.alpha * self.cfg.alpha_scale
        H, g, C, b, l_box, u_box = self._build_qp_data_continuous(
            u_ref_force, Lf_h, Lg_h_force, h, alpha_eff,
            ego_vel=ego_vel, v_min=v_min, v_max=v_max
        )

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        solver = JaxProxQP(qp, self._qp_settings)
        sol = solver.solve()

        # Cache solution for diagnostics
        self._last_qp_sol = sol

        # Extract control (first 2 components)
        u_opt = sol.x[:2]

        return u_opt

    def _cbf_value_and_grad(
            self,
            local_graph: GraphsTuple,
            cbf_ego: Callable[[GraphsTuple], jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute CBF value and Jacobian for ego agent.

        Works with variable-size local subgraphs:
        - 1 agent (ego only)
        - Multiple agents (ego + neighbors in comm range)

        Args:
            local_graph: Local subgraph with ego at index 0
            cbf_ego: CBF evaluation function

        Returns:
            h: CBF value (scalar)
            hx: Jacobian ∂h/∂x_ego, shape (4,)
        """
        # Get LOCAL agent count (variable across subgraphs!)
        local_agent_mask = local_graph.node_type == 0
        n_local_agents = int(jnp.sum(local_agent_mask))

        # Extract all agent states in local graph
        agent_states = local_graph.type_states(type_idx=0, n_type=n_local_agents)
        ego_state = agent_states[0]  # Ego always first

        def h_of_state(x_ego: jnp.ndarray) -> jnp.ndarray:
            """
            Wrapper for autodiff: h as function of ego state.

            Note: h depends on ALL agents in local graph (via GNN), but we only
            perturb ego state to compute ∂h/∂x_ego (correct for decentralized control).
            """
            # Get LOCAL counts from graph structure
            local_agent_mask = local_graph.node_type == 0
            local_goal_mask = local_graph.node_type == 1
            local_lidar_mask = local_graph.node_type == 2

            n_agents_local = int(jnp.sum(local_agent_mask))
            n_goals_local = int(jnp.sum(local_goal_mask))
            n_lidar_local = int(jnp.sum(local_lidar_mask))

            # Extract states with CORRECT local dimensions
            agents = local_graph.type_states(type_idx=0, n_type=n_agents_local)
            goals = local_graph.type_states(type_idx=1, n_type=n_goals_local)
            lidar = local_graph.type_states(type_idx=2, n_type=n_lidar_local)

            # Update only ego state (index 0), keep neighbors unchanged
            agents_new = agents.at[0].set(x_ego)

            # Concatenate in correct order: agents, goals, lidar
            new_states = jnp.concatenate([agents_new, goals, lidar], axis=0)

            # Rebuild graph with updated states and edge features
            new_graph = self.env.add_edge_feats(local_graph, new_states)

            # Evaluate CBF
            h_val = cbf_ego(new_graph)
            return jnp.squeeze(h_val)

        # Evaluate h and its Jacobian w.r.t. ego state
        h_val = h_of_state(ego_state)
        grad_fn = jax.jacobian(h_of_state)
        hx = grad_fn(ego_state)  # Shape: (4,) = [∂h/∂x, ∂h/∂y, ∂h/∂vx, ∂h/∂vy]

        return h_val, hx

    def _lie_derivatives(
            self,
            local_graph: GraphsTuple,
            hx: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Lie derivatives for CBF constraint.

        For double integrator with control input as force:
            ẋ = v
            v̇ = u/m

        CBF derivative:
            ḣ = ∇h_pos · v + ∇h_vel · (u/m)
              = Lf_h + Lg_h · u

        Args:
            local_graph: Local subgraph
            hx: CBF Jacobian, shape (4,) = [∂h/∂x, ∂h/∂y, ∂h/∂vx, ∂h/∂vy]

        Returns:
            Lf_h: Drift term (scalar)
            Lg_h_force: Control Jacobian w.r.t. force input, shape (2,)
        """
        # Get LOCAL agent count
        local_agent_mask = local_graph.node_type == 0
        n_local_agents = int(jnp.sum(local_agent_mask))

        # Extract ego state
        agent_states = local_graph.type_states(type_idx=0, n_type=n_local_agents)
        ego_state = agent_states[0]  # [x, y, vx, vy]

        # Split Jacobian
        hx_pos = hx[:2]  # ∂h/∂pos
        hx_vel = hx[2:]  # ∂h/∂vel

        # Drift term: ∇h_pos · v
        ego_vel = ego_state[2:]
        Lf_h = jnp.dot(hx_pos, ego_vel)

        # Control term: ∇h_vel · (u/m)
        # Since u is force, acceleration = u/m
        Lg_h_force = hx_vel / self.mass

        return Lf_h, Lg_h_force

    def _get_u_ref(self, local_graph: GraphsTuple) -> jnp.ndarray:
        """
        Get nominal reference control for ego agent.

        Uses simple PD control toward goal.

        Args:
            local_graph: Local subgraph

        Returns:
            u_ref: Reference control (force), shape (2,)
        """
        # Get LOCAL counts
        local_agent_mask = local_graph.node_type == 0
        local_goal_mask = local_graph.node_type == 1

        n_local_agents = int(jnp.sum(local_agent_mask))
        n_local_goals = int(jnp.sum(local_goal_mask))

        # Extract states
        agents = local_graph.type_states(type_idx=0, n_type=n_local_agents)
        goals = local_graph.type_states(type_idx=1, n_type=n_local_goals)

        # Ego is always first
        ego_agent = agents[0]
        ego_goal = goals[0]

        # PD control to goal
        k_p = 1.0
        k_d = 2.0
        pos_error = ego_goal[:2] - ego_agent[:2]
        vel = ego_agent[2:]
        u_ref_force = k_p * pos_error - k_d * vel

        # Clip to action limits
        u_ref_clipped = jnp.clip(u_ref_force, self._u_lb, self._u_ub)

        return jnp.asarray(u_ref_clipped, dtype=self._dtype)

    def _build_qp_data_continuous(
            self,
            u_ref_force: jnp.ndarray,
            Lf_h: jnp.ndarray,
            Lg_h_force: jnp.ndarray,
            h: jnp.ndarray,
            alpha_eff: float,
            ego_vel,
            v_min,
            v_max,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build QP data matching GCBF+ formulation exactly.

        QP problem:
            min  0.5 ||u - u_ref||^2 + 0.5 * relax_quad_weight * r^2 + relax_penalty * r
            s.t. -Lg_h·u - r ≤ Lf_h + α_eff·h + margin
                 u_min ≤ u ≤ u_max
                 0 ≤ r ≤ ∞

        Decision variable: z = [u_x, u_y, r]^T ∈ ℝ³

        Standard form:
            min  0.5 z^T H z + g^T z
            s.t. C z ≤ b
                 l_box ≤ z ≤ u_box

        Args:
            u_ref_force: Reference control (force), shape (2,)
            Lf_h: Drift term (scalar)
            Lg_h_force: Control Jacobian w.r.t. force, shape (2,)
            h: CBF value (scalar)
            alpha_eff: Effective alpha (typically 0.1 * alpha)

        Returns:
            H: Hessian matrix, shape (3, 3)
            g: Linear cost vector, shape (3,)
            C: Constraint matrix, shape (1, 3)
            b: Constraint bound, shape (1,)
            l_box: Lower box constraint, shape (3,)
            u_box: Upper box constraint, shape (3,)
        """
        DT = self._dtype
        k = jnp.array(self.dt / self.mass, dtype=DT)

        # z = [ux, uy, r, spx, spy, smx, smy]
        H = jnp.diag(jnp.array([
            1.0, 1.0, self.cfg.relax_quad_weight,
            self.cfg.vel_relax_quad_weight, self.cfg.vel_relax_quad_weight,
            self.cfg.vel_relax_quad_weight, self.cfg.vel_relax_quad_weight
        ], dtype=DT))

        g = jnp.concatenate([
            -u_ref_force.astype(DT),
            jnp.array([self.cfg.relax_penalty], dtype=DT),
            jnp.ones((4,), dtype=DT) * self.cfg.vel_relax_penalty
        ])

        # --- CBF inequality (1 row), pad zeros for new slack vars
        C_cbf = jnp.concatenate([
            -Lg_h_force[None, :].astype(DT),  # (1,2) on u
            -jnp.ones((1, 1), dtype=DT),  # (1,1) on r
            jnp.zeros((1, 4), dtype=DT),  # (1,4) on vel slacks
        ], axis=1)

        b_cbf = jnp.array([float(Lf_h) + alpha_eff * float(h) + self.cfg.margin], dtype=DT)

        # --- Velocity soft constraints (4 rows)
        vx, vy = ego_vel.astype(DT)
        rhs_up = (v_max.astype(DT) - ego_vel.astype(DT))  # (2,)
        rhs_low = (ego_vel.astype(DT) - v_min.astype(DT))  # (2,)

        # Rows: k*ux - spx <= rhs_up[0]
        #       k*uy - spy <= rhs_up[1]
        C_v_up = jnp.array([
            [k, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, k, 0.0, 0.0, -1.0, 0.0, 0.0],
        ], dtype=DT)
        b_v_up = rhs_up

        # Rows: -k*ux - smx <= rhs_low[0]
        #       -k*uy - smy <= rhs_low[1]
        C_v_low = jnp.array([
            [-k, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, -k, 0.0, 0.0, 0.0, 0.0, -1.0],
        ], dtype=DT)
        b_v_low = rhs_low

        # Stack all inequalities
        C = jnp.concatenate([C_cbf, C_v_up, C_v_low], axis=0)  # (1+2+2, 7)
        b = jnp.concatenate([b_cbf, b_v_up, b_v_low], axis=0)  # (5,)

        # Box bounds: u bounds, r>=0, slacks>=0
        l_box = jnp.array([
            float(self._u_lb[0]), float(self._u_lb[1]),
            0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=DT)

        u_box = jnp.array([
            float(self._u_ub[0]), float(self._u_ub[1]),
            1e6, 1e6, 1e6, 1e6, 1e6
        ], dtype=DT)

        return H, g, C, b, l_box, u_box

    def get_diagnostics(self) -> dict:
        """Get diagnostic information from last QP solve - CORRECTED."""

        if self._current_local_graph is None or self._last_qp_sol is None:
            return {}

        local_graph = self._current_local_graph
        sol = self._last_qp_sol

        # Graph structure
        local_agent_mask = local_graph.node_type == 0
        local_goal_mask = local_graph.node_type == 1
        local_lidar_mask = local_graph.node_type == 2

        n_local_agents = int(jnp.sum(local_agent_mask))
        n_local_goals = int(jnp.sum(local_goal_mask))
        n_local_lidar = int(jnp.sum(local_lidar_mask))

        # Extract states
        agent_states = local_graph.type_states(type_idx=0, n_type=n_local_agents)
        ego_state = agent_states[0]

        # Solution (these attributes definitely exist)
        u_opt = sol.x[:2]
        r_opt = sol.x[2]

        # Iteration counts (exist in sol.info)
        qp_iter_inner = int(sol.info.iter_inner) if hasattr(sol.info, 'iter_inner') else -1
        qp_iter_ext = int(sol.info.iter_ext) if hasattr(sol.info, 'iter_ext') else -1

        # Residuals (exist directly in sol)
        qp_pri_res = float(sol.pri_res)
        qp_dua_res = float(sol.dua_res)

        # Objective and gap
        obj_value = float(sol.obj_value) if hasattr(sol, 'obj_value') else None
        duality_gap = float(sol.duality_gap) if hasattr(sol, 'duality_gap') else None

        # Determine status from convergence metrics
        if qp_pri_res < 1e-6 and qp_dua_res < 1e-6:
            qp_status = 'SOLVED'
        elif duality_gap is not None and duality_gap < 1e-6:
            qp_status = 'SOLVED'
        else:
            qp_status = 'OPTIMAL' if qp_pri_res < 1e-4 else 'SUBOPTIMAL'

        diagnostics = {
            'n_local_agents': n_local_agents,
            'n_local_goals': n_local_goals,
            'n_local_lidar': n_local_lidar,
            'ego_state': np.array(ego_state),
            'u_opt': np.array(u_opt),
            'relaxation': float(r_opt),
            'qp_status': qp_status,
            'qp_iter_inner': qp_iter_inner,
            'qp_iter_ext': qp_iter_ext,
            'qp_pri_res': qp_pri_res,
            'qp_dua_res': qp_dua_res,
            'obj_value': obj_value,
            'duality_gap': duality_gap,
        }

        return diagnostics

    def print_diagnostics(self, cbf_ego: Callable = None):
        """
        Print detailed diagnostics of last QP solve.

        Args:
            cbf_ego: Optional CBF function for additional safety checks
        """
        if self._current_local_graph is None or self._last_qp_sol is None:
            print("[Fallback] No diagnostic data available")
            return

        local_graph = self._current_local_graph
        sol = self._last_qp_sol

        # Graph structure
        local_agent_mask = local_graph.node_type == 0
        n_local_agents = int(jnp.sum(local_agent_mask))

        agent_states = local_graph.type_states(type_idx=0, n_type=n_local_agents)
        ego_state = agent_states[0]

        print("\n" + "=" * 60)
        print("FALLBACK CONTROLLER DIAGNOSTICS")
        print("=" * 60)

        # Graph info
        print(f"\nLocal Subgraph:")
        print(f"  Total agents: {n_local_agents} (ego + {n_local_agents - 1} neighbors)")
        print(f"  Ego state: {ego_state}")
        if n_local_agents > 1:
            print(f"  Neighbor states:")
            for i in range(1, n_local_agents):
                print(f"    Agent {i}: {agent_states[i]}")

        # CBF evaluation
        if cbf_ego is not None:
            h, hx = self._cbf_value_and_grad(local_graph, cbf_ego)
            Lf_h, Lg_h_force = self._lie_derivatives(local_graph, hx)

            print(f"\nCBF Evaluation:")
            print(f"  h = {float(h):.6f}")
            print(f"  ∇h = {hx}")
            print(f"  Lf_h = {float(Lf_h):.6f}")
            print(f"  Lg_h = {Lg_h_force}")

        # QP solution
        u_opt = sol.x[:2]
        r_opt = sol.x[2]

        print(f"\nQP Solution:")
        print(f"  u_opt = {u_opt}")
        print(f"  relaxation r = {float(r_opt):.6f}")
        print(f"  Status: {sol.info.status.name if hasattr(sol.info, 'status') else 'UNKNOWN'}")
        print(f"  Iterations: {sol.info.iter if hasattr(sol.info, 'iter') else 'N/A'}")
        print(f"  Primal residual: {float(sol.info.pri_res) if hasattr(sol.info, 'pri_res') else 'N/A'}")
        print(f"  Dual residual: {float(sol.info.dua_res) if hasattr(sol.info, 'dua_res') else 'N/A'}")

        # Constraint verification
        if cbf_ego is not None:
            alpha_eff = self.cfg.alpha * self.cfg.alpha_scale
            constraint_val = float(Lf_h + Lg_h_force @ u_opt + alpha_eff * h)

            print(f"\nSafety Constraint:")
            print(f"  ḣ + α·h = {constraint_val:.6f}")
            if constraint_val >= -1e-6:
                print(f"  Status: ✓ SATISFIED")
            else:
                print(f"  Status: ⚠️ VIOLATED by {abs(constraint_val):.6f}")

            if r_opt > 1e-6:
                print(f"  ⚠️ Relaxation active: r = {float(r_opt):.6f}")

        print("=" * 60 + "\n")


