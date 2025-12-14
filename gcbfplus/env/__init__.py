# from typing import Optional
#
# from .base import MultiAgentEnv                     # Base class to inherit from
# from .single_integrator import SingleIntegrator
# from .double_integrator import DoubleIntegrator
# from .linear_drone import LinearDrone
# from .dubins_car import DubinsCar
# from .crazyflie import CrazyFlie
#
#
# ENV = {                                             # Dictionary mapping string names to class objects
#     'SingleIntegrator': SingleIntegrator,
#     'DoubleIntegrator': DoubleIntegrator,
#     'LinearDrone': LinearDrone,
#     'DubinsCar': DubinsCar,
#     'CrazyFlie': CrazyFlie,
# }
#
#
# DEFAULT_MAX_STEP = 256
#
#
# def make_env(
#         env_id: str,
#         num_agents: int,
#         area_size: float = None,
#         max_step: int = None,
#         max_travel: Optional[float] = None,
#         num_obs: Optional[int] = None,
#         n_rays: Optional[int] = None,
# ) -> MultiAgentEnv:
#     assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'       # Sanity check: assert condition, message
#     params = ENV[env_id].PARAMS                                                 # ENV[env_id] returns the class, then access PARAMS from corresponding environment class
#     max_step = DEFAULT_MAX_STEP if max_step is None else max_step
#     if num_obs is not None:
#         params['n_obs'] = num_obs                                               # Override obstacle number according to make
#     if n_rays is not None:
#         params['n_rays'] = n_rays                                               # Override ray number according to make
#     return ENV[env_id](
#         num_agents=num_agents,
#         area_size=area_size,
#         max_step=max_step,
#         max_travel=max_travel,
#         dt=0.03,
#         params=params
#     )


from typing import Optional

from .base import MultiAgentEnv  # Base class to inherit from
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .double_integrator_no_clipping import DoubleIntegratorNoClipping
from .linear_drone import LinearDrone
from .dubins_car import DubinsCar
from .crazyflie import CrazyFlie
from .fixed_double_integrator import FixedObstacleDoubleIntegrator

ENV = {  # Dictionary mapping string names to class objects
    'SingleIntegrator': SingleIntegrator,
    'DoubleIntegrator': DoubleIntegrator,
    'DoubleIntegratorNoClipping': DoubleIntegratorNoClipping,
    'LinearDrone': LinearDrone,
    'DubinsCar': DubinsCar,
    'CrazyFlie': CrazyFlie,
    'FixedObstacleDoubleIntegrator': FixedObstacleDoubleIntegrator
}

DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        # MPC-specific parameters
        mpc_horizon: Optional[int] = None,
        Q_mpc: Optional[float] = None,
        R_mpc: Optional[float] = None,
        u_max: Optional[float] = None,
        terminal_weight: Optional[float] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'  # Sanity check: assert condition, message
    params = ENV[
        env_id].PARAMS.copy()  # ENV[env_id] returns the class, then access PARAMS from corresponding environment class
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs  # Override obstacle number according to make
    if n_rays is not None:
        params['n_rays'] = n_rays  # Override ray number according to make

    # Create base environment
    env_kwargs = {
        'num_agents': num_agents,
        'area_size': area_size,
        'max_step': max_step,
        'max_travel': max_travel,
        'dt': 0.03,
        'params': params
    }

    # Add MPC-specific parameters if this is an MPC environment
    if env_id == 'DoubleIntegratorMPC':
        if mpc_horizon is not None:
            env_kwargs['mpc_horizon'] = mpc_horizon
        if Q_mpc is not None:
            env_kwargs['Q_mpc'] = Q_mpc
        if R_mpc is not None:
            env_kwargs['R_mpc'] = R_mpc
        if u_max is not None:
            env_kwargs['u_max'] = u_max
        if terminal_weight is not None:
            env_kwargs['terminal_weight'] = terminal_weight

    return ENV[env_id](**env_kwargs)
