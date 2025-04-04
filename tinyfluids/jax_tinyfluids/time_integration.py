"""
This is a baseline first order Eulerian hydrodynamics solver
with a HLL Riemann solver for the Euler equations written in JAX.

Parallelization using shardmap and halo exchange.

For a more feature-rich fluid code for astrophysics in JAX,
check out https://github.com/leo1200/jf1uids, for other
purposes https://github.com/tumaer/JAXFLUIDS.
"""

# general imports
from functools import partial
from types import NoneType
import jax.numpy as jnp
import jax

# sharding
from jax.sharding import Mesh
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# typing
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

from tinyfluids.jax_tinyfluids.sharding_helpers import _halo_exchange
from tinyfluids.jax_tinyfluids.fluid import STATE_TYPE, _cfl_time_step, _evolve_state

# -------------------------------------------------------------
# ==================== ↓ Time Integration ↓ ===================
# -------------------------------------------------------------

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["shard_mapped", "padding", "split"])
def _time_integration_inner(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    t_final: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    shard_mapped: bool,
    padding: Union[NoneType, Tuple[Tuple[int, int], ...]] = None,
    split: Union[NoneType, Tuple[int, ...]] = None,
) -> Tuple[STATE_TYPE, Union[float, Int[Array, ""]]]:
    """
    Evolve the state of the fluid in time.

    Args:
        primitive_state: The primitive state of the fluid,
                         sharded according to the sharding_mesh.
        sharding_mesh: The mesh used for sharding.
        grid_spacing: The grid spacing.
        dt: The time step.
        gamma: The adiabatic index of the fluid.
        num_steps: The number of time steps to evolve the state.

    Returns:
        The evolved state of the fluid.
    """

    def time_step_fn(state):
        primitive_state, t, num_iterations = state

        if shard_mapped:
            primitive_state = _halo_exchange(primitive_state, padding, split)

        # the wave speed calculation is currently done twice, once
        # for finding dt and once for the state update, which
        # is not optimal
        dt = _cfl_time_step(primitive_state, grid_spacing, gamma)

        if shard_mapped:
            # see https://github.com/jax-ml/jax/issues/27665
            dt = jax.lax.pmin(dt, axis_name=(1.0, 2.0))

        primitive_state = _evolve_state(primitive_state, grid_spacing, dt, gamma)

        t += dt
        num_iterations += 1

        return primitive_state, t, num_iterations

    def cond_fn(state):
        _, t, _ = state
        return t < t_final

    initial_state = (primitive_state, 0.0, 0)
    primitive_state, _, num_iterations = jax.lax.while_loop(cond_fn, time_step_fn, initial_state)

    return primitive_state, num_iterations

def time_integration(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    t_final: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    shard_mapped: bool,
    padding: Union[NoneType, Tuple[Tuple[int, int], ...]] = None,
    split: Union[NoneType, Tuple[int, ...]] = None,
) -> Tuple[STATE_TYPE, Union[float, Int[Array, ""]]]:
    """
    Evolve the state of the fluid in time.

    Args:
        primitive_state: The primitive state of the fluid,
                         sharded according to the sharding_mesh.
        grid_spacing: The grid spacing.
        dt: The time step.
        gamma: The adiabatic index of the fluid.
        num_steps: The number of time steps to evolve the state.

    Returns:
        The evolved state of the fluid.
    """
    if shard_mapped:
        sharded_time_integration = shard_map(
            _time_integration_inner,
            mesh = primitive_state.sharding.mesh,
            in_specs = (primitive_state.sharding.spec, None, None, None, None, None, None),
            out_specs = (primitive_state.sharding.spec, P()),
            check_rep = False
        )
        return sharded_time_integration(primitive_state, grid_spacing, t_final, gamma, True, padding, split)
    else:
        return _time_integration_inner(primitive_state, grid_spacing, t_final, gamma, False)

# -------------------------------------------------------------
# ==================== ↑ Time Integration ↑ ===================
# -------------------------------------------------------------