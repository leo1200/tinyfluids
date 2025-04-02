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
STATE_TYPE = Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"]
FIELD_TYPE = Float[Array, "num_cells_x num_cells_y num_cells_z"]

# constants
DENSITY_INDEX = 0
VELOCITY_X_INDEX = 1
VELOCITY_Y_INDEX = 2
VELOCITY_Z_INDEX = 3
PRESSURE_INDEX = 4
VAR_AXIS = 0
X_AXIS = 1
Y_AXIS = 2
Z_AXIS = 3

def pad(input_array, padding, sharding):
    sharded_pad = shard_map(
        jnp.pad,
        mesh = sharding.mesh,
        in_specs = (sharding.spec, P()),
        out_specs = sharding.spec,
    )
    return sharded_pad(input_array, padding)


def unpad(input_array, padding, sharding):
    def _slice_to_unpad(padded_shard, padding_spec):
        return padded_shard[
            tuple(slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in padding_spec)
        ]

    # Apply the slicing operation across shards using shard_map
    sharded_unpad = shard_map(
        _slice_to_unpad,
        mesh=sharding.mesh,
        in_specs=(sharding.spec, P()),
        out_specs=sharding.spec
    )

    return sharded_unpad(input_array, padding)

@partial(jax.jit, static_argnames=['axis', 'num_blocks_along_axis'])
def send_right(shard, axis, num_blocks_along_axis):
    left_perm = [(i, (i + 1) % num_blocks_along_axis) for i in range(num_blocks_along_axis)] #  -1 for non periodic
    return jax.lax.ppermute(shard, perm=left_perm, axis_name = axis)

@partial(jax.jit, static_argnames=['axis', 'num_blocks_along_axis'])
def send_left(shard, axis, num_blocks_along_axis):
    # Create permutation based on device IDs along the axis
    left_perm = [((i + 1) % num_blocks_along_axis, i) for i in range(num_blocks_along_axis)] # -1 for non periodic
    return jax.lax.ppermute(shard, perm=left_perm, axis_name = axis)

@partial(jax.jit, static_argnames=['padding', 'axis'])
def collect_right_along_axis(shard, padding, axis):
    padding_axis = padding[axis]
    return jax.lax.slice_in_dim(shard, -padding_axis[1] - padding_axis[0], -padding_axis[1], axis=axis)

@partial(jax.jit, static_argnames=['padding', 'axis'])
def collect_left_along_axis(shard, padding, axis):
    padding_axis = padding[axis]
    return jax.lax.slice_in_dim(shard, padding_axis[0], padding_axis[0] + padding_axis[1], axis=axis)

@partial(jax.jit, static_argnames=['padding', 'num_blocks_along_axis', 'axis'])
def _halo_exchange_along_axis(shard, padding, num_blocks_along_axis, axis):

    left_halo = collect_left_along_axis(shard, padding, axis)
    right_halo = collect_right_along_axis(shard, padding, axis)

    right, left = send_left(left_halo, axis, num_blocks_along_axis), send_right(right_halo, axis, num_blocks_along_axis)

    shard = shard.at[(slice(None, None),) * axis + (slice(0, padding[axis][0]),) + (slice(None, None),) * (shard.ndim - axis - 1)].set(left)
    shard = shard.at[(slice(None, None),) * axis + (slice(-padding[axis][1], None),) + (slice(None, None),) * (shard.ndim - axis - 1)].set(right)

    return shard

@partial(jax.jit, static_argnames=['padding', 'split'])
def _halo_exchange(shard, padding, split):
    for _, axis in enumerate([X_AXIS, Y_AXIS, Z_AXIS]):
        if split[axis] > 1:
            shard = _halo_exchange_along_axis(shard, padding, split[axis], axis)
    return shard

# -------------------------------------------------------------
# ================== ↓ Basic Fluid Equations ↓ ================
# -------------------------------------------------------------

@jaxtyped(typechecker=typechecker)
@jax.jit
def conserved_state_from_primitive(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]]
) -> STATE_TYPE:
    """Convert the primitive state to the conserved state.

    Args:
        primitive_state: The primitive state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The conserved state.
    """
    
    rho = primitive_state[DENSITY_INDEX]
    p = primitive_state[PRESSURE_INDEX]

    # calculate the total energy
    utotal = jnp.sqrt(
        primitive_state[VELOCITY_X_INDEX]**2 + 
        primitive_state[VELOCITY_Y_INDEX]**2 + 
        primitive_state[VELOCITY_Z_INDEX]**2
    )
    E = p / (gamma - 1) + 0.5 * rho * utotal**2
    conserved_state = primitive_state.at[PRESSURE_INDEX].set(E)

    # set momentum = density * velocity
    conserved_state = conserved_state.at[VELOCITY_X_INDEX].set(
        rho * primitive_state[VELOCITY_X_INDEX]
    )
    conserved_state = conserved_state.at[VELOCITY_Y_INDEX].set(
        rho * primitive_state[VELOCITY_Y_INDEX]
    )
    conserved_state = conserved_state.at[VELOCITY_Z_INDEX].set(
        rho * primitive_state[VELOCITY_Z_INDEX]
    )

    return conserved_state

@jaxtyped(typechecker=typechecker)
@jax.jit
def primitive_state_from_conserved(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """Convert the conserved state to the primitive state.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The primitive state.
    """

    rho = conserved_state[DENSITY_INDEX]
    E = conserved_state[PRESSURE_INDEX]

    # retrieve the velocity components
    ux = conserved_state[VELOCITY_X_INDEX] / rho
    uy = conserved_state[VELOCITY_Y_INDEX] / rho
    uz = conserved_state[VELOCITY_Z_INDEX] / rho
    u = jnp.sqrt(ux**2 + uy**2 + uz**2)

    # calculate the pressure
    p = (gamma - 1) * rho * (E / rho - 0.5 * u**2)

    # set the primitive state
    primitive_state = conserved_state.at[PRESSURE_INDEX].set(p)
    primitive_state = primitive_state.at[VELOCITY_X_INDEX].set(ux)
    primitive_state = primitive_state.at[VELOCITY_Y_INDEX].set(uy)
    primitive_state = primitive_state.at[VELOCITY_Z_INDEX].set(uz)

    return primitive_state

@jax.jit
def speed_of_sound(rho, p, gamma):
    """Calculate the speed of sound.

    Args:
        rho: The density.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The speed of sound.
    """
    return jnp.sqrt(gamma * p / rho)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['flux_direction_index'])
def get_wave_speeds(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    flux_direction_index: int
) -> Tuple[FIELD_TYPE, FIELD_TYPE]:
    """
    Wave speed calculation.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index
        flux_direction_index: The index of the velocity component
                              in the flux direction of interest.

    Returns:
        Returns the wave speeds at the interfaces to the left and right.

    """
    
    # left state
    rho_L = primitives_left[DENSITY_INDEX]
    u_L = primitives_left[flux_direction_index]
    p_L = primitives_left[PRESSURE_INDEX]

    # right state
    rho_R = primitives_right[DENSITY_INDEX]
    u_R = primitives_right[flux_direction_index]
    p_R = primitives_right[PRESSURE_INDEX]

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # wave speeds
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    return wave_speeds_left_minus, wave_speeds_right_plus

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['flux_direction_index'])
def get_max_wave_speeds(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    flux_direction_index: int
) -> Union[float, Float[Array, ""]]:
    """
    Wave speed maximum

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index
        flux_direction_index: The index of the velocity component
                              in the flux direction of interest.

    Returns:
        Returns the maximum wave speed at the interfaces.

    """
    
    wave_speeds_left_minus, wave_speeds_right_plus = get_wave_speeds(
        primitives_left,
        primitives_right,
        gamma,
        flux_direction_index
    )

    return jnp.maximum(
        jnp.max(jnp.abs(wave_speeds_right_plus)),
        jnp.max(jnp.abs(wave_speeds_left_minus))
    )


# -------------------------------------------------------------
# ================== ↑ Basic Fluid Equations ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ==================== ↓ Flux Calculation ↓ ===================
# -------------------------------------------------------------

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['flux_direction_index'])
def _euler_flux(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    flux_direction_index: int
) -> STATE_TYPE:
    """Compute the Euler fluxes for the given primitive states.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        gamma: The adiabatic index of the fluid.
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component 
                              in the flux direction of interest.

    Returns:
        The Euler fluxes for the given primitive states.
    """
    rho = primitive_state[DENSITY_INDEX]
    p = primitive_state[PRESSURE_INDEX]
    
    # start with a copy of the primitive states
    flux_vector = primitive_state

    # calculate the total energy
    utotal = jnp.sqrt(
        primitive_state[VELOCITY_X_INDEX]**2 + 
        primitive_state[VELOCITY_Y_INDEX]**2 + 
        primitive_state[VELOCITY_Z_INDEX]**2
    )
    E = p / (gamma - 1) + 0.5 * rho * utotal**2

    # add the total energy to the pressure_index of the flux vector
    flux_vector = flux_vector.at[PRESSURE_INDEX].add(E)

    # scale the velocity components with the density
    flux_vector = flux_vector.at[VELOCITY_X_INDEX].set(
        primitive_state[VELOCITY_X_INDEX] * rho
    )
    flux_vector = flux_vector.at[VELOCITY_Y_INDEX].set(
        primitive_state[VELOCITY_Y_INDEX] * rho
    )
    flux_vector = flux_vector.at[VELOCITY_Z_INDEX].set(
        primitive_state[VELOCITY_Z_INDEX] * rho
    )

    # multiply the whole vector with the velocity component in the flux direction
    flux_vector = primitive_state[flux_direction_index] * flux_vector
    
    # add the pressure to the velocity component in the flux direction
    flux_vector = flux_vector.at[flux_direction_index].add(p)

    return flux_vector

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['flux_direction_index'])
def _hll_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    flux_direction_index: int
) -> STATE_TYPE:
    """
    Returns the conservative fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.

    Returns:
        The conservative fluxes at the interfaces.
    """
    
    # get the left and right states and fluxes
    fluxes_left = _euler_flux(primitives_left, gamma, flux_direction_index)
    fluxes_right = _euler_flux(primitives_right, gamma, flux_direction_index)
    
    # wave speeds
    wave_speeds_left_minus, wave_speeds_right_plus = get_wave_speeds(
        primitives_left,
        primitives_right,
        gamma,
        flux_direction_index
    )

    # get the left and right conserved variables
    conserved_left = conserved_state_from_primitive(primitives_left, gamma)
    conserved_right = conserved_state_from_primitive(primitives_right, gamma)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (
        wave_speeds_right_plus * fluxes_left -
        wave_speeds_left_minus * fluxes_right + 
        wave_speeds_left_minus * wave_speeds_right_plus * (conserved_right - conserved_left)
    ) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes

# -------------------------------------------------------------
# ==================== ↑ Flux Calculation ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ==================== ↓ Time Integration ↓ ===================
# -------------------------------------------------------------


@jaxtyped(typechecker=typechecker)
@jax.jit
def _cfl_time_step(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    C_CFL: Union[float, Float[Array, ""]] = 0.4
) -> Float[Array, ""]:

    """Calculate the time step based on the CFL condition.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The cell width.
        dt_max: The maximum time step.
        gamma: The adiabatic index.
        C_CFL: The CFL number.

    Returns:
        The time step.
    
    """

    # ==================== ↓ cell communication here ↓ ===================

    # wave speeds in x direction
    primitive_state_left = jax.lax.slice_in_dim(primitive_state, 0, -1, axis = X_AXIS)
    primitive_state_right = jax.lax.slice_in_dim(primitive_state, 1, None, axis = X_AXIS)
    max_wave_speed_x = get_max_wave_speeds(
        primitive_state_left,
        primitive_state_right,
        gamma,
        VELOCITY_X_INDEX
    )

    # wave speeds in y direction
    primitive_state_left = jax.lax.slice_in_dim(primitive_state, 0, -1, axis = Y_AXIS)
    primitive_state_right = jax.lax.slice_in_dim(primitive_state, 1, None, axis = Y_AXIS)
    max_wave_speed_y = get_max_wave_speeds(
        primitive_state_left,
        primitive_state_right,
        gamma,
        VELOCITY_Y_INDEX
    )

    # wave speeds in z direction
    primitive_state_left = jax.lax.slice_in_dim(primitive_state, 0, -1, axis = Z_AXIS)
    primitive_state_right = jax.lax.slice_in_dim(primitive_state, 1, None, axis = Z_AXIS)
    max_wave_speed_z = get_max_wave_speeds(
        primitive_state_left,
        primitive_state_right,
        gamma,
        VELOCITY_Z_INDEX
    )

    # ==================== ↑ cell communication here ↑ ===================

    # get the maximum wave speed
    max_wave_speed = jnp.maximum(
        jnp.maximum(max_wave_speed_x, max_wave_speed_y),
        max_wave_speed_z
    )

    dt = C_CFL * grid_spacing / max_wave_speed

    return dt

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis'])
def _evolve_state_along_axis(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    axis: int
) -> STATE_TYPE:
    
    # get conserved variables
    conservative_states = conserved_state_from_primitive(primitive_state, gamma)

    # ==================== ↓ cell communication here ↓ ===================
    primitive_state_left = jax.lax.slice_in_dim(primitive_state, 0, -1, axis = axis)
    primitive_state_right = jax.lax.slice_in_dim(primitive_state, 1, None, axis = axis)

    fluxes = _hll_solver(primitive_state_left, primitive_state_right, gamma, axis)

    # update the conserved variables
    conserved_change = -1 / grid_spacing * (
        jax.lax.slice_in_dim(fluxes, 1, None, axis = axis) - 
        jax.lax.slice_in_dim(fluxes, 0, -1, axis = axis)
    ) * dt

    conservative_states = conservative_states.at[
        tuple(slice(1, -1) if i == axis else slice(None) for i in range(conservative_states.ndim))
    ].add(conserved_change)
    # ==================== ↑ cell communication here ↑ ===================

    primitive_state = primitive_state_from_conserved(conservative_states, gamma)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@jax.jit
def _evolve_state(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    primitive_state = _evolve_state_along_axis(primitive_state, grid_spacing, dt, gamma, X_AXIS)
    primitive_state = _evolve_state_along_axis(primitive_state, grid_spacing, dt, gamma, Y_AXIS)
    primitive_state = _evolve_state_along_axis(primitive_state, grid_spacing, dt, gamma, Z_AXIS)

    return primitive_state

@jaxtyped(typechecker=typechecker)
@jax.jit
def time_integration(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    t_final: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
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

        padding = ((0, 0), (1, 1), (1, 1), (0, 0))
        split = (1, 2, 2, 1)
        primitive_state = _halo_exchange(primitive_state, padding, split)

        # the wave speed calculation is currently done twice, once
        # for finding dt and once for the state update, which
        # is not optimal
        dt_local = _cfl_time_step(primitive_state, grid_spacing, gamma)

        dt = jax.lax.pmin(dt_local, axis_name=(1.0, 2.0))

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

# -------------------------------------------------------------
# ==================== ↑ Time Integration ↑ ===================
# -------------------------------------------------------------