import os

import jax

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"

from jax.experimental.shard_map import shard_map

from tinyfluids.jax_tinyfluids.jax_shardmap import _halo_exchange, pad, time_integration, unpad
from tinyfluids.jax_tinyfluids.jax_shardmap import DENSITY_INDEX, PRESSURE_INDEX, X_AXIS, Y_AXIS, Z_AXIS, VAR_AXIS
import timeit
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding

import matplotlib.pyplot as plt

def plot_results(final_state):
    num_cells = final_state.shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(final_state[DENSITY_INDEX, :, :, num_cells // 2], extent = [0, 1, 0, 1])
    axs[0].set_title("Density")
    axs[1].imshow(final_state[PRESSURE_INDEX, :, :, num_cells // 2], extent = [0, 1, 0, 1])
    axs[1].set_title("Pressure")
    plt.savefig("figures/check_{:d}.png".format(num_cells))

def setup_ics(num_cells, num_injection_cells=2):
    grid_spacing = 1 / (num_cells - 1)

    rho = jnp.ones((num_cells, num_cells, num_cells)) * 0.125
    u_x = jnp.zeros((num_cells, num_cells, num_cells))
    u_y = jnp.zeros((num_cells, num_cells, num_cells))
    u_z = jnp.zeros((num_cells, num_cells, num_cells))
    p = jnp.ones((num_cells, num_cells, num_cells)) * 0.1

    center = num_cells // 2

    injection_slice = slice(center - num_injection_cells, center + num_injection_cells)

    rho = rho.at[injection_slice, injection_slice, injection_slice].set(1.0)
    p = p.at[injection_slice, injection_slice, injection_slice].set(1.0)

    primitive_state = jnp.stack([rho, u_x, u_y, u_z, p], axis = 0)

    return primitive_state, grid_spacing

num_cells = 256
num_injection_cells = num_cells // 16
primitive_state, grid_spacing = setup_ics(num_cells, num_injection_cells)

t_final = 0.2
gamma = 5/3

# TODO: do outer boarders better

sharding_mesh = jax.make_mesh((1, 2, 2, 1), (VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
sharding = jax.NamedSharding(sharding_mesh, P(VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
primitive_state = jax.device_put(primitive_state, sharding)

padding = ((0, 0), (1, 1), (1, 1), (0, 0))
primitive_state = pad(primitive_state, padding, sharding)

sharded_halo_exchange = shard_map(
    _halo_exchange,
    mesh=sharding_mesh,
    in_specs=(sharding.spec, None, None),
    out_specs=sharding.spec
)
primitive_state = sharded_halo_exchange(primitive_state, padding, (1, 2, 2, 1))
plot_results(primitive_state)

jax.debug.visualize_array_sharding(primitive_state[0, :, :, 0])

sharded_time_integration = shard_map(
    time_integration,
    mesh=sharding_mesh,
    in_specs = (sharding.spec, None, None, None),
    out_specs = (sharding.spec, P()),
    check_rep = False
)

# Execute once for compilation and warmup
final_state, num_iterations = sharded_time_integration(primitive_state, grid_spacing, t_final, gamma)
final_state.block_until_ready()

final_state = unpad(final_state, padding, sharding)

plot_results(final_state)