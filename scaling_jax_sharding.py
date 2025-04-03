import os

import jax

os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6,7"

from jax.experimental.shard_map import shard_map

from tinyfluids.jax_tinyfluids.time_integration import _halo_exchange, _time_integration_inner, time_integration
from tinyfluids.jax_tinyfluids.sharding_helpers import pad, unpad
from tinyfluids.jax_tinyfluids.fluid import DENSITY_INDEX, PRESSURE_INDEX, X_AXIS, Y_AXIS, Z_AXIS, VAR_AXIS
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

num_cells = 512
num_injection_cells = num_cells // 16
primitive_state, grid_spacing = setup_ics(num_cells, num_injection_cells)

t_final = 0.2
gamma = 5/3

shard = True
shard_mapped = True

# TODO: do outer boarders better
if shard:
    sharding_mesh = jax.make_mesh((1, 2, 2, 1), (VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
    sharding = jax.NamedSharding(sharding_mesh, P(VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
    primitive_state = jax.device_put(primitive_state, sharding)

    padding = ((0, 0), (1, 1), (1, 1), (0, 0))

    if shard_mapped:
        primitive_state = pad(primitive_state, padding, sharding)

print("started first run")

# Execute once for compilation and warmup
if shard_mapped:
    final_state, num_iterations = time_integration(primitive_state, grid_spacing, t_final, gamma, shard_mapped)
else:
    final_state, num_iterations = _time_integration_inner(primitive_state, grid_spacing, t_final, gamma, shard_mapped)
final_state.block_until_ready()

print("finished first run")

if shard_mapped:
    final_state = unpad(final_state, padding, sharding)

plot_results(final_state)

def time_execution():
    if shard_mapped:
        final_state, _ = time_integration(primitive_state, grid_spacing, t_final, gamma, shard_mapped)
    else:
        final_state, _ = _time_integration_inner(primitive_state, grid_spacing, t_final, gamma, shard_mapped)
    final_state.block_until_ready()

# Measure execution time
times = timeit.repeat(
    time_execution,
    repeat = 3,  # More repeats for better statistics
    number = 1   # Number of calls per measurement
)
print(times)
print(f"Execution time: {min(times)} seconds")