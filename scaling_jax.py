import os

import jax
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5" 

from tinyfluids.jax_tinyfluids import time_integration
from tinyfluids.jax_tinyfluids import DENSITY_INDEX, PRESSURE_INDEX
import timeit
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding

jax.config.update("jax_debug_nans", True)

import matplotlib.pyplot as plt

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

def measure_execution_time(primitive_state, grid_spacing, t_final, gamma):
    # execute once for compilation and warmup

    final_state, num_iterations = time_integration(primitive_state, grid_spacing, t_final, gamma)
    final_state.block_until_ready()

    plot_results(final_state)
    
    # Create a function for timing
    def timed_execution():
        final_state, _ = time_integration(primitive_state, grid_spacing, t_final, gamma)
        final_state.block_until_ready()  # Ensure execution completes before timing stops
        
    # Measure execution time
    times = timeit.repeat(
        timed_execution,
        repeat=5,  # More repeats for better statistics
        number=3   # Number of calls per measurement
    )
    
    # Calculate statistics
    min_time = min(times) / 3  # Per single execution
    mean_time = sum(times) / len(times) / 3

    min_time_per_iteration = min_time / num_iterations
    
    print(f"Execution time: min={min_time:.6f}s, mean={mean_time:.6f}s")
    return min_time, min_time_per_iteration  # Minimum time is often most representative for benchmarking

def plot_results(final_state):
    num_cells = final_state.shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(final_state[DENSITY_INDEX, :, :, num_cells // 2], extent = [0, 1, 0, 1])
    axs[0].set_title("Density")
    axs[1].imshow(final_state[PRESSURE_INDEX, :, :, num_cells // 2], extent = [0, 1, 0, 1])
    axs[1].set_title("Pressure")
    plt.savefig("figures/check_{:d}.png".format(num_cells))

def make_scaling_plots(sharding = False):
    
    t_final = 0.2
    gamma = 5/3

    num_cells_list = [32, 64, 128, 256, 512]
    execution_times = []
    execution_times_per_iteration = []

    for num_cells in num_cells_list:
        primitive_state, grid_spacing = setup_ics(num_cells, num_cells // 16)

        if sharding:
            sharding_mesh = jax.make_mesh((1, 2, 2, 1), ('variables', 'x', 'y'))
            sharding = jax.NamedSharding(sharding_mesh, P('variables', 'x', 'y'))
            primitive_state = jax.device_put(primitive_state, sharding)

            jax.debug.visualize_array_sharding(primitive_state[0, :, :, 0])

        execution_time, execution_time_per_iteration = measure_execution_time(primitive_state, grid_spacing, t_final, gamma)
        execution_times.append(execution_time)
        execution_times_per_iteration.append(execution_time_per_iteration)


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(num_cells_list, execution_times, 'o-')
    axs[0].set_title("Execution time")
    axs[0].set_xlabel("Number of cells")
    axs[0].set_ylabel("Execution time (s)")

    axs[1].plot(num_cells_list, execution_times_per_iteration, 'o-')
    axs[1].set_title("Execution time per timestep")
    axs[1].set_xlabel("Number of cells")
    axs[1].set_ylabel("Execution time per timestep (s)")

    plt.savefig("figures/scaling{}.png".format("_sharding" if sharding else ""))

make_scaling_plots(sharding = False)
make_scaling_plots(sharding = True)