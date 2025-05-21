"""
Benchmark of the baseline implementation in JAX.
(currently rather untidy, code, TODO: clean up)
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 4)
# =======================

import jax

from tinyfluids.jax_tinyfluids.sharding_helpers import pad, unpad
from tinyfluids.jax_tinyfluids.time_integration import time_integration
from tinyfluids.jax_tinyfluids.fluid import DENSITY_INDEX, PRESSURE_INDEX, VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS
import timeit
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding

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

def measure_execution_time(primitive_state, grid_spacing, t_final, gamma, shard_mapped, padding = None, split = None, handle_boundaries = False):
    # execute once for compilation and warmup

    final_state, num_iterations = time_integration(primitive_state, grid_spacing, t_final, gamma, handle_boundaries, shard_mapped, padding, split)
    final_state.block_until_ready()

    if shard_mapped:
        final_state = unpad(final_state, padding, final_state.sharding)

    plot_results(final_state)
    
    # Create a function for timing
    def timed_execution():
        final_state, _ = time_integration(primitive_state, grid_spacing, t_final, gamma, handle_boundaries, shard_mapped, padding, split)
        final_state.block_until_ready()  # Ensure execution completes before timing stops
        
    # Measure execution time
    times = timeit.repeat(
        timed_execution,
        repeat = 3,  # More repeats for better statistics
        number = 1   # Number of calls per measurement
    )
    
    # Calculate statistics
    min_time = min(times)
    mean_time = sum(times) / len(times)

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

def make_scaling_plots(sharding = False, shard_mapped = False, num_cells_list = [32, 64, 128, 256, 512, 1024], handle_boundaries = False):
    
    t_final = 0.2
    gamma = 5/3

    execution_times = []
    execution_times_per_iteration = []

    for num_cells in num_cells_list:
        primitive_state, grid_spacing = setup_ics(num_cells, num_cells // 16)

        if sharding:
            split = (1, 2, 2, 1)
            sharding_mesh = jax.make_mesh(split, (VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
            sharding = jax.NamedSharding(sharding_mesh, P(VAR_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
            primitive_state = jax.device_put(primitive_state, sharding)

            jax.debug.visualize_array_sharding(primitive_state[0, :, :, 0])

            if shard_mapped:
                padding = ((0, 0), (1, 1), (1, 1), (0, 0))
                primitive_state = pad(primitive_state, padding, sharding)
            else:
                padding = None
        else:
            padding = None
            split = None
        
        execution_time, execution_time_per_iteration = measure_execution_time(primitive_state, grid_spacing, t_final, gamma, shard_mapped, padding, split, handle_boundaries)
        execution_times.append(execution_time)
        execution_times_per_iteration.append(execution_time_per_iteration)


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(num_cells_list, execution_times, 'o-', label = "execution time")


    # set x and y log scale
    axs[0].set_xscale("log", base = 2)
    axs[0].set_yscale("log", base = 2)
    axs[0].legend()

    axs[0].set_title("Execution time")
    axs[0].set_xlabel("Number of cells")
    axs[0].set_ylabel("Execution time (s)")

    axs[1].plot(num_cells_list, execution_times_per_iteration, 'o-', label = "execution time per iteration")
    # set x and y log scale
    axs[1].set_xscale("log", base = 2)
    axs[1].set_yscale("log", base = 2)
    axs[1].legend()
    axs[1].set_title("Execution time per timestep")
    axs[1].set_xlabel("Number of cells")
    axs[1].set_ylabel("Execution time per timestep (s)")

    num_cells_list = jnp.array(num_cells_list, dtype=jnp.float32)
    execution_times = jnp.array(execution_times, dtype=jnp.float32)
    execution_times_per_iteration = jnp.array(execution_times_per_iteration)

    file_appendix = ""

    if sharding and not shard_mapped:
        file_appendix = "_sharding"
    elif sharding and shard_mapped:
        file_appendix = "_shard_mapped"

    # save the data for later analysis
    jnp.savez(
        "results/scaling_data{}.npz".format(file_appendix),
        num_cells_list = num_cells_list,
        execution_times = execution_times,
        execution_times_per_iteration = execution_times_per_iteration
    )

    # save the figure
    plt.savefig("figures/scaling{}.png".format(file_appendix))

def plot_scaling_results(plot_sharded = True, plot_sharded_mapped = True):
    # load the data
    data_unsharded = jnp.load("results/scaling_data.npz")

    if plot_sharded:
        data_sharded = jnp.load("results/scaling_data_sharding.npz")
    if plot_sharded_mapped:
        data_sharded_mapped = jnp.load("results/scaling_data_shard_mapped.npz")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # set figure title "scaling for a 3D shock test"
    fig.suptitle("Scaling for a 3D shock test")

    axs[0].plot(data_unsharded["num_cells_list"], data_unsharded["execution_times"], 'o-', label = "unsharded", color = "blue")

    if plot_sharded:
        axs[0].plot(data_sharded["num_cells_list"], data_sharded["execution_times"], 'o-', label = "sharded (4 GPUs)", color = "orange")

    if plot_sharded_mapped:
        axs[0].plot(data_sharded_mapped["num_cells_list"], data_sharded_mapped["execution_times"], 'o-', label = "sharded (4 GPUs, shard mapped)", color = "green")
    # set x and y log scale
    axs[0].set_xscale("log", base = 2)
    axs[0].set_yscale("log", base = 2)

    axs[0].legend()
    axs[0].set_title("Execution time")
    axs[0].set_xlabel("Number of cells per dimension")
    axs[0].set_ylabel("Execution time in s")
    axs[1].plot(data_unsharded["num_cells_list"], data_unsharded["execution_times_per_iteration"], 'o-', label = "unsharded", color = "blue")

    if plot_sharded:
        axs[1].plot(data_sharded["num_cells_list"], data_sharded["execution_times_per_iteration"], 'o-', label = "sharded (4 GPUs)", color = "orange")

    if plot_sharded_mapped:
        axs[1].plot(data_sharded_mapped["num_cells_list"], data_sharded_mapped["execution_times_per_iteration"], 'o-', label = "sharded (4 GPUs, shard mapped)", color = "green")
    # set x and y log scale
    axs[1].set_xscale("log", base = 2)
    axs[1].set_yscale("log", base = 2)
    axs[1].legend()
    axs[1].set_title("Execution time per timestep")
    axs[1].set_xlabel("Number of cells per dimension")
    axs[1].set_ylabel("Execution time per timestep in s")

    # plot the speedup
    if plot_sharded and not plot_sharded_mapped:
        common_length = min(len(data_unsharded["num_cells_list"]), len(data_sharded["num_cells_list"]))
    if plot_sharded_mapped and not plot_sharded:
        common_length = min(len(data_unsharded["num_cells_list"]), len(data_sharded_mapped["num_cells_list"]))
    if plot_sharded and plot_sharded_mapped:
        common_length = min(len(data_unsharded["num_cells_list"]), len(data_sharded["num_cells_list"]), len(data_sharded_mapped["num_cells_list"]))

    if plot_sharded:
        speedup = data_unsharded["execution_times"][0:common_length] / data_sharded["execution_times"][0:common_length]

    if plot_sharded_mapped:
        speedup_mapped = data_unsharded["execution_times"][0:common_length] / data_sharded_mapped["execution_times"][0:common_length]

    if plot_sharded:
        axs[2].plot(data_unsharded["num_cells_list"][0:common_length], speedup, 'o-', label = "speedup sharded (4 GPUs)", color = "orange")

    if plot_sharded_mapped:
        axs[2].plot(data_unsharded["num_cells_list"][0:common_length], speedup_mapped, 'o-', label = "speedup (shard mapped)", color = "green")
    
    # theoretical speedup is 4
    axs[2].plot(data_unsharded["num_cells_list"][0:common_length], jnp.ones_like(data_unsharded["num_cells_list"][0:common_length]) * 4, '--', label = "theoretical speedup", color = "black")
    axs[2].legend()
    axs[2].set_title("Speedup")
    axs[2].set_xlabel("Number of cells per dimension")

    plt.tight_layout()

    plt.savefig("figures/scaling_results.png")


handle_boundaries = True

make_scaling_plots(sharding = False, shard_mapped = False, num_cells_list = [32, 64, 96, 128, 196, 256], handle_boundaries = handle_boundaries)
make_scaling_plots(sharding = True, shard_mapped = False, num_cells_list  = [32, 64, 96, 128, 196, 256], handle_boundaries = handle_boundaries)
# make_scaling_plots(sharding = True, shard_mapped = True, num_cells_list  = [32, 64, 96, 128, 196, 256], handle_boundaries = handle_boundaries)

plot_scaling_results(plot_sharded = True, plot_sharded_mapped = False)