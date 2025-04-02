import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import numpy as jnp
import jaxdecomp
from functools import partial

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4" 

# general
from functools import partial

# typechecking
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple, Union

    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['indices', 'axis', 'zero_pad'])
def _stencil_add(
        input_array: jnp.ndarray,
        indices: Tuple[int, ...],
        factors: Tuple[Union[float, Float[Array, ""]], ...],
        axis: int,
        zero_pad: bool = True
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    By default, the output is zero-padded to the same shape as 
    the input array (as we handle boundaries via ghost cells in 
    the overall simulation code). This behavior can be disabled,
    then the output will have a different shape along the specified
    axis.

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.
        zero_pad: Whether to zero-pad the output to have the same shape as the input.
        
    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """

    num_cells = input_array.shape[axis]

    first_write_index = -min(0, min(indices))
    last_write_index = num_cells - max(0, max(indices))

    # for the first write index, the elements considered are
    first_handled_indices = tuple(first_write_index + index for index in indices)

    # for the last write index, the elements considered are
    last_handled_indices = tuple(last_write_index + index for index in indices)

    output = (
        sum(
            factor * jax.lax.slice_in_dim(
                input_array,
                first_handled_index,
                last_handled_index,
                axis = axis
            )
            for factor, first_handled_index, last_handled_index in zip(
                factors, first_handled_indices, last_handled_indices
            )
        )
    )

    if zero_pad:
        result = jnp.zeros_like(input_array)
        selection = (
            (slice(None),) * axis +
            (slice(first_write_index, last_write_index),) +
            (slice(None),)*(input_array.ndim - axis - 1)
        )
        result = result.at[selection].set(output)
        return result
    else:
        return output


# -----------------------------
# Create sharded array
# -----------------------------

# split into 2x2x1 partitions
pdims = (2, 2)
global_shape = (16, 16, 16)

# Compute local slice sizes
local_shape = (
    global_shape[0] // pdims[0],
    global_shape[1] // pdims[1],
    global_shape[2]
)

# Create a mesh of devices based on pdims
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('x', 'y'))

# Define the sharding spec
sharding = NamedSharding(mesh, P('x', 'y'))

# Create a distributed global array
global_array = jax.make_array_from_callback(
    global_shape,
    sharding,
    data_callback=lambda _: jnp.ones(local_shape, dtype=jnp.float32),
)

# -----------------------------
# Sharded computation helpers
# -----------------------------
# We will also demonstrate applying a halo exchange afterwards.

padding_width = ((1, 1), (1, 1), (0, 0))  # must be a tuple of tuples

# Shard-map helper to pad an array
@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def pad(arr, padding):
    return jnp.pad(arr, padding)

# # Shard-map helper to remove the padded halo
# @partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
# def reduce_halo(x, pad_width):
#     halo_x, _ = pad_width[0]
#     halo_y, _ = pad_width[1]
#     # Apply corrections along x
#     x = x.at[halo_x:halo_x + halo_x // 2].add(x[:halo_x // 2])
#     x = x.at[-(halo_x + halo_x // 2):-halo_x].add(x[-halo_x // 2:])
#     # Apply corrections along y
#     x = x.at[:, halo_y:halo_y + halo_y // 2].add(x[:, :halo_y // 2])
#     x = x.at[:, -(halo_y + halo_y // 2):-halo_y].add(x[:, -halo_y // 2:])
#     return x[halo_x:-halo_x, halo_y:-halo_y]

@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), None), out_specs=P('x', 'y'))
def reduce_halo(x_shard, pad_width_global): # Renamed arg for clarity
    """Removes padding from a shard based on the global padding width."""
    slices = []
    for i, axis_pad in enumerate(pad_width_global):
        left_pad, right_pad = axis_pad
        dim_size = x_shard.shape[i]
        stop_index = dim_size - right_pad if right_pad > 0 else None
        s = slice(left_pad, stop_index)
        slices.append(s)
    return x_shard[tuple(slices)]

# make a shard mapped version of the _stencil_add function
@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), None, None, None), out_specs=P('x', 'y'))
def _stencil_add_shard_map(
        input_array: jnp.ndarray,
        indices: Tuple[int, ...],
        factors: Tuple[Union[float, Float[Array, ""]], ...],
        axis: int
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    By default, the output is zero-padded to the same shape as 
    the input array (as we handle boundaries via ghost cells in 
    the overall simulation code). This behavior can be disabled,
    then the output will have a different shape along the specified
    axis.

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.
        zero_pad: Whether to zero-pad the output to have the same shape as the input.
        
    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """
    return _stencil_add(input_array, indices, factors, axis)


# -----------------------------
# halo exchange
# -----------------------------

# Example: pad the array, exchange halos, then remove the padding
padded_array = pad(global_array, padding_width)
print(padded_array[:, :, 0])
padded_array = jaxdecomp.halo_exchange(
    padded_array,
    halo_extents=(1, 1),
    halo_periods=(True, True)
)
print(padded_array[:, :, 0])
reduced_array = reduce_halo(padded_array, padding_width)
print(reduced_array[:, :, 0])

# Exchange halo across processes
exchanged_array = jaxdecomp.halo_exchange(
    padded_array,
    halo_extents=(1, 1),
    halo_periods=(True, True)
)

finite_diff_result = _stencil_add_shard_map(
    exchanged_array,
    (1, 0),
    (1.0, -1.0),
    0,
)


finite_diff_result_non_sharded = _stencil_add(
    global_array,
    (1, 0),
    (1.0, -1.0),
    0,
)



# Remove the halo paddings after exchange
reduced_array = reduce_halo(finite_diff_result, padding_width)

print("non-sharded result:")
print(finite_diff_result_non_sharded[:, :, 0])
print("sharded result:")
print(reduced_array[:, :, 0])

print("non-sharded result shape:")
print(finite_diff_result_non_sharded.shape)

print("sharded result shape:")
print(reduced_array.shape)

# check that the results are the same
assert jnp.allclose(
    finite_diff_result_non_sharded,
    reduced_array
), "The results of the sharded and non-sharded computations do not match!"
print("The results of the sharded and non-sharded computations match!")

