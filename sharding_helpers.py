import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9" 

VARIABLE_AXIS = 0
X_AXIS = 1
Y_AXIS = 2
Z_AXIS = 3

# general
from functools import partial
import jax
import jax.numpy as jnp

# sharding
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

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
    left_perm = [(i, (i + 1) % num_blocks_along_axis) for i in range(num_blocks_along_axis - 1)]
    return jax.lax.ppermute(shard, perm=left_perm, axis_name = axis)

@partial(jax.jit, static_argnames=['axis', 'num_blocks_along_axis'])
def send_left(shard, axis, num_blocks_along_axis):
    # Create permutation based on device IDs along the axis
    left_perm = [((i + 1) % num_blocks_along_axis, i) for i in range(num_blocks_along_axis - 1)]
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


x = jnp.ones((4, 4, 4, 4))
split = (1, 2, 2, 1)
sharding_mesh = jax.make_mesh(split, (VARIABLE_AXIS, X_AXIS, Y_AXIS, Z_AXIS))
partition_spec = P(VARIABLE_AXIS, X_AXIS, Y_AXIS, Z_AXIS)
sharding = jax.NamedSharding(sharding_mesh, partition_spec)
x = jax.device_put(x, sharding)
print(x[0, :, :, 0])

padding = ((0, 0), (1, 1), (1, 1), (0, 0))
x_pad = pad(x, padding, sharding)
print(x_pad[0, :, :, 0])
x_unpad = unpad(x_pad, padding, sharding)
print(x_unpad[0, :, :, 0])

sharded_halo_exchange = shard_map(
    _halo_exchange,
    mesh=sharding.mesh,
    in_specs=(sharding.spec, None, None),
    out_specs=sharding.spec
)
x_halo = sharded_halo_exchange(x_pad, padding, split)
print(x_halo[0, :, :, 0])

@partial(jax.jit, static_argnames=['axis'])
def ttt(axis):
    print(axis)
    return(axis)

ttt("hh")