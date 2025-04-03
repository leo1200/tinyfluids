import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import numpy as jnp

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8" 

# general
from functools import partial

# split into 2x2x1 partitions
pdims = (2, 2)
global_shape = (16, 16, 16)

# Create a mesh of devices based on pdims
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names = ("x", "y"))

# Define the sharding spec
sharding = NamedSharding(mesh, P("x", "y"))

# random global array then put to device
global_array = jax.random.uniform(
    jax.random.PRNGKey(0),
    shape=global_shape,
    minval=0.0,
    maxval=1.0,
)

# print the true global min
global_min = jnp.min(global_array)
print("True global min:", global_min)

global_array = jax.device_put(global_array, sharding)

def min_test(array):
    # Compute the local minimum
    local_min = jnp.min(array)
    print("Local min:", local_min)
    
    # Use collectives to compute the global minimum across all devices
    global_min = jax.lax.pmin(local_min, axis_name = ("x", "y"))
    print("Global min:", global_min)
    
    return array

# Create a sharded version of the function
sharded_get_min = shard_map(
    min_test,
    mesh=mesh,
    in_specs=(sharding.spec,),
    out_specs=sharding.spec,
)

# Call the sharded function
sharded_result = sharded_get_min(global_array)