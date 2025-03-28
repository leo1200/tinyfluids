from tinyfluids.jax_tinyfluids import time_integration
from tinyfluids.jax_tinyfluids import DENSITY_INDEX
import jax.numpy as jnp

num_cells = 20
grid_spacing = 1 / (num_cells - 1)

rho = jnp.ones((num_cells, num_cells, num_cells)) * 0.125
u_x = jnp.zeros((num_cells, num_cells, num_cells))
u_y = jnp.zeros((num_cells, num_cells, num_cells))
u_z = jnp.zeros((num_cells, num_cells, num_cells))
p = jnp.ones((num_cells, num_cells, num_cells)) * 0.1

num_injection_cells = 2
center = num_cells // 2

injection_slice = slice(center - num_injection_cells, center + num_injection_cells)

rho = rho.at[injection_slice, injection_slice, injection_slice].set(1.0)
p = p.at[injection_slice, injection_slice, injection_slice].set(1.0)

gamma = 5/3
dt = 0.0001
num_steps = 100

primitive_state = jnp.stack([rho, u_x, u_y, u_z, p], axis = 0)

final_state = time_integration(primitive_state, grid_spacing, num_steps, dt, gamma)

final_density = final_state[DENSITY_INDEX]

# plot the final density at the center of the domain
import matplotlib.pyplot as plt
plt.imshow(final_density[num_cells // 2, :, :])
plt.savefig("final_density.png")