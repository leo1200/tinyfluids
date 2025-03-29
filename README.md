# tinyfluids

tinyfluids aims to contain a collection of small, simple 3D fluid simulators (first order Godunov schemes with HLL solver). Our goal is to find a strategy, to efficiently scale these simulations to multiple GPUs and later multiple nodes and to downstream these findings to our more advanced astrophysical magnetohydrodynamics + self gravity code [jf1uids](https://github.com/leo1200/jf1uids).

## jax_tinyfluids - the baseline

Our [baseline implementation](tinyfluids/jax_tinyfluids/jax_tinyfluids.py) in `JAX` scales to multiple GPUs (and nodes) automatically via `JAX`'s "parallelization follows data" approach and sharding of the initial conditions. The maximum speedup on 4 H200 GPUs compared to 1, however, seems to be around x2.6 (the optimal would be x4), a scaling plot is shown below.

| ![Scaling Plot](figures/scaling_results.png) |
|:--------------------------------------------:|
| Scaling Plot                                 |

| ![Example Simulation](figures/check_1024.png)  |
|:----------------------------------------------:|
| Example Simulation with (1024)^3 Cells         |

## parallelization strategies

- the classical approach, as done in [JAXFLUIDS 2.0](https://arxiv.org/abs/2402.05193) would be a domain decomposition into patches with halo cells and halo exchange, efficient halo exchange is e.g. implemented in [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) based on NVIDIA's [cuDecomp](https://nvidia.github.io/cuDecomp/index.html)