# tinyfluids

tinyfluids aims to contain a collection of small, simple 3D fluid simulators (first order Godunov schemes with HLL solver). Our goal is to find a strategy, to efficiently scale these simulations to multiple GPUs and later multiple nodes and to downstream these findings to our more advanced astrophysical magnetohydrodynamics + self gravity code [jf1uids](https://github.com/leo1200/jf1uids).

## jax_tinyfluids - the baseline

Our [baseline implementation](tinyfluids/jax_tinyfluids/jax_tinyfluids.py) in `JAX` scales to multiple GPUs (and nodes) automatically via `JAX`'s "parallelization follows data" approach and sharding of the initial conditions. We have also implemented a shard map parallelization strategy with halo exchange. Scaling strongly depends on the devices used, the scaling plot below was generated on 4 NVIDIA H100s. Scaling up to x2.6 was achieved for the baseline parallelization on H200 but as of busy computational ressources I could not test the scaling there for the shard-mapped parallelization strategy. My current guess is that just jitting has a pretty significant memory exchange overhead - I also tried to test this on lower-end GPUs (with slower inter-device communication) where scaling with shard map still worked ok but via only jitting, runtimes were usually longer than for a single GPU run. Looking at the JAXFLUIDS paper, they seem to get ideal speedups, while we only get up to ~3.5 here. One reason might be that the per shard computations are currently as simple as possible (1st order godunov) so communication overhead becomes more significant.

| ![Scaling Plot](figures/scaling_results.png) |
|:--------------------------------------------:|
| Scaling Plot                                 |

| ![Example Simulation](figures/check_1024.png)  |
|:----------------------------------------------:|
| Example Simulation with (1024)^3 Cells         |

## parallelization strategies

- the classical approach, as done in [JAXFLUIDS 2.0](https://arxiv.org/abs/2402.05193) would be a domain decomposition into patches with halo cells and halo exchange, efficient halo exchange is e.g. implemented in [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) based on NVIDIA's [cuDecomp](https://nvidia.github.io/cuDecomp/index.html)