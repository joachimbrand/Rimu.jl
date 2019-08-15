# Random Numbers

Generating good quality random numbers is crucial for any Monte Carlo code. In addition to generating pseudo random numbers with good statistics and in minimal CPU time, we also have the requirement that computations should be reproducible and the random number sequences independent on each worker when the code runs in parallel mode.

We define the random number generator in the module `ConsistentRNG.jl`, which is loaded onto each process. Furthermore, independent seeds are used to seed the RNGs on each worker (from `goQMC.jl`). These seeds are generated using the `Random.RandomDevice` random number generator, which draws entropy from the operating system / hardware. The seeds are saved to file with a filename that includes the number of processes used. If a suitable file is found, then seeds are read in from the file. This behaviour can be controlled by the flag `reuseRandomSeeds` in the input file.

For the random number generator we are currently using
'Xoroshiro128Plus' from 'RandomNumbers.jl'. For benchmarks and statistical test results see the
[Documentation of `RandomNumbers.jl`](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1).

## Module `ConsistentRNG.jl`

```@autodocs
Modules = [ConsistentRNG]
```
