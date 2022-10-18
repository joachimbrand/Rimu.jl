# Random numbers in Rimu

Rimu uses Julia's built-in random number generator, which currently defaults to
[Xoshiro256++](https://docs.julialang.org/en/v1/stdlib/Random/#Random.Xoshiro).
There is currently no official way to change the RNG.

## Reproducibility

If you want FCIQMC runs to be reproducible, make sure to seed the RNG with
[Random.seed!](https://docs.julialang.org/en/v1/stdlib/Random/#Random.seed!) and to use [`lomc!`](@ref) in single-threaded mode by passing it the `threading=false` keyword argument.

MPI-distrubted runs can also be made reproducible by seeding the RNG with [mpi_seed!](@ref).
