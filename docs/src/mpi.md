# Working with MPI

If you are using [`PDVec`](@ref Main.DictVectors.PDVec)s to store your vectors, working with
MPI should be fairly straightforward. Generally, [`PDVec`](@ref Main.DictVectors.PDVec) will
work with MPI automatically, as long as MPI is set up correctly and a few common pitfalls
are avoided.

## Configuring MPI

When running on a cluster, ensure that MPI.jl is using the system binary. See [the MPI.jl
documentation](https://juliaparallel.org/MPI.jl/latest/configuration/) for more information.

It is always a good idea to start your script with a quick test that ensures the MPI is set up correctly. One way to do this is to open with

```julia
mpi_allprintln("hello")
```

which should print something like

```
[ rank 0: hello
[ rank 1: hello
[ rank 2: hello
[ rank 3: hello
```

If it prints `rank 0` several times, the code will run, but ranks will not communicate.

## Using Slurm

When using [`PDVec`](@ref Main.DictVectors.PDVec), the recommended setup is to use threads to parallelise the
computation within a node, and to only use MPI for inter-node communication. In a slurm
script, this is done as follows:

```bash
...
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4            # replace 4 with the desired number of nodes
#SBATCH --cpus-per-task=28   # replace 28 with the number of cores available in a node
#SBATCH --hint=nomultithread # don't use hyperthreading
...

srun julia --project -tauto script.jl
```

On some clusters, additional settings must be used with `srun`, for example the CTCP cluster
requires the following.

```bash
srun mpi=pmi2 julia --project -tauto script.jl
```

## Common pitfalls with reducing functions

### Using `@mpi_root`

Take care to not use reducing functions (such as `length`, `sum`, `norm`, ...) inside
[`@mpi_root`](@ref Main.Rimu.RMPI.@mpi_root) blocks. Doing so will only initiate the
distributed reduction on one rank only, which will cause the code to go out of sync and
freeze.

For example, the following code will cause MPI to freeze:

```julia
@mpi_root println("vector length is $(length(pdvec))")
```

To fix it, write it as:

```julia
len = length(pdvec)
@mpi_root println("vector length is $len")
```

## Anonymous functions

Suppose we want to scale a vector by its length:

```julia
map!(values(pdvec)) do x
	x / length(pdvec)
end
```

This will cause issues because `length` is a reduction. As `map!` is threaded many
threads will initiate MPI communication at the same time, which will probably cause a
crash. The correct way to rewrite this code is as

```julia
len = length(pdvec)
map!(values(pdvec)) do x
	x / len
end
```

This will not only save you from a crash, but will also be slightly more efficient as the
length is only calculated once. In this specific case, an even better option is to use the `scale!` function:

```julia
scale!(pdvec, 1 / length(pdvec))
```
