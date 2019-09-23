# Running on NeSI with MPI

This folder contains an example Julia script `ros_BHM_M50_U6_W1M.jl` that will work with MPI and a corresponding Slurm submit script `submit.sl`.

Before running, make sure to `Pkg.develop()` `Rimu` and `Pkg.add()` the following additional packages: `MPI`, `Feather`, `Humanize`. It may also be useful to issue
```julia-repl
julia> using Rimu, MPI, Feather, Humanize
```
on the Julia prompt before attempting to run the code in order to precompile the packages.

The example runs for about 8 minutes with 4000 time steps and an example output file is also contained in the folder. The run time can be scaled up arbitrarily by modifying the number after `timesteps` on line 25 of the Julia script.
