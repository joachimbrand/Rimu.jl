# Rimu.jl Package Guide

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be
represented by a Hamiltonian in second quantisation language. Currently there
are tools to find the ground state with FCIQMC or with a Lanczos algorithm
(using KrylovKit for small Hilbert spaces). We will add tools to solve the
time-dependent Schrödinger equation and Master equations for open system
time evolution.

## Contents
```@contents
Pages = ["index.md","hamiltonians.md","consistentrng.md","documentation.md",
            "testing.md","API.md","BHM-example.md"]
Depth = 4
```

## Installation

### Install `Rimu` for running jobs only

`Rimu` can be installed with the package manager directly from the bitbucket
repository. Either hit the `]` key at the Julia REPL to get into `Pkg` mode and
type
```julia-repl
pkg> add https://joachimbrand@bitbucket.org/joachimbrand/rimu.jl#master
```
where `master` can be exchanged with the name of the desired git branch.
Alternatively, use
```julia-repl
julia> using Pkg; Pkg.add("https://joachimbrand@bitbucket.org/joachimbrand/rimu.jl#master")
```

### Install `Rimu` for development

In order to be able to edit the source code, push changes, change and make new git branches,
etc.,
clone the git repository with `git clone` to a convenient location, e.g.
`~/mygitpackagefolder/`. Then
hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```julia-repl
pkg> develop ~/mygitpackagefolder/rimu.jl
```
where the file path has to be adjusted to the location of the cloned git
repository.

## Usage

The package is now installed and can be imported with
```julia-repl
julia> using Rimu
```

When planning to edit the code of the package it is advisable to use the
`Revise` package by issuing
```julia-repl
julia> using Revise
```
**before** `using Rimu`. This will track any changes made to the source code of
`Rimu` and the changed package will be available after saving the source code
(hopefully, and most of the time, without restarting the Julia REPL).

Rimu offers a number of tools for representing Hamiltonians (see
[`Hamiltonians`](@ref)) and state vectors / wave functions
(see [`DictVectors`](@ref))
as well as algorithms to find the ground state, e.g. [`lomc!`](@ref).

## Scripts (New)

Rimu is written as a Julia package to be imported with `using Rimu` as described
above. It supplies useful
functions and types. Performing actual calculations and analysing the results
is done with scripts. The folder `scripts/` contains a collections of scripts
that are either examples for use of the Rimu package or useful scripts for
data analysis. In particular:

- `scripts/BHM-example.jl` is an example script that runs fciqmc on the 1D Bose-Hubbard model. A data frame with results is written to the file `fciqmcdata.feather`.
- `scripts/BHM-example-mpi.jl` is an example script that runs the same fciqmc calculation as above with MPI enabled.

## Scripts

Rimu is written as a Julia package to be imported with `using Rimu` as described
above. It supplies useful
functions and types. Performing actual calculations and analysing the results
is done with scripts. The folder `scripts/` contains a collections of scripts
that are either examples for use of the Rimu package or useful scripts for
data analysis. In particular:

- `scripts/ros-BHM_M50_U6-W10k.jl` is an example script that runs fciqmc on the 1D Bose-Hubbard model with MPI enabled. A data frame with results is written to the file `fciqmcdata.feather`.
- `scripts/read_file_and_plot.jl` reads the feather file (from the working directory) and displays basic plots and blocking analysis of the shift.
- `plotting.jl` is a collection of (currently very primitive) plotting function. On purpose these are not part of the Rimu package in order to avoid a dependency on a plotting package.

## MPI

The Rimu package can run in parallel on different processes or node and
distribute work by making use of MPI, or "message passing interface". For example,
on my laptop, running
```
> julia scripts/ros-BHM_M50_U6-W10k.jl
```
will run on one processor with the main `fciqmc!()` computation (i.e. after
package loading and compilation) completing in 18 seconds.

Running
```
> mpirun -np 4 julia scripts/ros-BHM_M50_U6-W10k.jl
```
on the same hardware makes use of 4 cores and the main part completes in 5.6
seconds, a speedup factor of 3.1. This seems reasonable, given that extra work
needs to be done for communicating between different processes.

Initialising and finalising MPI communication has to be handled at the script level. Enabling MPI communication for use in [`fciqmc!()`](@ref) is done by wrapping the primary data structures as [`MPIData`](@ref). A number of different strategies
for data communication are implemented and most easily accessed
with the functions:

- [`mpi_default`](@ref)
- [`mpi_one_sided`](@ref)
- [`mpi_no_exchange`](@ref)

See examples in the [Scripts](@ref) folder.
