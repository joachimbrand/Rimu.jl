# Rimu.jl

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be
represented by a Hamiltonian in second quantisation language. Currently there
are tools to find the ground state with FCIQMC or with a Lanczos algorithm
(using KrylovKit for small Hilbert spaces). We will add tools to solve the
time-dependent Schrödinger equation and Master equations for open system
time evolution.

# Installation

## Install `Rimu` for running jobs only

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

## Install `Rimu` for development

In order to be able to edit the source code, push changes, change and make new git branches,
etc.,
clone the git repository with `git clone` to a convenient location, e.g.
`~/mygitpackagefolder/`. Then
hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```julia-repl
pkg> develop ~/mygitpackagefolder/Rimu
```
where the file path has to be adjusted to the location of the cloned git
repository.

# Usage

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
as well as algorithms to find the ground state, e.g. [`fciqmc!`](@ref).


# Examples

Use the side panel to navigate to an example script.
