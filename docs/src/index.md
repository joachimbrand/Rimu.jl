# Rimu.jl Package Guide

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be
represented by a Hamiltonian in second quantisation language.
Currently there
are tools to find the ground state with a Lanczos algorithm
(using [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) for small Hilbert spaces), or with projector
quantum Monte Carlo in the flavour of FCIQMC (see [References](@ref)).
We will add tools to solve the
time-dependent Schrödinger equation and Master equations for open system
time evolution.


## Installation

### Installing `Rimu` for usage

`Rimu` can be installed with the package manager directly from the github
repository. Hit the `]` key at the Julia REPL to get into `Pkg` mode and
type
```julia-repl
pkg> add https://github.com/joachimbrand/Rimu.jl
```
Alternatively, use
```julia-repl
julia> using Pkg; Pkg.add(PackageSpec(url="https://github.com/joachimbrand/Rimu.jl"))
```
in order to install `Rimu` from a script.

### Installing `Rimu` for development

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

## Scripts

Rimu is written as a Julia package to be imported with `using Rimu` as described
above. It supplies useful
functions and types. Performing actual calculations and analysing the results
is done with scripts. The folder `scripts/` contains a collections of scripts
that are either examples for use of the Rimu package or useful scripts for
data analysis. In particular:

- `scripts/BHM-example.jl` is an example script that runs fciqmc on the 1D Bose-Hubbard model. A data frame with results is written to the file `fciqmcdata.arrow`.
- `test/script_mpi_minimum_ptp.jl` demonstrates basic usage of `Rimu` with MPI.

## MPI

The Rimu package can run in parallel on different processes or node and
distribute work by making use of MPI, or "message passing interface". For example, running
```
> julia scripts/BHM-example.jl
```
will run on one processor with the main `lomc!()` computation (i.e. after
package loading and compilation) completing in 2.69 seconds.

Running
```
> mpirun -np 4 julia scripts/BHM-example-mpi.jl
```
on the same hardware makes use of 4 cores and the main part completes in 1.04
seconds, a speedup factor of 2.6. This seems reasonable, given that extra work
needs to be done for communicating between different processes.

Initialising and finalising MPI communication has to be handled at the script level. Enabling MPI communication for use in [`lomc!()`](@ref) is done by wrapping the primary data structures as [`MPIData`](@ref). The relevant functionality is
provided by the module `Rimu.RMPI`.

See examples in the [Scripts](@ref) folder.

## References
The code implements the FCIQMC algorithm described in
- "Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space", G. H. Booth, A. J. W. Thom, A. Alavi, [*J. Chem. Phys.* **131**, 054106 (2009)](https://doi.org/10.1063/1.3193710).

Scientific papers using the `Rimu` code:
- "Improved walker population control for full configuration interaction quantum Monte Carlo", M. Yang, E. Pahl, J. Brand, [*J. Chem. Phys.* **153**, 170143 (2020)](https://doi.org/10.1063/5.0023088); DOI: 10.1063/5.0023088; [arXiv:2008.01927](https://arxiv.org/abs/2008.01927).
- "Stochastic differential equation approach to understanding the population control bias in full configuration interaction quantum Monte Carlo", J. Brand, M. Yang, E. Pahl; [arXiv:2103.07800](http://arxiv.org/abs/2103.07800) (2021).
