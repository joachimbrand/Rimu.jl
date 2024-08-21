# Rimu.jl Package Guide

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be represented by a Hamiltonian in second quantisation language. Currently supported features include:
### Interacting with quantum many-body models
* **Full configuration interaction quantum Monte Carlo (FCIQMC)**, a flavour of projector quantum Monte Carlo for stochastically solving the time-independent Schrödinger equation. See [References](@ref).
* **Matrix-free exact diagonalisation** of quantum Hamiltonians (with external package [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl)).
* **Sparse matrix representation** of quantum Hamiltonians for exact diagonalisation with sparse linear algebra package of your choice (fastest for small systems).

### Representing quantum many-body models
* A composable and efficient type system for representing single- and multi-component **Fock states** of bosons, fermions, and mixtures thereof, to be used as a basis for representing Hamiltonians.
* An **interface for defining many-body Hamiltonians**.
* Pre-defined models include:
  * **Hubbard model** in real space for bosons and fermions and mixtures in 1, 2, and 3 spatial dimensions.
  * Hubbard and related **lattice models in momentum space** for bosons and fermions in one spatial dimension.
  * **Transcorrelated Hamiltonian** for contact interactions in one dimension for fermions, as described in Jeszenski *et al.* [arXiv:1806.11268](http://arxiv.org/abs/1806.11268).

### Statistical analysis of Monte Carlo data
* **Blocking analysis** following Flyvberg & Peterson [JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480), and automated with hypothesis testing by Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304).
* **Unbiased estimators** for the ground state energy by re-reweighting following Nightingale & Blöte [PRB (1986)](https://link.aps.org/doi/10.1103/PhysRevB.33.659) and Umrigar *et al.* [JCP (1993)](http://aip.scitation.org/doi/10.1063/1.465195).

The code supports parallelisation with MPI (harnessing [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl)) as well as native Julia threading (experimental). In the future, we may add tools to solve the time-dependent Schrödinger equation and Master equations for open system time evolution.

**Concept:** Joachim Brand and Elke Pahl.

**Contributors:** Joachim Brand, Elke Pahl, Mingrui Yang, Matija Čufar, Chris Bradly.

Discussions, help, and additional contributions are acknowledged by Ali Alavi,
Didier Adrien, Chris Scott (NeSI), Alexander Pletzer (NeSI).


## Installation

### Installing `Rimu` for usage

`Rimu` is a registered package and can be installed with the package manager.
Hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```
pkg> add Rimu
```
Alternatively, use
```julia-repl
julia> using Pkg; Pkg.add(name="Rimu")
```
in order to install `Rimu` from a script.

### Installing `Rimu` for development

In order to be able to edit the source code, push changes, change and make new git branches,
etc.,
clone the git repository with `git clone` to a convenient location, e.g.
`~/mygitpackagefolder/`. Then
hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```
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
as well as algorithms to find the ground state, e.g. [`ProjectorMonteCarloProblem`](@ref), [`ExactDiagonalizationProblem`](@ref).

## Scripts

`Rimu` is written as a Julia package to be imported with `using Rimu` as described
above. It supplies useful
functions and types. Performing actual calculations and analysing the results
is done with scripts. The folder `scripts/` contains a collections of scripts
that are either examples for use of the Rimu package or useful scripts for
data analysis. In particular:

- `scripts/BHM-example.jl` is an example script that runs fciqmc on the 1D Bose-Hubbard model. A data frame with results is written to the file `fciqmcdata.arrow`.
- `scripts/BHM-example-mpi.jl` demonstrates basic usage of `Rimu` with MPI.

## MPI

The Rimu package can run in parallel on different processes or node and
distribute work by making use of MPI, or "message passing interface". For example, running
```
> julia scripts/BHM-example.jl
```
will run on one processor with the main computation (i.e. after
package loading and compilation) completing in 2.69 seconds.

Running
```
> mpirun -np 4 julia scripts/BHM-example-mpi.jl
```
on the same hardware makes use of 4 cores and the main part completes in 1.04
seconds, a speedup factor of 2.6. This seems reasonable, given that extra work
needs to be done for communicating between different processes.

Using MPI parallelism with `Rimu` is easy. Enabling MPI enabled automatically if
[`PDVec`](@ref) is used to store a vector. In that case, data will be stored in a
distributed fashion among the MPI ranks and only communicated between ranks when
necessary.

## Compatibility

We recommend using `Rimu` with the latest Julia release version. Rimu requires at least julia `v1.9`.

## References
The code implements the FCIQMC algorithm originally described in
- "Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space", G. H. Booth, A. J. W. Thom, A. Alavi, [*J. Chem. Phys.* **131**, 054106 (2009)](https://doi.org/10.1063/1.3193710).
-  "Communications: Survival of the fittest: accelerating convergence in full configuration-interaction quantum Monte Carlo.", D. Cleland,  G. H. Booth, A. Alavi, [*J. Chem. Phys.* **132**, 041103 (2010)](https://doi.org/10.1063/1.3302277).

Scientific papers describing additional features implemented in `Rimu`:
- "Improved walker population control for full configuration interaction quantum Monte Carlo", M. Yang, E. Pahl, J. Brand, [*J. Chem. Phys.* **153**, 170143 (2020)](https://doi.org/10.1063/5.0023088); [arXiv:2008.01927](https://arxiv.org/abs/2008.01927).
- "Stochastic differential equation approach to understanding the population control bias in full configuration interaction quantum Monte Carlo", J. Brand, M. Yang, E. Pahl, [arXiv:2103.07800](http://arxiv.org/abs/2103.07800) (2021).

Papers discussing results obtained with `Rimu`:
- "Polaron-Depleton Transition in the Yrast Excitations of a One-Dimensional Bose Gas with a Mobile Impurity", M. Yang, M. Čufar, E. Pahl, J. Brand, [*Condens. Matter* **7**, 15 (2022)](https://www.mdpi.com/2410-3896/7/1/15).
- "Magnetic impurity in a one-dimensional few-fermion system", L. Rammelmüller, D. Huber, M. Čufar, J. Brand, A. Volosniev, [arXiv:2204.01606](http://arxiv.org/abs/2204.01606) (2022).
