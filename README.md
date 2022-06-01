# Rimu

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://joachimbrand.github.io/Rimu.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://joachimbrand.github.io/Rimu.jl/dev/)
[![Coverage Status](https://coveralls.io/repos/github/joachimbrand/Rimu.jl/badge.svg)](https://coveralls.io/github/joachimbrand/Rimu.jl)

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be represented by a Hamiltonian in second quantisation language. Currently supported features include:
### Interacting with quantum many-body models
* **Full configuration interaction quantum Monte Carlo (FCIQMC)**, a flavour of projector quantum Monte Carlo for stochastically solving the time-independent Schrödinger equation.
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

**Contributors:** Joachim Brand, Elke Pahl, Mingrui Yang, Matija Cufar, Chris Bradly.

Discussions, help, and additional contributions are acknowledged by Ali Alavi,
Didier Adrien, Chris Scott (NeSI), Alexander Pletzer (NeSI).

### Installing Rimu

`Rimu` is a registered package and can be installed with the package manager.
Hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```julia-repl
pkg> add Rimu
```
Alternatively, use
```julia-repl
julia> using Pkg; Pkg.add(name="Rimu")
```
in order to install `Rimu` from a script.

### Usage

The package is now installed and can be imported with
```julia-repl
julia> using Rimu
```

Note that `Rimu` is under active development and breaking changes to the user interface may occur at any time. We encourage potential users of the package to contact the authors for efficient communication.


### References
The code implements the FCIQMC algorithm described in
- "Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space", G. H. Booth, A. J. W. Thom, A. Alavi, [*J. Chem. Phys.* **131**, 054106 (2009)](https://doi.org/10.1063/1.3193710).
-  "Communications: Survival of the fittest: accelerating convergence in full configuration-interaction quantum Monte Carlo.", D. Cleland,  G. H. Booth, A. Alavi, [*J. Chem. Phys.* **132**, 041103 (2010)](https://doi.org/10.1063/1.3302277).

Scientific papers describing additional features implemented in `Rimu`:
- "Improved walker population control for full configuration interaction quantum Monte Carlo", M. Yang, E. Pahl, J. Brand, [*J. Chem. Phys.* **153**, 170143 (2020)](https://doi.org/10.1063/5.0023088); [arXiv:2008.01927](https://arxiv.org/abs/2008.01927).
- "Stochastic differential equation approach to understanding the population control bias in full configuration interaction quantum Monte Carlo", J. Brand, M. Yang, E. Pahl, [arXiv:2103.07800](http://arxiv.org/abs/2103.07800) (2021).

Papers discussing results obtained with `Rimu`:
- "Polaron-Depleton Transition in the Yrast Excitations of a One-Dimensional Bose Gas with a Mobile Impurity", M. Yang, M. Čufar, E. Pahl, J. Brand, [*Condens. Matter* **7**, 15 (2022)](https://www.mdpi.com/2410-3896/7/1/15).
- "Magnetic impurity in a one-dimensional few-fermion system", L. Rammelmüller, D. Huber, M. Čufar, J. Brand, A. Volosniev, [arXiv:2204.01606](http://arxiv.org/abs/2204.01606) (2022).

For more information, consult the [documentation](https://joachimbrand.github.io/Rimu.jl/dev/).
