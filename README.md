# Rimu

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://joachimbrand.github.io/Rimu.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://joachimbrand.github.io/Rimu.jl/dev/)

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be
represented by a Hamiltonian in second quantisation language. Currently
there
are tools to find the ground state with a Lanczos algorithm
(using [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) for small Hilbert spaces), or with projector
quantum Monte Carlo in the flavour of FCIQMC (see [References](@ref)).
Later, we may add tools to solve the
time-dependent Schrödinger equation and Master equations for open system
time evolution.

**Concept:** Joachim Brand and Elke Pahl.

**Contributors:** Joachim Brand, Elke Pahl, Mingrui Yang, Matija Cufar.

Discussions, help, and additional contributions are acknowledged by Ali Alavi,
Didier Adrien, Chris Scott (NeSI), Alexander Pletzer (NeSI).

### Installing Rimu

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

### Usage

The package is now installed and can be imported with
```julia-repl
julia> using Rimu
```
`Rimu` offers a number of tools for representing Hamiltonians and
state vectors / wave functions
as well as algorithms for finding the ground state.


### References
The code implements the FCIQMC algorithm described in
- "Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space", G. H. Booth, A. J. W. Thom, A. Alavi, [*J. Chem. Phys.* **131**, 054106 (2009)](https://doi.org/10.1063/1.3193710).

Scientific papers using the `Rimu` code:
- "Stochastic differential equation approach to understanding the population control bias in full configuration interaction quantum Monte Carlo", J. Brand, M. Yang, E. Pahl, [arXiv:2103.07800](http://arxiv.org/abs/2103.07800) (2021).
- "Improved walker population control for full configuration interaction quantum Monte Carlo", M. Yang, E. Pahl, J. Brand, [*J. Chem. Phys.* **153**, 170143 (2020)](https://doi.org/10.1063/5.0023088); DOI: 10.1063/5.0023088; [arXiv:2008.01927](https://arxiv.org/abs/2008.01927).

For more information, consult the [documentation](https://joachimbrand.github.io/Rimu.jl/dev/).
