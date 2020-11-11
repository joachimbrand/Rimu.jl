# Rimu

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://joachimbrand.github.io/Rimu.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://joachimbrand.github.io/Rimu.jl/dev/)

*Random Integrators for many-body quantum systems*

**Concept:** Joachim Brand and Elke Pahl.
**Contributors:** Joachim Brand, Elke Pahl, Mingrui Yang
Discussions, help, and additional contributions are acknowledged by Ali Alavi, Didier Adrien, Chris Scott (NeSI), Alexander Pletzer (NeSI).

To install the package clone the git repository with `git clone` to a convenient location, e.g.
`~/mygitpackagefolder/`. Then
hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```julia-repl
pkg> develop ~/mygitpackagefolder/rimu.jl
```
where the file path has to be adjusted to the location of the cloned git
repository.

**References:**
The code implements the FCIQMC algorithm described in
- "Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space", G. H. Booth, A. J. W. Thom, A. Alavi, [*J. Chem. Phys.* **131**, 054106 (2009)](https://doi.org/10.1063/1.3193710).

Scientific papers using the `Rimu` code:
- "Improved walker population control for full configuration interaction quantum Monte Carlo", M. Yang, E. Pahl, J. Brand, [*J. Chem. Phys.* **153**, 170143 (2020)](https://doi.org/10.1063/5.0023088); DOI: 10.1063/5.0023088; [arXiv:2008.01927](https://arxiv.org/abs/2008.01927).

For more information, consult the [documentation](https://joachimbrand.github.io/Rimu.jl/dev/).
