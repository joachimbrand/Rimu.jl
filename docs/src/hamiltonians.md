# Module `Hamiltonians`

This module contains definitions of Hamiltonians, in particular specific
physical models of interest. These are organised by means of an interface
around the abstract type [`AbstractHamiltonian`](@ref), in the spirit of the
`AbstractArray` interface as discussed in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/interfaces/).

The Hamiltonians can be used for projector quantum Monte Carlo with [`ProjectorMonteCarloProblem`](@ref) or for exact diagonalization with [`ExactDiagonalizationProblem`](@ref), see [Exact Diagonalization](@ref).

```@docs
Hamiltonians
```


## Model Hamiltonians

Here is a list of fully implemented model Hamiltonians. There are several variants
of the Hubbard model in real and momentum space, as well as some other models.

### Real space Hubbard models
```@docs
HubbardReal1D
BoseHubbardReal1D2C
HubbardReal1DEP
HubbardRealSpace
ExtendedHubbardReal1D
```

### Momentum space Hubbard models
```@docs
HubbardMom1D
BoseHubbardMom1D2C
HubbardMom1DEP
```

### Harmonic oscillator models
```@docs
HOCartesianContactInteractions
HOCartesianEnergyConservedPerDim
HOCartesianCentralImpurity
```

### Other
```@docs
MatrixHamiltonian
Transcorrelated1D
FroehlichPolaron
```

### Convenience functions
```@docs
rayleigh_quotient
momentum
hubbard_dispersion
continuum_dispersion
shift_lattice
shift_lattice_inv
```

## Hamiltonian wrappers
The following Hamiltonians are constructed from an existing
Hamiltonian instance and change its behaviour:
```@docs
GutzwillerSampling
GuidingVectorSampling
ParitySymmetry
TimeReversalSymmetry
Stoquastic
```

## Observables
Observables are [`AbstractHamiltonian`](@ref)s that represent a physical
observable. Their ground state expectation values can be sampled by passing
them into [`AllOverlaps`](@ref).
```@docs
ParticleNumberOperator
G2RealCorrelator
G2RealSpace
G2MomCorrelator
SuperfluidCorrelator
StringCorrelator
DensityMatrixDiagonal
SingleParticleExcitation
TwoParticleExcitation
Momentum
AxialAngularMomentumHO
```

## Hamiltonians interface

Behind the implementation of a particular model is a more abstract interface for defining
Hamiltonians. If you want to define a new model you should make use of this interface. The
most general form of a model Hamiltonian should subtype to `AbstractHamiltonian` and
implement the relevant methods.

```@docs
AbstractHamiltonian
offdiagonals
diagonal_element
starting_address
```

The following functions may be implemented instead of [`offdiagonals`](@ref).

```@docs
num_offdiagonals
get_offdiagonal
```

The following functions come with default implementations, but may be customized.

```@docs
random_offdiagonal
Hamiltonians.LOStructure
dimension
has_adjoint
allows_address_type
```

This interface relies on unexported functionality, including
```@docs
Hamiltonians.adjoint
Hamiltonians.dot
Hamiltonians.AbstractOffdiagonals
Hamiltonians.Offdiagonals
Hamiltonians.check_address_type
Hamiltonians.number_conserving_dimension
Hamiltonians.number_conserving_bose_dimension
Hamiltonians.number_conserving_fermi_dimension
```

## Geometry

Lattices in higher dimensions are defined here for [`HubbardRealSpace`](@ref) and [`G2RealSpace`](@ref).

```@docs
CubicGrid
Hamiltonians.Directions
Hamiltonians.Displacements
Hamiltonians.neighbor_site
PeriodicBoundaries
HardwallBoundaries
LadderBoundaries
```

## Harmonic Oscillator
Useful utilities for harmonic oscillator in Cartesian basis, see [`HOCartesianContactInteractions`](@ref)
and [`HOCartesianEnergyConservedPerDim`](@ref).
```@docs
get_all_blocks
fock_to_cart
```
Underlying integrals for the interaction matrix elements are implemented in the following unexported functions
```@docs
Hamiltonians.four_oscillator_integral_general
Hamiltonians.ho_delta_potential
Hamiltonians.log_abs_oscillator_zero
```

## Index
```@index
Pages   = ["hamiltonians.md"]
```
