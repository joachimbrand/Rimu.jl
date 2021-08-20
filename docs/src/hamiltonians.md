## Module `Hamiltionians.jl`

This module contains definitions of Hamiltonians, in particular specific
physical models of interest. These are organised by means of an interface
around the abstract type [`AbstractHamiltonian`](@ref), in the spirit of the
`AbstractArray` interface as discussed in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/interfaces/).

```@docs
Hamiltonians
```
#### Relation to other parts of the `Rimu` code

In order to define a specific model Hamiltonian with relevant parameters
for the model, instantiate the model like this in the input file:

`ham = HubbardReal1D(BoseFS((1,2,0,3)); u=1.0, t=1.0)`

In the rest of the `Rimu` code, access to properties and matrix elements
of the model are then provided by the following methods:

 * `ham[address1, address2]`:  indexing of matrix elements (slow - use with caution)
 * `ham * dv`, `ham(dv::AbstractDVec)` or `mul!(dv1, ham, dv2)`: use as linear operator
 * [`diagonal_element(ham, add)`](@ref): diagonal matrix element
 * [`num_offdiagonals(ham, add)`](@ref): number of off-diagonals
 * [`get_offdiagonal(ham, add, chosen)`](@ref): access off-diagonal matrix element
 * [`offdiagonals(ham, add)`](@ref): iterator over off-diagonal matrix elements
 * [`random_offdiagonal(hops)`](@ref): choose random off-diagonal
 * [`dimension(T, ham)`](@ref): dimension of linear space
 * [`nearUniform(ham)`](@ref): configuration with particles spread across modes
 * [`starting_address(ham)`](@ref): address for accessing one of the diagonal elements of `ham`

### Model Hamiltonians

Here is a list of fully implemented model Hamiltonians. So far there are two variants
implemented of the one-dimensional Bose-Hubbard model real space as well as a momentum-space
Hubbard chain.

```@docs
HubbardReal1D
ExtendedHubbardReal1D
HubbardMom1D
BoseHubbardReal1D2C
BoseHubbardMom1D2C
HubbardRealSpace
MatrixHamiltonian
```

### Hamiltonians interface

Behind the implementation of a particular model is a more abstract interface for defining
hamiltonians. If you want to define a new model you should make use of this interface. The
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
```

### Geometry

```@docs
LatticeGeometry
PeriodicBoundaries
HardwallBoundaries
LadderBoundaries
num_neighbours
neighbour_site
```
