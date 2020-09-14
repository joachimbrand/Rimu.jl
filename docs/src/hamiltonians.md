# Developer Documentation

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

`ham = BoseHubbardReal1D(n=6, m=6, u=1.0, t=1.0, AT = BSAdd64)`

In the rest of the `Rimu` code, access to properties and matrix elements
of the model are then provided by the following methods:

 * `ham[address1, address2]`:  indexing of matrix elements (slow - use with caution)
 * `ham(dv::AbstractDVec)` or `mul!(dv1, ham, dv2)`: use as linear operator
 * [`diagME(ham, add)`](@ref): diagonal matrix element
 * [`numOfHops(ham, add)`](@ref): number of off-diagonals
 * [`hop(ham, add, chosen)`](@ref): access off-diagonal matrix element
 * [`Hops(ham, add)`](@ref): iterator over off-diagonal matrix elements
 * [`generateRandHop(hops::Hops)`](@ref): choose random off-diagonal
 * [`bit_String_Length(ham)`](@ref): number of bits in the configuration
 * `ham(:dim)` and `ham(:fdim)`: dimension of linear space. See  [`hasIntDimension(ham)`](@ref), [`dimensionLO(ham)`](@ref), [`fDimensionLO(ham)`](@ref)
 * [`nearUniform(ham)`](@ref): configuration with particles spread across modes


### Model Hamiltonians

Here is a list of fully implemented model Hamiltonians. So far there are two
variants implemented of the Bose-Hubbard model in one dimensional in real space.

```@docs
BoseHubbardReal1D
ExtendedBHReal1D
```


### Hamiltonians interface

Behind the implementation of a particular model is a more abstract interface
for defining hamiltonians.
If you want to define a new model you should make use of this interface.
The most general form of a model Hamiltonian should subtype to
`AbstractHamiltonian` and implement the relevant methods.

```@docs
AbstractHamiltonian
Hops
generateRandHop
```
#### Core functions

The following functions are part of the core functionality of a Hamiltonian and
need to be implemented efficiently and specifically for each model.

```@docs
numOfHops
hop
diagME
```

#### BosonicHamiltonian

For a many-body system consisting of spinless bosons, we already know more
about the structure of the problem. `BosonicHamiltonian` is a pre-defined
subtype of `AbstractHamiltonian`.

```@docs
BosonicHamiltonian
hasIntDimension
dimensionLO
fDimensionLO
bit_String_Length
nearUniform
```
