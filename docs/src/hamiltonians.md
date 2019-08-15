# Module Hamiltionians.jl

This module contains definitions of Hamiltonians, in particular specific
physical models of interest. These are organised by means of an interface
around the abstract type `LinearOperator`, in the spirit of the
`AbstractArray` interface as discussed in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/interfaces/).

```@docs
Hamiltonians
```
### Relation to other parts of the JuliaMC code

In order to define a specific model Hamiltonian with relevant parameters
for the model, instantiate the model like this in the input file:

`hamiltonian = BoseHubbardReal1D(n=6, m=6, u=1.0, t=1.0, AT = BSAdd64)`

In the rest of the JuliaMC code, access to properties and matrix elements
of the model are then provided by the following methods:
```
hamiltonian[address1, address2]
diagME(hamiltonian, add)
numOfHops(hamiltonian, add)
hop(hamiltonian, add, chosen)
Hops(hamiltonian, add)
generateRandHop(hamiltonian, add), generateRandHop(hops::Hops)
bit_String_Length(hamiltonian)
hasIntDimension(hamiltonian)
dimensionLO(hamiltonian)
fDimensionLO(hamiltonian)
generateInitialWalker(hamiltonian, num, WT)
```

## Model Hamiltonians

Here is a list of fully implemented model Hamiltonians. So far there are two
variants of the Bose-Hubbard model in one dimension implemented in real space.

```@docs
BoseHubbardReal1D
ExtendedBHReal1D
```


## Hamiltonians interface

Behind the implementation of a particular model is a more abstract interface
for defining hamiltonians.
If you want to define a new model you should make use of this interface.
The most general form of a model Hamiltonian should subtype to
`LinearOperator` and implement the relevant methods.

```@docs
LinearOperator
Hops
generateRandHop
```

### BosonicHamiltonian

For a many-body system consisting of spinless bosons, we already know more
about the structure of the problem. `BosonicHamiltonian` is a pre-defined
subtype of `LinearOperator`.

```@docs
BosonicHamiltonian
hasIntDimension
dimensionLO
fDimensionLO
bit_String_Length
generateInitialWalker
```

### Core functions

The following functions are part of the core functionality of a Hamiltonian and
need to be implemented efficiently and specifically for each model.

```@docs
numOfHops
hop
diagME
```
