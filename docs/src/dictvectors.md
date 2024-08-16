# Module `DictVectors`

```@docs
DictVectors
AbstractDVec
```

## Concrete implementations

```@docs
DVec
InitiatorDVec
PDVec
```

## Interface functions

```@docs
deposit!
storage
freeze
localpart
apply_operator!
sort_into_targets!
working_memory
```

## Supported operations

[`AbstractDVec`](@ref)s generally support most operations that are defined on `Vector`s and
`Dict`s. This includes the interface from
[VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl), and many functions from
the LinearAlgebra standard library.

A significant difference between [`AbstractDVec`](@ref)s, `Vector`s, and `Dict`s, is that
iteration on them is disabled by default. Iteration must be explicitly performed on `keys`,
`values`, or `pairs`, however, it is highly recommended you use `mapreduce`, `reduce`, or
similar functions when performing reductions, as that will make the operations compatible
with MPI.

In addition, Rimu defines the following function.

```@docs
walkernumber
walkernumber_and_length
dot_from_right
```

## Projectors

```@docs
AbstractProjector
NormProjector
Norm2Projector
UniformProjector
Norm1ProjectorPPop
Rimu.DictVectors.FrozenDVec
Rimu.DictVectors.FrozenPDVec
```

## Initiator rules

```@docs
Rimu.DictVectors.InitiatorRule
Rimu.DictVectors.AbstractInitiatorValue
Rimu.DictVectors.InitiatorValue
Rimu.DictVectors.initiator_valtype
Rimu.DictVectors.to_initiator_value
Rimu.DictVectors.from_initiator_value
Rimu.DictVectors.Initiator
Rimu.DictVectors.SimpleInitiator
Rimu.DictVectors.CoherentInitiator
Rimu.DictVectors.NonInitiator
Rimu.DictVectors.NonInitiatorValue
```

## `PDVec` internals

### Working memory

```@autodocs
Modules = [DictVectors]
Pages = ["pdworkingmemory.jl"]
```

### Communicators

```@autodocs
Modules = [DictVectors]
Pages = ["communicators.jl"]
```

## Index
```@index
Pages   = ["dictvectors.md"]
```
