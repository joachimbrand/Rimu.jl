# Module `DictVectors`

```@docs
DictVectors
AbstractDVec
```

## Concrete implementations

```@docs
DVec
InitiatorDVec
```

## Interface functions

```@docs
deposit!
storage
freeze
localpart
fciqmc_step!
sort_into_targets!
working_memory!
```

## Supported operations

[`AbstractDVec`](@ref)s generally support most operations that are defined on `Vector`s and
`Dict`s. This includes common linear algebra operations such as `dot` or `norm`.

A significant difference between [`AbstractDVec`](@ref)s, `Vector`s, and `Dict`s, is that
iteration on them is disabled by default. Iteration must be explicitly performed on `keys`,
`calues`, or `pairs`, however, it is highly recommended you use `mapreduce`, `reduce`, or
similar functions when performing reductions, as that will make the operations compatible
with MPI.

In addition, Rimu defines the following functions.

```@docs
zero!
add!
walkernumber
```

## Projectors

```@docs
AbstractProjector
NormProjector
Norm2Projector
UniformProjector
Norm1ProjectorPPop
```

## Initiator rules

```@docs
Rimu.DictVectors.InitiatorRule
Rimu.DictVectors.InitiatorValue
Rimu.DictVectors.value
Rimu.DictVectors.Initiator
Rimu.DictVectors.SimpleInitiator
Rimu.DictVectors.CoherentInitiator
```

## Index
```@index
Pages   = ["dictvectors.md"]
```
