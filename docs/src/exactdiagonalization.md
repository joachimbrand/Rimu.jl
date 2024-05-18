# Exact Diagonalization

The main functionality of Rimu for exact diagonalization is contained in the module `ExactDiagonalization`.

```@docs
ExactDiagonalization
```

## `ExactDiagonalizationProblem`

```@docs
ExactDiagonalizationProblem
solve(::ExactDiagonalizationProblem)
init(::ExactDiagonalizationProblem)
```

## Solver algorithms

```@docs
KrylovKitSolver
LinearAlgebraSolver
ArpackSolver
LOBPCGSolver
```

## Converting a Hamiltonian in to a matrix

```@docs
BasisSetRepresentation
build_basis
Matrix
sparse
```

## Deprecated
```@docs
BasisSetRep
```


