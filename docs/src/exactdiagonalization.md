# Exact Diagonalization

The main functionality of Rimu for exact diagonalization is contained in the module `ExactDiagonalization`.

```@docs
ExactDiagonalization
```

<!-- ## Usage with FCIQMC and exact diagonalisation

In order to define a specific model Hamiltonian with relevant parameters
for the model, instantiate the model like this in the input file:

```julia-repl
hubb = HubbardReal1D(BoseFS((1,2,0,3)); u=1.0, t=1.0)
```

The Hamiltonian `hubb` is now ready to be used for FCIQMC in [`lomc!`](@ref)
and for exact diagonalisation with [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) directly, or after
transforming into a sparse matrix first with
```julia-repl
using SparseArrays
sh = sparse(hubb)
```
or into a full matrix with
```julia-repl
using LinearAlgebra
fh = Matrix(hubb)
```
This functionality relies on
```@docs
ExactDiagonalization.BasisSetRepresentation
sparse
Matrix
```
If only the basis is required and not the matrix representation it is more efficient to use
```@docs
ExactDiagonalization.build_basis
``` -->

## `ExactDiagonalizationProblem`

```@docs
ExactDiagonalizationProblem
```

## Solver algorithms

```@docs
KrylovKitSolver
LinearAlgebraSolver
ArpackSolver
LOBPCGSolver
```

## Underlying functionality

```@docs
BasisSetRepresentation
build_basis
Matrix
sparse
```


