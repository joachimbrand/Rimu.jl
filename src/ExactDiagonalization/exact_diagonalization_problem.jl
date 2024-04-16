"""
    ExactDiagonalizationProblem(h::AbstractHamiltonian, [v0]; kwargs...)

Defines an exact diagonalization problem with an [`AbstractHamiltonian`](@ref) `h`.
Optionally, a starting vector of type [`AbstractDVec`](@ref), or a single address or a
collection of addresses can be passed as `v0`.

`ExactDiagonalizationProblem`s can be initialized with [`init`](@ref), and solved
with [`solve`](@ref).

# Keyword arguments
- `algorithm=LinearAlgebraEigen()`: The algorithm to use for solving the problem. The
    algorithm can also be specified as the second positional argument in the `init`
    function.
- Optional keyword arguments will be passed on to the `init` and `solve` functions.

# Algorithms
- [`LinearAlgebraEigen()`](@ref): An algorithm for solving the problem using the dense-matrix
    eigensolver from the `LinearAlgebra` standard library. Only suitable for small matrices.
- [`KrylovKitDirect()`](@ref): An algorithm for solving the problem without instantiating a
    sparse matrix. Requires `using KrylovKit`.
- [`KrylovKitMatrix()`](@ref): An algorithm for solving the problem after instantiating a
    sparse matrix. Requires `using KrylovKit`.
- [`ArpackEigs()`](@ref): An algorithm for solving the problem after instantiating a
    sparse matrix and using the Arpack Fortran library. Requires `using Arpack`.
- [`LOBPCG()`](@ref): An algorithm for solving the problem without instantiating a
    sparse matrix using the LOBPCG method. Requires `using IterativeSolvers`.

# Keyword arguments for `init` for matrix-based algorithms
- `sizelim`: The maximum size of the basis set representation. The default is `10^6` for
    sparse matrices and `10^4` for dense matrices.
- `cutoff`: A cutoff value for the basis set representation.
- `filter`: A filter function for the basis set representation.
- `nnzs = 0`: The number of non-zero elements in the basis set representation. Setting a
    non-zero value can speed up the computation.
- `col_hint = 0`: A hint for the number of columns in the basis set representation.
- `sort = false`: Whether to sort the basis set representation.

# Keyword arguments for `solve` for iterative algorithms
- `verbose = false`: Whether to print additional information.
- `abstol = nothing`: The absolute tolerance for the solver. If `nothing`, the solver
    chooses a default value.
- `reltol = nothing`: The relative tolerance for the solver (if applicable).
- `howmany = 1`: The minimum number of eigenvalues to compute.
- `which = :SR`: Whether to compute the largest or smallest eigenvalues.
- `maxiters = nothing`: The maximum number of iterations for the solver. If `nothing`, the
    solver chooses a default value.

# Solving an `ExactDiagonalizationProblem`
The `solve` function can be called directly on an `ExactDiagonalizationProblem` to solve it.
Alternatively, the `init` function can be used to initialize a solver, which can then be
solved with the `solve` function. The solve function returns a result type with the
eigenvalues, eigenvectors, and convergence information. While the type and the fields of the
result type depend on the algorithm it is guaranteed to have the fields `values`, `vectors`,
and `info`. Iterating the result type will yield the eigenvalues, eigenvectors, and
convergence information in that order.

# Examples
```jldoctest
julia> p = ExactDiagonalizationProblem(HubbardReal1D(BoseFS(1,1,1)))
ExactDiagonalizationProblem(
  HubbardReal1D(fs"|1 1 1⟩"; u=1.0, t=1.0),
  nothing;
  NamedTuple()...
)

julia> result = solve(p) # convert to dense matrix and solve with LinearAlgebra.eigen
MatrixEDEigenResult with 10 eigenvalue(s),
  values = [-5.09593, -1.51882, -1.51882, 1.55611, 1.6093, 1.6093, 4.0, 4.53982, 4.90952, 4.90952],
  and vectors of length 10.

julia> using KrylovKit # the next example requires julia v1.9 or later

julia> s = init(p; algorithm = KrylovKitDirect()) # solve without building a matrix
KrylovKitDirectEDSolver
 with algorithm KrylovKitDirect() for h = HubbardReal1D(fs"|1 1 1⟩"; u=1.0, t=1.0),
  v0 = 1-element PDVec: style = IsDeterministic{Float64}()
  fs"|1 1 1⟩" => 1.0,
  kwargs = NamedTuple()
)

julia> values, vectors = solve(s);

julia> result.values[1] ≈ values[1]
true
```
See also [`init`](@ref), [`solve`](@ref), [`KrylovKitDirect`](@ref),
[`KrylovKitMatrix`](@ref), [`ArpackEigs`](@ref), [`LinearAlgebraEigen`](@ref).
!!! note
    Using the `KrylovKitMatrix()` or `KrylovKitDirect()` algorithms requires the
    KrylovKit.jl package. The package can be loaded with `using KrylovKit`.
    Using the `ArpackEigs()` algorithm requires the Arpack.jl package. The package can be
    loaded with `using Arpack`.
    Using the `LOBPCG()` algorithm requires the IterativeSolvers.jl package. The package
    can be loaded with `using IterativeSolvers`.
    Algorithms with external packages require julia v1.9 or later.
"""
struct ExactDiagonalizationProblem{H<:AbstractHamiltonian, V, K<:NamedTuple}
    h::H
    v0::V
    kw_nt::K # NamedTuple

    function ExactDiagonalizationProblem(h::AbstractHamiltonian, v0=nothing; kwargs...)
        kw_nt = NamedTuple(kwargs)
        return new{typeof(h),typeof(v0),typeof(kw_nt)}(h, v0, kw_nt)
    end
end
function ExactDiagonalizationProblem(h::AbstractHamiltonian, v0::AbstractDVec; kwargs...)
    return ExactDiagonalizationProblem(h, freeze(v0); kwargs...)
end

function Base.show(io::IO, p::ExactDiagonalizationProblem)
    io = IOContext(io, :compact => true)
    print(io, "ExactDiagonalizationProblem(\n  ")
    show(io, p.h)
    print(io, ",\n  ")
    show(io, p.v0)
    print(io, ";\n  ")
    show(io, p.kw_nt)
    print(io, "...\n)")
end
function Base.:(==)(p1::ExactDiagonalizationProblem, p2::ExactDiagonalizationProblem)
    return p1.h == p2.h && p1.v0 == p2.v0 && p1.kw_nt == p2.kw_nt
end
