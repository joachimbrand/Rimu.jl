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
    eigensolver from the `LinearAlgebra` standard library.
- [`KrylovKitMatrix()`](@ref): An algorithm for solving the problem after instantiating a
    sparse matrix. Requires `using KrylovKit`.
- [`KrylovKitDirect()`](@ref): An algorithm for solving the problem without instantiating a
    sparse matrix. Requires `using KrylovKit`.
- [`ArpackEigs()`](@ref): An algorithm for solving the problem without instantiating a
    sparse matrix and using the Arpack Fortran library. Requires `using Arpack`.

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

julia> values, vectors = solve(p);

julia> round(values[1], digits=3) # ground state energy
-5.096

julia> using KrylovKit # the next example requires julia v1.9 or later

julia> s = init(p; algorithm = KrylovKitDirect())
KrylovKitDirectEDSolver
  for h = HubbardReal1D(fs"|1 1 1⟩"; u=1.0, t=1.0),
  v0 = 1-element PDVec: style = IsDeterministic{Float64}()
  fs"|1 1 1⟩" => 1.0,
  kwargs = NamedTuple()
)

julia> result = solve(s);

julia> result.values[1] ≈ values[1]
true
```
See also [`init`](@ref), [`solve`](@ref), [`LinearAlgebraEigen`](@ref),
[`KrylovKitMatrix`](@ref), [`KrylovKitDirect`](@ref).
!!! note
    Using the `KrylovKitMatrix()` or `KrylovKitDirect()` algorithms require julia v1.9 or
    later as well as the KrylovKit.jl package. The package can be loaded with
    `using KrylovKit`.
"""
struct ExactDiagonalizationProblem{H<:AbstractHamiltonian, V, K}
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

struct MatrixEDSolver{ALG, P, BSR<:BasisSetRep, V, K}
    algorithm::ALG
    problem::P
    basissetrep::BSR
    v0::V
    kw_nt::K # NamedTuple
end

function Base.show(io::IO, s::MatrixEDSolver)
    io = IOContext(io, :compact => true)
    b = s.basissetrep
    print(io, "MatrixEDSolver\n  with algorithm = ")
    show(io, s.algorithm)
    print(io, "\n for h  = ")
    show(io, s.problem.h)
    print(io, ":\n  ")
    show(io, MIME"text/plain"(), b.sm)
    print(io, ",\n  v0 = ")
    show(io, s.v0)
    print(io, ",\n  kwargs = ")
    show(io, NamedTuple(s.kw_nt))
    print(io, "\n)")
end

"""
    KrylovKitMatrix(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a
sparse matrix. This is faster than [`KrylovKitDirect()`](@ref), but it requires more memory
and will only be useful if the matrix fits into memory.

The `kwargs` are passed on to the function [`KrylovKit.eigsolve()`](
https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the KrylovKit.jl package to be loaded with `using KrylovKit`.
"""
struct KrylovKitMatrix{K}
    kw_nt::K # NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function KrylovKitMatrix(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitMatrix() requires that KrylovKit.jl is loaded, i.e. `using KrylovKit`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::KrylovKitMatrix)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "KrylovKitMatrix()")
    else
        print(io, "KrylovKitMatrix")
        show(io, s.kw_nt)
    end
end

"""
    ArpackEigs(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a
sparse matrix. This is faster than [`KrylovKitDirect()`](@ref), but it requires more memory
and will only be useful if the matrix fits into memory.

The `kwargs` are passed on to the function [`Arpack.eigs()`](
https://arpack.julialinearalgebra.org/stable/eigs/).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the Arpack.jl package to be loaded with `using Arpack`.
"""
struct ArpackEigs{K}
    kw_nt::K # NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function ArpackEigs(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :ArpackExt)
        if ext === nothing
            error("ArpackEigs() requires that Arpack.jl is loaded, i.e. `using Arpack`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::ArpackEigs)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "ArpackEigs()")
    else
        print(io, "ArpackEigs")
        show(io, s.kw_nt)
    end
end

"""
    LinearAlgebraEigen(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) using the dense-matrix
eigensolver from the `LinearAlgebra` standard library. This is only suitable for small
matrices.

The `kwargs` are passed on to function [`LinearAlgebra.eigen`](@ref
https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen).

# Keyword arguments
- `permute = true`: Whether to permute the matrix before diagonalization.
- `scale = true`: Whether to scale the matrix before diagonalization.
- `sortby`: The sorting order for the eigenvalues.

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
"""
struct LinearAlgebraEigen{K}
    kw_nt::K # NamedTuple
end
LinearAlgebraEigen(; kwargs...) = LinearAlgebraEigen(NamedTuple(kwargs))

function Base.show(io::IO, s::LinearAlgebraEigen)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "LinearAlgebraEigen()")
    else
        print(io, "LinearAlgebraEigen")
        show(io, s.kw_nt)
    end
end

"""
    init(p::ExactDiagonalizationProblem, [algorithm]; kwargs...)

Initialize a solver for an [`ExactDiagonalizationProblem`](@ref) `p` with an optional
`algorithm`. Returns a solver instance that can be solved with [`solve`](@ref).

For a description of the keyword arguments, see the documentation for
[`ExactDiagonalizationProblem`](@ref).
"""
function CommonSolve.init( # no algorithm specified as positional argument
    p::ExactDiagonalizationProblem;
    kwargs...
)
    kw_nt = (; p.kw_nt..., kwargs...) # remove duplicates
    if isdefined(kw_nt, :algorithm)
        algorithm = kw_nt.algorithm
        kw_nt = delete(kw_nt, (:algorithm,))
    else
        algorithm = LinearAlgebraEigen()
    end

    return init(p, algorithm; kw_nt...)
end

# init with matrix-based algorithms
function CommonSolve.init(
    p::ExactDiagonalizationProblem, algorithm::ALG;
    kwargs...
) where {ALG<:Union{KrylovKitMatrix,LinearAlgebraEigen,ArpackEigs}}
    # set keyword arguments for BasisSetRep
    kw = (; p.kw_nt..., algorithm.kw_nt..., kwargs...) # remove duplicates
    if isdefined(kw, :sizelim)
        sizelim = kw.sizelim
    elseif algorithm isa LinearAlgebraEigen
        sizelim = 10^4 # default for dense matrices
    else
        sizelim = 10^6 # default for sparse matrices
    end
    cutoff = get(kw, :cutoff, nothing)
    filter = if isdefined(kw, :filter)
        kw.filter
    elseif isnothing(cutoff)
        nothing
    else
        a -> diagonal_element(p.h, a) ≤ cutoff
    end
    nnzs = get(kw, :nnzs, 0)
    col_hint = get(kw, :col_hint, 0)
    sort = get(kw, :sort, false)

    # determine the starting address or vector
    addr_or_vec = if isnothing(p.v0)
            starting_address(p.h)
        elseif p.v0 isa Union{
            NTuple{<:Any, <:AbstractFockAddress},
            AbstractVector{<:AbstractFockAddress},
            AbstractFockAddress
        }
            p.v0
        elseif p.v0 isa DictVectors.FrozenDVec{<:AbstractFockAddress}
            keys(p.v0)
        else
            throw(ArgumentError("Invalid starting vector in `ExactDiagonalizationProblem`."))
    end

    # create the BasisSetRep
    bsr = BasisSetRep(p.h, addr_or_vec; sizelim, filter, nnzs, col_hint, sort)

    # prepare kwargs for the solver
    kw = (; kw..., sizelim, cutoff, filter, nnzs, col_hint, sort)
    kw_nt = delete(kw, (:sizelim, :cutoff, :filter, :nnzs, :col_hint, :sort))

    return MatrixEDSolver(algorithm, p, bsr, p.v0, kw_nt)
end

# solve directly on the ExactDiagonalizationProblem
"""
    solve(p::ExactDiagonalizationProblem, [algorithm]; kwargs...)

Solve an [`ExactDiagonalizationProblem`](@ref) `p` directly. Optionally specify an
`algorithm.` Returns a result type with the eigenvalues, eigenvectors, and convergence
information.

For a description of the keyword arguments, see the documentation for
[`ExactDiagonalizationProblem`](@ref).
"""
function CommonSolve.solve(p::ExactDiagonalizationProblem; kwargs...)
    s = init(p; kwargs...)
    return solve(s)
end
function CommonSolve.solve(p::ExactDiagonalizationProblem, algorithm; kwargs...)
    s = init(p, algorithm; kwargs...)
    return solve(s)
end

# The code for `CommonSolve.solve(::MatrixEDSolver; ...)` is part of the
# `KrylovKitExt.jl` extension.

struct KrylovKitDirectEDSolver{P,V<:PDVec,K}
    problem::P
    v0::V
    kw_nt::K
end
function Base.show(io::IO, s::KrylovKitDirectEDSolver)
    io = IOContext(io, :compact => true)
    print(io, "KrylovKitDirectEDSolver\n  for h = ")
    show(io, s.problem.h)
    print(io, ",\n  v0 = ")
    show(io, s.v0)
    print(io, ",\n  kwargs = ")
    show(io, s.kw_nt)
    print(io, "\n)")
end

"""
    KrylovKitDirect(; kwargs...)
Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) without instantiating a
sparse matrix. This is slower than [`KrylovKitMatrix()`](@ref), but it requires less memory
and thus can be useful for large matrices that would not fit into memory.
Will parallelise using threading and MPI if available.

The `kwargs` are passed on to the function [`KrylovKit.eigsolve()`](
https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the KrylovKit.jl package to be loaded with `using KrylovKit`.
"""
struct KrylovKitDirect{K}
    kw_nt::K # NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function KrylovKitDirect(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitDirect requires that KrylovKit is loaded, i.e. `using KrylovKit`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::KrylovKitDirect)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "KrylovKitDirect()")
    else
        print(io, "KrylovKitDirect")
        show(io, s.kw_nt)
    end
end

function CommonSolve.init(
    p::ExactDiagonalizationProblem, algorithm::KrylovKitDirect;
    kwargs...
)
    kw = (; p.kw_nt..., algorithm.kw_nt..., kwargs...) # remove duplicates
    # set up the starting vector
    vec = if isnothing(p.v0)
        FrozenDVec([starting_address(p.h) => 1.0])
    elseif p.v0 isa AbstractFockAddress
        FrozenDVec([p.v0 => 1.0])
    elseif p.v0 isa Union{
        NTuple{<:Any,<:AbstractFockAddress},
        AbstractVector{<:AbstractFockAddress},
    }
        FrozenDVec([addr => 1.0 for addr in p.v0])
    elseif p.v0 isa FrozenDVec{<:AbstractFockAddress}
        p.v0
    else
        throw(ArgumentError("Invalid starting vector in `ExactDiagonalizationProblem`."))
    end
    svec = PDVec(vec)

    return KrylovKitDirectEDSolver(p, svec, kw)
end
# The code for `CommonSolve.solve(::KrylovKitDirectEDSolver; ...)` is part of the
# `KrylovKitExt.jl` extension.

# result types for `solve(::ExactDiagonalizationProblem; ...)`
abstract type AbstractEDResult end
# iteration for destructuring into components
Base.iterate(S::AbstractEDResult) = (S.values, Val(:vectors))
Base.iterate(S::AbstractEDResult, ::Val{:vectors}) = (S.vectors, Val(:info))
Base.iterate(S::AbstractEDResult, ::Val{:info}) = (S.info, Val(:done))
Base.iterate(::AbstractEDResult, ::Val{:done}) = nothing

struct MatrixEDEigenResult{P,B,F} <: AbstractEDResult
    problem::P
    basis::B
    eigen_factorization::F
end
function Base.show(io::IO, r::MatrixEDEigenResult)
    n = length(r.values)
    print(io, "Rimu.MatrixEDEigenResult with $n eigenvalue(s),\n  `values` = ")
    show(io, r.values)
    print(io, ",\n  and `vectors` of length $n.")
end
function Base.getproperty(r::MatrixEDEigenResult, key::Symbol)
    vs = getfield(r, :eigen_factorization).vectors
    n = size(vs, 2)
    if key === :values
        return getfield(r, :eigen_factorization).values
    elseif key === :vectors
        return [DVec(zip(getfield(r, :basis), view(vs, :, i))) for i in 1:n]
    elseif key === :info
        return "Dense matrix eigensolver solution from `LinearAlgebra.eigen`."
    else
        return getfield(r, key)
    end
end


function CommonSolve.solve(s::Rimu.MatrixEDSolver{<:LinearAlgebraEigen};
    kwargs...
)
    # combine keyword arguments
    kw_nt = (; s.kw_nt..., kwargs...)

    # extract relevant keyword arguments
    permute = get(kw_nt, :permute, true)
    scale = get(kw_nt, :scale, true)

    eigen_factorization = if isdefined(kw_nt, :sortby)
        sortby = kw_nt.sortby
        eigen(Matrix(s.basissetrep.sm); permute, scale, sortby)
    else
        eigen(Matrix(s.basissetrep.sm); permute, scale)
    end
    nt = delete(kw_nt, (:permute, :scale, :sortby))
    !isempty(nt) && @warn "Unused keyword arguments in `solve`: $nt"
    # eigen_factorization = eigen(Matrix(s.basissetrep.sm); kw_nt...)

    return MatrixEDEigenResult(s.problem, s.basissetrep.basis, eigen_factorization)
end
