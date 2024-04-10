"""
    ExactDiagonalizationProblem(h::AbstractHamiltonian, [v0]; kwargs...)

Defines an exact diagonalization problem in quantum physics by an
[`AbstractHamiltonian`](@ref) `h`, and optionally a starting vector or starting address
`v0`. `ExactDiagonalizationProblem`s can be initialized with [`init`](@ref), and solved
with [`solve`](@ref).

# Arguments
The constructor for `ExactDiagonalizationProblem` takes the following arguments:
- `h::AbstractHamiltonian`: The Hamiltonian of the system.
- `v0`: Optional. The initial state vector as a [`AbstractDVec`](@ref). Alternatively, a
    single address, a collection of addresses, or a single starting vector can be passed.
- Optional keyword arguments will be passed on to the `init` and `solve` functions.

# Keyword arguments for `init`
- `sizelim = 10^6`: The maximum size of the basis set representation.
- `cutoff`: A cutoff value for the basis set representation.
- `filter`: A filter function for the basis set representation.
- `nnzs = 0`: The number of non-zero elements in the basis set representation. Setting a
    non-zero value can speed up the computation.
- `col_hint = 0`: A hint for the number of columns in the basis set representation.
- `sort = false`: Whether to sort the basis set representation.
- `algorithm = KrylovKitMatrix()`: The algorithm to use for the solver.

# Keyword arguments for `solve`
- `verbose = false`: Whether to print additional information.
- `abstol = nothing`: The absolute tolerance for the solver. If `nothing`, the solver
    chooses a default value.
- `reltol = nothing`: The relative tolerance for the solver (if applicable).
- `howmany = 1`: The minimum number of eigenvalues to compute.
- `which = :SR`: Whether to compute the largest or smallest eigenvalues.
- `maxiters = nothing`: The maximum number of iterations for the solver. If `nothing`, the
    solver chooses a default value.

# Examples
```julia
julia> using Rimu, KrylovKit

julia> p = ExactDiagonalizationProblem(HubbardReal1D(BoseFS(1,1,1,1,1)), DVec(BoseFS(1,1,1,1,1)=>2.3); which=:LM, howmany=1)
ExactDiagonalizationProblem(
  HubbardReal1D(fs"|1 1 1 1 1⟩"; u=1.0, t=1.0),
  Rimu.FrozenDVec([fs"|1 1 1 1 1⟩"=>2.3]);
  (which = :LM, howmany = 1)...
)

julia> s = init(p, KrylovKitMatrix());

julia> res = solve(s)
Rimu.KrylovKitResult with 1 eigenvalue(s),
  vals = [-8.28099174658269],
  and vecs of length 126.
  Convergence info: one converged value after 1 iteration(s) and 30 applications of the linear map.
  The norms of the residuals are ≤ 7.128792403491991e-26.
```
!!! note
    Using the `KrylovKitMatrix()` algorithm requires the KrylovKit.jl package.
    It can be loaded with `using KrylovKit`.
"""
struct ExactDiagonalizationProblem{H<:AbstractHamiltonian, V, K}
    h::H
    v0::V
    kwargs::K

    function ExactDiagonalizationProblem(h::AbstractHamiltonian, v0=nothing; kwargs...)
        return new{typeof(h),typeof(v0),typeof(kwargs)}(h, v0, kwargs)
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
    show(io, NamedTuple(p.kwargs))
    print(io, "...\n)")
end
function Base.:(==)(p1::ExactDiagonalizationProblem, p2::ExactDiagonalizationProblem)
    return p1.h == p2.h && p1.v0 == p2.v0 && p1.kwargs == p2.kwargs
end

struct KrylovKitMatrixEDSolver{P, BSR<:BasisSetRep, V, K}
    problem::P
    basissetrep::BSR
    v0::V
    kwargs::K
end

function Base.show(io::IO, s::KrylovKitMatrixEDSolver)
    io = IOContext(io, :compact => true)
    b = s.basissetrep
    print(io, "KrylovKitMatrixEDSolver\n  for h = ")
    show(io, s.problem.h)
    print(io, ":\n  ")
    show(io, MIME"text/plain"(), b.sm)
    print(io, ",\n  v0 = ")
    show(io, s.v0)
    print(io, ",\n  kwargs = ")
    show(io, NamedTuple(s.kwargs))
    print(io, "\n)")
end

struct KrylovKitMatrix
    # empty struct; the inner constructor checks if KrylovKit is loaded
    function KrylovKitMatrix()
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitMatrix requires that KrylovKit is loaded, i.e. `using KrylovKit`")
        else
            return new()
        end
    end
end

"""
    init(p::ExactDiagonalizationProblem, [algorithm]; kwargs...)

Initialize a solver for an [`ExactDiagonalizationProblem`](@ref) `p` with an optional
`algorithm`. Returns a solver instance that can be solved with [`solve`](@ref).

For a description of the keyword arguments, see the documentation for
[`ExactDiagonalizationProblem`](@ref).
"""
function CommonSolve.init(
    p::ExactDiagonalizationProblem, ::KrylovKitMatrix;
    sizelim = 10^6,
    cutoff=nothing,
    filter=isnothing(cutoff) ? nothing : (a -> diagonal_element(ham, a) ≤ cutoff),
    nnzs=0, col_hint=0, # sizehints are opt-in
    sort=false,
    kwargs...
)
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
    bsr = BasisSetRep(p.h, addr_or_vec; sizelim, filter, nnzs, col_hint, sort)

    return KrylovKitMatrixEDSolver(p, bsr, p.v0, (p.kwargs..., kwargs...))
end

# no algorithm specified as positional argument
function CommonSolve.init(
    p::ExactDiagonalizationProblem;
    kwargs...
)
    # set default algorithm unless specified
    kwargs = (:algorithm => KrylovKitMatrix(), kwargs...)
    nt = NamedTuple(kwargs) # remove duplicates, only the last value is kept
    algorithm = nt.algorithm
    kwargs = delete(nt, (:algorithm,))

    return init(p, algorithm; kwargs...)
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

# The code for `CommonSolve.solve(::KrylovKitMatrixEDSolver; ...)` is part of the
# `KrylovKitExt.jl` extension.
