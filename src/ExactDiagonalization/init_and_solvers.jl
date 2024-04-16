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


struct KrylovKitDirectEDSolver{A<:KrylovKitDirect,P,V<:PDVec,K<:NamedTuple}
    algorithm::A
    problem::P
    v0::V
    kw_nt::K
end
function Base.show(io::IO, s::KrylovKitDirectEDSolver)
    io = IOContext(io, :compact => true)
    print(io, "KrylovKitDirectEDSolver\n with algorithm $(s.algorithm) for h = ")
    show(io, s.problem.h)
    print(io, ",\n  v0 = ")
    show(io, s.v0)
    print(io, ",\n  kwargs = ")
    show(io, s.kw_nt)
    print(io, "\n)")
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

    return KrylovKitDirectEDSolver(algorithm, p, svec, kw)
end

struct MatrixEDSolver{A,P,BSR<:BasisSetRep,V<:Union{Nothing,FrozenDVec},K<:NamedTuple}
    algorithm::A
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

# init with matrix-based algorithms
function CommonSolve.init(
    p::ExactDiagonalizationProblem, algorithm::ALG;
    kwargs...
) where {ALG<:Union{KrylovKitMatrix,LinearAlgebraEigen,ArpackEigs,LOBPCG}}
    !ishermitian(p.h) && algorithm isa LOBPCG &&
        @warn "LOBPCG() is not suitable for non-hermitian matrices."

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
    v0 = p.v0
    if isnothing(p.v0)
        addr_or_vec = starting_address(p.h)
    elseif p.v0 isa Union{
        NTuple{<:Any,<:AbstractFockAddress},
        AbstractVector{<:AbstractFockAddress}
    }
        addr_or_vec = p.v0
        v0 = FrozenDVec([addr => 1.0 for addr in p.v0])
    elseif p.v0 isa AbstractFockAddress
        addr_or_vec = p.v0
        v0 = FrozenDVec([p.v0 => 1.0])
    elseif p.v0 isa DictVectors.FrozenDVec{<:AbstractFockAddress}
        addr_or_vec = keys(p.v0)
    else
        throw(ArgumentError("Invalid starting vector in `ExactDiagonalizationProblem`."))
    end
    @assert v0 isa Union{FrozenDVec{<:AbstractFockAddress},Nothing}

    # create the BasisSetRep
    bsr = BasisSetRep(p.h, addr_or_vec; sizelim, filter, nnzs, col_hint, sort)

    # prepare kwargs for the solver
    kw = (; kw..., sizelim, cutoff, filter, nnzs, col_hint, sort)
    kw_nt = delete(kw, (:sizelim, :cutoff, :filter, :nnzs, :col_hint, :sort))

    return MatrixEDSolver(algorithm, p, bsr, v0, kw_nt)
end
