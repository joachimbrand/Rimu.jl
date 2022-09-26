"""
    FermiFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` fermions of the same spin in `M` modes
by wrapping a bitstring of type `S <: BitString`.

# Constructors

* `FermiFS{N,M}(bs::BitString)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* `FermiFS(::BitString)`: Automatically determine `N` and `M`. This constructor is not type
  stable!

* `FermiFS{[N,M,S]}(onr)`: Create `FermiFS{N,M}` from [`onr`](@ref) representation. This is
  efficient as long as at least `N` is provided.

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`BitString`](@ref).
"""
struct FermiFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

function check_fermi_onr(onr, N)
    sum(onr) == N ||
        throw(ArgumentError("Invalid ONR: $N particles expected, $(sum(onr)) given."))
    all(in((0, 1)), onr) ||
        throw(ArgumentError("Invalid ONR: may only contain 0s and 1s."))
end

function FermiFS{N,M,S}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {N,M,S}
    @boundscheck check_fermi_onr(onr, N)
    return FermiFS{N,M,S}(from_fermi_onr(S, onr))
end
function FermiFS{N,M}(onr::Union{AbstractArray,Tuple}) where {N,M}
    @boundscheck check_fermi_onr(onr, N)
    spl_type = select_int_type(M)
    S_sparse = SortedParticleList{N,M,spl_type}
    S_dense = typeof(BitString{M}(0))
    # Pick smaller address type, but prefer dense.
    # Alway pick dense if it fits into one chunk.
    sparse_size_64 = ceil(Int, sizeof(S_dense) / 8)
    dense_size_64 = ceil(Int, sizeof(S_dense) / 8)
    if num_chunks(S_dense) == 1 || dense_size_64 ≤ sparse_size_64
        S = S_dense
    else
        S = S_sparse
    end
    return FermiFS{N,M,S}(from_fermi_onr(S, SVector{M,Int}(onr)))
end
function FermiFS{N}(onr::Union{SVector{M},Tuple{M}}) where {N,M}
    return FermiFS{N,M}(onr)
end
function FermiFS(onr::Union{AbstractArray,Tuple})
    M = length(onr)
    N = sum(onr)
    return FermiFS{N,M}(onr)
end

function print_address(io::IO, f::FermiFS{N,M}; compact=false) where {N,M}
    if compact
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩")
    else
        print(io, "FermiFS{$N,$M}(", tuple(onr(f)...), ")")
    end
end

Base.bitstring(a::FermiFS) = bitstring(a.bs)
Base.isless(a::FermiFS, b::FermiFS) = isless(a.bs, b.bs)
Base.hash(a::FermiFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiFS, b::FermiFS) = a.bs == b.bs
num_occupied_modes(::FermiFS{N}) where {N} = N
occupied_modes(a::FermiFS{N,<:Any,S}) where {N,S} = FermiOccupiedModes{N,S}(a.bs)

function near_uniform(::Type{FermiFS{N,M}}) where {N,M}
    return FermiFS([fill(1, N); fill(0, M - N)])
end

@inline function onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    @inbounds for (_, mode) in occupied_modes(a)
        result[mode] = 1
    end
    return SVector(result)
end

find_mode(a::FermiFS, i) = fermi_find_mode(a.bs, i)

function find_occupied_mode(a::FermiFS, i::Integer)
    for k in occupied_modes(a)
        i -= 1
        i == 0 && return k
    end
    return FermiFSIndex(0, 0, 0)
end

function Base.reverse(f::FermiFS)
    return typeof(f)(reverse(f.bs))
end

function excitation(a::FermiFS{N,M,S}, creations, destructions) where {N,M,S}
    new_bs, value = fermi_excitation(a.bs, creations, destructions)
    return FermiFS{N,M,S}(new_bs), value
end
