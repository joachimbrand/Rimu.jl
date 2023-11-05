"""
    FermiFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` fermions of the same spin in `M` modes by
wrapping a [`BitString`](@ref), or a [`SortedParticleList`](@ref). Which is wrapped is
chosen automatically based on the properties of the address.

# Constructors

* `FermiFS{[N,M]}(onr)`: Create `FermiFS{N,M}` from [`onr`](@ref) representation. This is
  efficient if `N` and `M` are provided, and `onr` is a statically-sized collection, such as
  a `Tuple{M}` or `SVector{M}`.

* `FermiFS{[N,M]}([M, ]pairs...)`: Provide the number of modes `M` and pairs of the form
  `mode => 1`. If `M` is provided as a type parameter, it should not be provided as the
  first argument.  Useful for creating sparse addresses. `pairs` can be multiple arguments
  or an iterator of pairs.

* `FermiFS{N,M,S}(bs::S)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`, or whether each mode only contains one particle.

* [`@fs_str`](@ref): addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the last example below.

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`BitString`](@ref).

# Examples

```jldoctest
julia> FermiFS{3,5}((0, 1, 1, 1, 0))
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> FermiFS([abs(i - 3) ≤ 1 for i in 1:5])
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> FermiFS(5, 2 => 1, 3 => 1, 4 => 1)
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> FermiFS{3,5}(i => 1 for i in 2:4)
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> fs"|⋅↑↑↑⋅⟩"
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> fs"|f 5: 2 3 4⟩"
FermiFS{3,5}((0, 1, 1, 1, 0))
```

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`CompositeFS`](@ref),
[`FermiFS2C`](@ref).
"""
struct FermiFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

function check_fermi_onr(onr, N, M)
    sum(onr) == N ||
        throw(ArgumentError("Invalid ONR: $N particles expected, $(sum(onr)) given."))
    length(onr) == M ||
        throw(ArgumentError("Invalid ONR: $M modes expected, $(length(onr)) given."))
    all(in((0, 1)), onr) ||
        throw(ArgumentError("Invalid ONR: may only contain 0s and 1s."))
end

function FermiFS{N,M,S}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {N,M,S}
    @boundscheck begin
        check_fermi_onr(onr, N, M)
        if S <: BitString
            M == num_bits(S) || throw(ArgumentError(
                "invalid ONR: $B-bit BitString does not fit $M modes"
            ))
        elseif S <: SortedParticleList
            N == num_particles(S) && M == num_modes(S) || throw(ArgumentError(
                "invalid ONR: $S does not fit $N particles in $M modes"
            ))
        end
    end
    return FermiFS{N,M,S}(from_fermi_onr(S, onr))
end
function FermiFS{N,M}(onr::Union{AbstractArray{<:Integer},NTuple{M,<:Integer}}) where {N,M}
    @boundscheck check_fermi_onr(onr, N, M)
    spl_type = select_int_type(M)
    # Pick smaller address type, but prefer dense.
    # Alway pick dense if it fits into one chunk.

    # Compute the size of container in words
    sparse_sizeof = ceil(Int, N * sizeof(spl_type) / 8)
    dense_sizeof = ceil(Int, M / 64)
    if dense_sizeof == 1 || dense_sizeof ≤ sparse_sizeof
        S = typeof(BitString{M}(0))
    else
        S = SortedParticleList{N,M,spl_type}
    end
    return FermiFS{N,M,S}(from_fermi_onr(S, onr))
end
function FermiFS(onr::Union{AbstractArray,Tuple})
    M = length(onr)
    N = sum(onr)
    return FermiFS{N,M}(onr)
end

# Sparse constructors
FermiFS(M::Integer, pairs::Pair...) = FermiFS(M, pairs)
FermiFS(M::Integer, pairs) = FermiFS(sparse_to_onr(M, pairs))
FermiFS{N,M}(pairs::Vararg{Pair,N}) where {N,M} = FermiFS{N,M}(pairs)
FermiFS{N,M}(pairs) where {N,M} = FermiFS{N,M}(sparse_to_onr(M, pairs))

function print_address(io::IO, f::FermiFS{N,M}; compact=false) where {N,M}
    if compact && f.bs isa SortedParticleList
        print(io, "|f ", M, ": ", join(Int.(f.bs.storage), ' '), "⟩")
    elseif compact
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩")
    elseif f.bs isa SortedParticleList
        print(io, "FermiFS{$N,$M}(", onr_sparse_string(onr(f)), ")")
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

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{FermiFS{N,M}}) where {N,M}
    onr = zeros(MVector{M, Int})
    left = N
    @inbounds while left > 0
        i = rand(rng, 1:M)
        left -= onr[i] == 0
        onr[i] = 1
    end
    return FermiFS{N,M}(onr)
end
