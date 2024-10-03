"""
    OccupationNumberFS{M,T} <: SingleComponentFockAddress
Address type that stores the occupation numbers of a single component bosonic Fock state
with `M` modes. The occupation numbers must fit into the type `T <: Unsigned`. The number of
particles is runtime data, and can be retrieved with `num_particles(address)`.

# Constructors
- `OccupationNumberFS(val::Integer...)`: Construct from occupation numbers. Must be
  < 256 to fit into `UInt8`.
- `OccupationNumberFS{[M,T]}(onr)`: Construct from collection `onr` with `M` occupation
  numbers with type `T`. If unspecified, the type `T` of the occupation numbers is inferred
  from the type of the arguments.
- `OccupationNumberFS(fs::BoseFS)`: Construct from [`BoseFS`](@ref).
- With shortform macro [`@fs_str`](@ref). Specify the number of
  significant bits in braces. See example below.

# Examples
```jl_doctest
julia> ofs = OccupationNumberFS(1,2,3)
OccupationNumberFS{3, UInt8}(1, 2, 3)

julia> ofs == fs"|1 2 3⟩{8}"
true

julia> num_particles(ofs)
6
```
"""
struct OccupationNumberFS{M,T<:Unsigned} <: SingleComponentFockAddress{missing,M}
    onr::SVector{M,T}
end

function OccupationNumberFS{M,T}(args...) where {M,T}
    return OccupationNumberFS(SVector{M,T}(args...))
end

function OccupationNumberFS(args...)
    sv = SVector(args...)
    all(isinteger, sv) || throw(ArgumentError("all arguments must be integers"))
    all(x -> x ≥ 0, sv) || throw(ArgumentError("all arguments must be non-negative"))
    all(x -> x < 256, sv) || throw(ArgumentError("arguments don't fit in a byte, specify type"))
    return OccupationNumberFS(SVector{length(sv),UInt8}(args...))
end

function OccupationNumberFS{M}(args...) where M
    sv = SVector{M}(args...)
    all(isinteger, sv) || throw(ArgumentError("all arguments must be integers"))
    all(x -> x ≥ 0, sv) || throw(ArgumentError("all arguments must be non-negative"))
    all(x -> x < 256, sv) || throw(ArgumentError("arguments don't fit in a byte, specify type"))
    return OccupationNumberFS(SVector{M,UInt8}(args...))
end

function OccupationNumberFS(fs::BoseFS{N,M}) where {N,M}
    return OccupationNumberFS{M,select_int_type(N)}(onr(fs))
end

function print_address(io::IO, ofs::OccupationNumberFS{M,T}; compact=false) where {M,T}
    if compact
        BITS = sizeof(T) * 8
        print(io, "|", join(ofs.onr, ' '), "⟩{", BITS, "}")
    else
        print(io, "OccupationNumberFS{", M, ", ", T, "}", Int.(tuple(ofs.onr...)))
    end
end

Base.reverse(ofs::OccupationNumberFS) = OccupationNumberFS(reverse(ofs.onr))

onr(ofs::OccupationNumberFS) = ofs.onr
function Base.isless(a::OccupationNumberFS{M}, b::OccupationNumberFS{M}) where {M}
    # equivalent to `isless(reverse(a.onr), reverse(b.onr))`
    i = length(a.onr)
    while i > 1 && a.onr[i] == b.onr[i]
        i -= 1
    end
    return isless(a.onr[i], b.onr[i])
end
# reversing the order here to make it consistent with BoseFS
Base.:(==)(a::OccupationNumberFS, b::OccupationNumberFS) = a.onr == b.onr
Base.hash(ofs::OccupationNumberFS, h::UInt) = hash(ofs.onr, h)

num_particles(ofs::OccupationNumberFS) = sum(Int, onr(ofs))
# `num_modes` does not have to be defined here, because it is defined for the abstract type

"""
    destroy(ofs::OccupationNumberFS{M,T}, i::Integer) where {M,T}

Destroy one particle at the `i`-th mode of the occupation number Fock state `ofs`.

# Arguments
- `ofs::OccupationNumberFS{M,T}`: The occupation number Fock state, where `M` is the number
  of modes and `T` is the type of the occupation numbers.
- `i::Integer`: The index of the mode where the particle is to be destroyed.

# Returns
- A tuple containing the updated Fock state and the occupation number at the `i`-th site
  before the destruction.

See also: [`create`](@ref), [`excitation`](@ref).
"""
@inline function destroy(ofs::OccupationNumberFS{M,T}, i::Integer) where {M,T}
    val = ofs.onr[i]
    @set! ofs.onr[i] = val - one(T)
    return (ofs, val)
end

"""
    create(ofs::OccupationNumberFS{M,T}, i::Integer) where {M,T}

Create one particle at the `i`-th mode of the occupation number Fock state `ofs`.

# Arguments
- `ofs::OccupationNumberFS{M,T}`: The occupation number Fock state, where `M` is the number
  of modes and `T` is the type of the occupation numbers.
- `i::Integer`: The index of the mode where the particle is to be created.

# Returns
- A tuple containing the updated Fock state and the occupation number at the `i`-th mode
  after the creation.

See also: [`destroy`](@ref), [`excitation`](@ref).
"""
@inline function create(ofs::OccupationNumberFS{M,T}, i::Integer) where {M,T}
    val = ofs.onr[i] + one(T)
    @set! ofs.onr[i] = val
    return (ofs, val)
end

"""
    excitation(addr::OccupationNumberFS, c::NTuple, d::NTuple)
    → (nadd, α)
Generate an excitation on an [`OccupationNumberFS`](@ref) by applying the creation and
destruction operators specified by the tuples of mode numbers `c` and `d` to the Fock state
`addr`. The modes are indexed by integers (starting at 1), or by indices of type
`BoseFSIndex`. The value of `α` is given by the square root of the product of mode
occupations before destruction and after creation.

The number of particles may change by this type of excitation.

# Example
```jl_doctest
julia> s = fs"|1 2 3⟩{8}"
OccupationNumberFS{3, UInt8}(1, 2, 3)

julia> num_particles(s)
6

julia> es, α = excitation(s, (1,1), (3,))
(OccupationNumberFS{3, UInt8}(3, 2, 2), 4.242640687119285)

julia> num_particles(es)
7
```
"""
function excitation(
    fs::OccupationNumberFS{<:Any,T},
    c::NTuple{<:Any,Int},
    d::NTuple{<:Any,Int}
) where {T}
    accu = one(T)
    for i in d
        fs, val = destroy(fs, i)
        accu *= val
    end
    for i in c
        fs, val = create(fs, i)
        accu *= val
    end
    return fs, √accu
end
function excitation(
    fs::OccupationNumberFS,
    c::NTuple{N1,BoseFSIndex},
    d::NTuple{N2,BoseFSIndex}
) where {N1, N2}
    creations = ntuple(i -> c[i].mode, Val(N1)) # convert BoseFSIndex to mode number
    destructions = ntuple(i -> d[i].mode, Val(N2))
    return excitation(fs, creations, destructions)
end


# `SingleComponentFockAddress` interface

find_mode(ofs::OccupationNumberFS, n::Integer) = BoseFSIndex(ofs.onr[n], n, n)
function find_mode(ofs::OccupationNumberFS, ns::NTuple{N,Integer}) where N
    return ntuple(i -> find_mode(ofs, ns[i]), Val(N))
end

num_occupied_modes(ofs::OccupationNumberFS) = count(!iszero, onr(ofs))

# for the lazy iterator `occupied_modes` we adapt the `BoseOccupiedModes` type
function occupied_modes(ofs::OccupationNumberFS{M}) where {M}
    return BoseOccupiedModes{missing, M, typeof(ofs)}(ofs)
end

function Base.length(bom::BoseOccupiedModes{<:Any,<:Any,<:OccupationNumberFS})
    return num_occupied_modes(bom.storage)
end

function Base.iterate(bom::BoseOccupiedModes{<:Any,<:Any,<:OccupationNumberFS}, i=1)
    s = onr(bom.storage) # is an SVector with the onr
    while true
        i > length(s) && return nothing
        iszero(s[i]) || return BoseFSIndex(s[i], i, i), i + 1
        i += 1
    end
end

# find_occupied_modes provided by generic implementation
# OccupiedModeMap provided by generic implementation
