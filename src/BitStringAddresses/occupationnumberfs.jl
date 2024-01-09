"""
    select_uint_type(::Val{N}) where {N}

Select a suitable unsigned integer type for storing `N` particles in a single mode.
"""
function select_uint_type(::Val{N}) where {N}
    if 0 < N < 2^8
        return UInt8
    elseif 2^8 ≤ N < 2^16
        return UInt16
    elseif 2^16 ≤ N < 2^32
        return UInt32
    else
        throw(ArgumentError("`N` must be ≤ 2^32 and > 0, got $N"))
    end
end

"""
    OccupationNumberFS{M,T} <: SingleComponentFockAddress
Address type that stores the occupation numbers of a single component bosonic Fock state
with `M` modes. The occupation numbers must fit into the type `T <: Unsigned`. The number of
particles is runtime data, and can be retrieved with `num_particles(address)`.

# Constructors
- `OccupationNumberFS(number1, ...)`: Construct from occupation numbers. Must be `Integer`
  and < 256 to fit into `UInt8`.
- `OccupationNumberFS{[M,T]}(onr): Construct from collection `onr` with `M` occupation
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
struct OccupationNumberFS{M,T} <: SingleComponentFockAddress{missing,M}
    onr::SVector{M,T}

    function OccupationNumberFS(onr::SVector{M,T}) where {M,T<:Unsigned}
        new{M,T}(onr)
    end
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
    return OccupationNumberFS{M,select_uint_type(Val(N))}(onr(fs))
end

function print_address(io::IO, ofs::OccupationNumberFS{M,T}; compact=false) where {M,T}
    if compact
        BITS = sizeof(T) * 8
        print(io, "|", join(ofs.onr, ' '), "⟩{", BITS, "}")
    else
        print(io, "OccupationNumberFS{", M, ", ", T, "}", Int.(tuple(ofs.onr...)))
    end
end

onr(ofs::OccupationNumberFS) = ofs.onr
num_occupied_modes(ofs::OccupationNumberFS) = mapreduce(!iszero, +, onr(ofs))
num_particles(ofs::OccupationNumberFS) = sum(onr(ofs))|>Int

# TODO: methods for building Hamiltonians
# - `excitation`
# - `occupied_modes`
# - `OccupiedModMap`

# Which methods do we need? (`find_occupied_mode`?)
