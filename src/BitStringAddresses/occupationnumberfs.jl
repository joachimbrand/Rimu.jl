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

onr(ofs::OccupationNumberFS) = ofs.onr
num_occupied_modes(ofs::OccupationNumberFS) = count(!iszero, onr(ofs))
num_particles(ofs::OccupationNumberFS) = Int(sum(onr(ofs)))
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
    excitation(addr::OccupationNumberFS, c::NTuple{<:Any,Int}, d::NTuple{<:Any,Int})
    → (nadd, α)
Generate an excitation on an [`OccupationNumberFS`](@ref) by applying the creation and
destruction operators specified by the tuples of mode numbers `c` and `d` to the Fock state
`addr`. The modes are simply indexed by integers (starting at 1). The value of `α` is given
by the square root of the product of mode occupations before destruction and after creation.

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
function excitation(fs::OccupationNumberFS{<:Any,T}, c::NTuple{<:Any,Int}, d::NTuple{<:Any,Int}) where {T}
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

# Do we need more methods for building Hamiltonians? (`find_occupied_mode`,
# `OccupiedModMap`, `occupied_modes`?)
