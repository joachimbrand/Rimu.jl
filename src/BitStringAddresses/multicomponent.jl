"""
    BoseFS2C{NA,NB,M,AA,AB} <: AbstractFockAddress
    BoseFS2C(onr_a, onr_b)

Address type that constructed with two [`BoseFS{N,M,S}`](@ref). It represents a
Fock state with two components, e.g. two different species of bosons with particle
number `NA` from species S and particle number `NB` from species B. The number of
modes `M` is expected to be the same for both components.
"""
struct BoseFS2C{NA,NB,M,SA,SB,N} <: AbstractFockAddress{N,M}
    bsa::BoseFS{NA,M,SA}
    bsb::BoseFS{NB,M,SB}

end

function BoseFS2C(bsa::BoseFS{NA,M,SA}, bsb::BoseFS{NB,M,SB}) where {NA,NB,M,SA,SB}
    N = NA + NB
    return BoseFS2C{NA,NB,M,SA,SB,N}(bsa, bsb)
end
BoseFS2C(onr_a::Tuple, onr_b::Tuple) = BoseFS2C(BoseFS(onr_a),BoseFS(onr_b))

function print_address(io::IO, b::BoseFS2C; compact)
    if compact
        print_address(io, b.bsa; compact)
        print(io, " ⊗ ")
        print_address(io, b.bsb; compact)
    else
        print(io, "BoseFS2C(", b.bsa, ", ", b.bsb, ")")
    end
end

num_components(::Type{<:BoseFS2C}) = 2
Base.isless(a::T, b::T) where {T<:BoseFS2C} = isless((a.bsa, a.bsb), (b.bsa, b.bsb))
onr(b2::BoseFS2C) = (onr(b2.bsa), onr(b2.bsb))


function near_uniform(::Type{<:BoseFS2C{NA,NB,M}}) where {NA,NB,M}
    return BoseFS2C(near_uniform(BoseFS{NA,M}), near_uniform(BoseFS{NB,M}))
end

function time_reverse(c::BoseFS2C{NA,NA,M,S,S,N}) where {NA,M,S,N}
    return  BoseFS2C{NA,NA,M,S,S,N}(c.bsb, c.bsa)
end

"""
    CompositeFS(addresses::SingleComponentFockAddress...) <: AbstractFockAddress

Used to encode addresses for multi-component models. All component addresses
are expected have the same number of modes.

See also: [`BoseFS`](@ref), [`FermiFS`](@ref), [`SingleComponentFockAddress`](@ref),
[`num_modes`](@ref), [`FermiFS2C`](@ref), [`AbstractFockAddress`](@ref).
"""
struct CompositeFS{C,N,M,T} <: AbstractFockAddress{N,M}
    components::T
    # C: components, N: total particles, M: modes in each component, T: tuple type with constituent address types
    function CompositeFS{C,N,M,T}(adds::T) where {C,N,M,T}
        return new{C,N,M,T}(adds)
    end
    function CompositeFS{C,N,M,T}(adds...) where {C,N,M,T}
        return new{C,N,M,T}(adds)
    end
end

# Slow constructor - not to be used internallly
function CompositeFS(adds::Vararg{SingleComponentFockAddress})
    N = sum(num_particles, adds)
    M1, M2 = extrema(num_modes, adds)
    if M1 ≠ M2
        throw(ArgumentError("all addresses must have the same number of modes"))
    end
    return CompositeFS{length(adds),N,M1,typeof(adds)}(adds)
end

num_components(::Type{<:CompositeFS{C}}) where {C} = C
Base.hash(c::CompositeFS, u::UInt) = hash(c.components, u)

function print_address(io::IO, c::CompositeFS{C}; compact=false) where {C}
    if compact
        for add in c.components[1:end-1]
            print_address(io, add; compact)
            print(io, " ⊗ ")
        end
        print_address(io, c.components[end]; compact)
    else
        println(io, "CompositeFS(")
        for add in c.components
            println(io, "  ", add, ",")
        end
        print(io, ")")
    end
end

function Base.reverse(c::CompositeFS)
    typeof(c)(map(reverse, c.components))
end

"""
    time_reverse(addr)
Apply the time-reversal operation on a two-component Fock address that flips all the spins.

Requires each component address to have the same type.
"""
function time_reverse(c::CompositeFS{2,N,M,T}) where {N, M, T <: NTuple{2}}
    return CompositeFS{2,N,M,T}(reverse(c.components))
end

"""
    update_component(c::CompositeFS, new, ::Val{i})

Replace the `i`-th component in `c` with `new`. Used for updating a single component in the
address.
"""
function update_component(c::CompositeFS, new, ::Val{I}) where {I}
    return typeof(c)(_update_component(c.components, new, Val(I)))
end

@inline _update_component((a, as...), new, ::Val{1}) = (new, as...)
@inline function _update_component((a, as...), new, ::Val{I}) where {I}
    return (a, _update_component(as, new, Val(I - 1))...)
end

Base.isless(a::T, b::T) where {T<:CompositeFS} = isless(a.components, b.components)

function onr(a::CompositeFS)
    map(onr, a.components)
end

# Convenience
"""
    FermiFS2C <: AbstractFockAddress
    FermiFS2C(onr_a, onr_b)

Fock state address with two fermionic (spin) components. Alias for [`CompositeFS`](@ref)
with two [`FermiFS`](@ref) components. Construct by specifying either two compatible
[`FermiFS`](@ref)s, two [`onr`](@ref)s, or the number of modes followed by `mode =>
occupation_number` pairs, where `occupation_number=1` will put a particle in the first
component and `occupation_number=-1` will put a particle in the second component.
See examples below.

# Examples

```jldoctest
julia> FermiFS2C(FermiFS(1,0,0), FermiFS(0,1,1))
CompositeFS(
  FermiFS{1,3}(1, 0, 0),
  FermiFS{2,3}(0, 1, 1),
)

julia> FermiFS2C((1,0,0), (0,1,1))
CompositeFS(
  FermiFS{1,3}(1, 0, 0),
  FermiFS{2,3}(0, 1, 1),
)

julia> FermiFS2C(3, 1 => 1, 2 => -1, 3 => -1)
CompositeFS(
  FermiFS{1,3}(1, 0, 0),
  FermiFS{2,3}(0, 1, 1),
)

julia> fs"|↑↓↓⟩"
CompositeFS(
  FermiFS{1,3}(1, 0, 0),
  FermiFS{2,3}(0, 1, 1),
)
```
"""
const FermiFS2C{N1,N2,M,N,F1,F2} =
    CompositeFS{2,N,M,Tuple{F1,F2}} where {F1<:FermiFS{N1,M},F2<:FermiFS{N2,M}}

FermiFS2C(f1::FermiFS{<:Any,M}, f2::FermiFS{<:Any,M}) where {M} = CompositeFS(f1, f2)
FermiFS2C(onr_a, onr_b) = FermiFS2C(FermiFS(onr_a), FermiFS(onr_b))
FermiFS2C(M::Integer, pairs::Pair...) = FermiFS2C(M, pairs)
function FermiFS2C(M::Integer, pairs)
    up_pairs = filter(p -> p[2] > 0, pairs)
    down_pairs = map(p -> p[1] => -p[2], filter(p -> p[2] < 0, pairs))
    return FermiFS2C(FermiFS(M, up_pairs), FermiFS(M, down_pairs))
end

function print_address(io::IO, f::FermiFS2C; compact=false)
    if compact
        o1, o2 = onr(f)
        str = join(
            [i && j ? '⇅' : i ? '↑' : j ? '↓' : '⋅' for (i, j) in zip(Bool.(o1), Bool.(o2))]
        )
        print(io, "|", str, "⟩")
    else
        # Show as normal CompositeFS
        invoke(print_address, Tuple{typeof(io),CompositeFS}, io, f)
    end
end

BoseFS2C(fs::CompositeFS{2}) = BoseFS2C(fs.components...)
CompositeFS(fs::BoseFS2C) = CompositeFS(fs.bsa, fs.bsb)
