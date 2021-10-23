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

function Base.show(io::IO, b::BoseFS2C)
    print(io, "BoseFS2C(")
    Base.show(io,b.bsa)
    print(io, ",")
    Base.show(io,b.bsb)
    print(io, ")")
end

num_components(::Type{<:BoseFS2C}) = 2
Base.isless(a::T, b::T) where {T<:BoseFS2C} = isless((a.bsa, a.bsb), (b.bsa, b.bsb))
onr(b2::BoseFS2C) = (onr(b2.bsa), onr(b2.bsb))


function near_uniform(::Type{<:BoseFS2C{NA,NB,M}}) where {NA,NB,M}
    return BoseFS2C(near_uniform(BoseFS{NA,M}), near_uniform(BoseFS{NB,M}))
end

"""
    CompositeFS(addresses::SingleComponentFockAddress...) <: AbstractFockAddress

Used to encode addresses for multi-component models. All component addresses
are expected have the same number of modes.

See also: [`BoseFS`](@ref), [`FermiFS`](@ref), [`SingleComponentFockAddress`](@ref),
[`num_modes`](@ref), [`FermiFS2C`](@ref).
"""
struct CompositeFS{C,N,M,T} <: AbstractFockAddress{N,M}
    components::T
    # C: components, N: total particles, M: modes in each component, T: tuple type with constituent address types
    function CompositeFS{C,N,M,T}(adds::T) where {C,N,M,T}
        return new{C,N,M,T}(adds)
    end
end

# Slow constructor - not to be used internallly
function CompositeFS(adds::Vararg{AbstractFockAddress}) # CompositeFS(adds)
    N = sum(num_particles, adds)
    M1, M2 = extrema(num_modes, adds)
    if M1 ≠ M2
        error("all addresses must have the same number of modes")
    end
    return CompositeFS{length(adds),N,M1,typeof(adds)}(adds)
end

num_components(::CompositeFS{C}) where {C} = C
Base.hash(c::CompositeFS, u::UInt) = hash(c.components, u)

function Base.show(io::IO, c::CompositeFS{C}) where {C}
    println(io, "CompositeFS(")
    for add in c.components
        println(io, "  ", add, ",")
    end
    print(io, ")")
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

Alias for [`CompositeFS`](@ref) with two [`FermiFS`](@ref) components. Construct by
specifying either two compatible [`FermiFS`](@ref)s or [`onr`](@ref)s.
"""
const FermiFS2C{N1,N2,M,N,F1,F2} =
    CompositeFS{2,N,M,Tuple{F1,F2}} where {F1<:FermiFS{N1,M},F2<:FermiFS{N2,M}}
FermiFS2C(onr_a, onr_b) = CompositeFS(FermiFS(onr_a), FermiFS(onr_b))
FermiFS2C(f1::FermiFS{<:Any,M}, f2::FermiFS{<:Any,M}) where {M} = CompositeFS(f1, f2)
