"""
    BoseFS2C{NA,NB,M,AA,AB} <: AbstractFockAddress

Address type that constructed with two [`BoseFS{N,M,S}`](@ref). It represents a
Fock state with two components, e.g. two different species of bosons with particle
number `NA` from species S and particle number `NB` from species B. The number of
modes `M` is expected to be the same for both components.
"""
struct BoseFS2C{NA,NB,M,SA,SB} <: AbstractFockAddress
    bsa::BoseFS{NA,M,SA}
    bsb::BoseFS{NB,M,SB}
end

BoseFS2C(onr_a::Tuple, onr_b::Tuple) = BoseFS2C(BoseFS(onr_a),BoseFS(onr_b))

function Base.show(io::IO, b::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    print(io, "BoseFS2C(")
    Base.show(io,b.bsa)
    print(io, ",")
    Base.show(io,b.bsb)
    print(io, ")")
end

num_particles(::Type{<:BoseFS2C{NA,NB}}) where {NA,NB} = NA + NB
num_modes(::Type{<:BoseFS2C{<:Any,<:Any,M}}) where {M} = M
num_components(::Type{<:BoseFS2C}) = 2
Base.isless(a::T, b::T) where {T<:BoseFS2C} = isless((a.bsa, a.bsb), (b.bsa, b.bsb))

function near_uniform(::Type{<:BoseFS2C{NA,NB,M}}) where {NA,NB,M}
    return BoseFS2C(near_uniform(BoseFS{NA,M}), near_uniform(BoseFS{NB,M}))
end

const SingleFS{M} = Union{BoseFS{<:Any,M},FermiFS{<:Any,M}}

"""
    CompositeFS(addresses...) <: AbstractFockAddress

Used to encode addresses for multi-component models.
"""
struct CompositeFS{C,M,T<:NTuple{C,SingleFS{M}}} <: AbstractFockAddress
    components::T
end

CompositeFS(adds::Vararg{AbstractFockAddress}) = CompositeFS(adds)

num_components(::CompositeFS{C}) where {C} = C
num_modes(::CompositeFS{<:Any,M}) where {M} = M
num_particles(c::CompositeFS) = sum(num_particles, c.components)
Base.hash(c::CompositeFS, u::UInt) = hash(c.components, u)

function Base.show(io::IO, c::CompositeFS{C}) where {C}
    println(io, "CompositeFS(")
    for add in c.components
        println(io, "  ", add, ",")
    end
    print(io, ")")
end

function update_component(c::CompositeFS, new, ::Val{I}) where {I}
    return typeof(c)(_update_component(c.components, new, Val(I)))
end

@inline _update_component((a, as...), new, ::Val{1}) = (new, as...)
@inline function _update_component((a, as...), new, ::Val{I}) where {I}
    return (a, _update_component(as, new, Val(I - 1))...)
end

Base.isless(a::T, b::T) where {T<:CompositeFS} = isless(a.components, b.components)
