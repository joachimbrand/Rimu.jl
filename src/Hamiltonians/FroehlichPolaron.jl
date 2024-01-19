"""
    FroehlichPolaron(
        address;
        alpha = 1.0,
        total_mom = 0.0,
        geometry = nothing,
        num_dimensions = 1
    )

The Froehlich polaron Hamiltonian is given by

```math
H = (P̂_f - P)^2 + N̂ + Σₖ νₖ (âₖ^† + â_{-k})
```

where ``P`` is the total momentum `total_mom`, ``P̂_f = Σ_k k âₖ^† âₖ`` is the momentum
operator for the bosons, ``N̂ = Σ_k âₖ^† âₖ`` is the number operator for the bosons, and
``νₖ = \\sqrt{α}/|k|`` is the coupling strength determined by the coupling parameter
`α == alpha`.

The optional `geometry` argument specifies the geometry of the lattice for ``k``-space and
should be of the type [`PeriodicBoundaries`](@ref). A simplified way of specifying the
geometry is to provide the number of dimensions `num_dimensions`. In this case the
[`num_modes(address)`](@ref) of `address` must be a square number for `num_dimensions = 2`,
or a cube number for `num_dimensions = 3`.

The `address` must be of type [`OccupationNumberFS`](@ref).
"""
struct FroehlichPolaron{
    P, # total momentum
    T, # eltype
    A<:OccupationNumberFS, # address type
    G # lattice type
} <: AbstractHamiltonian{T}
    addr::A
    alpha::T
    geometry::G
    # nu_k::Vector{T}
    # p_squared_k::Vector{T}
end

function FroehlichPolaron(
    addr::OccupationNumberFS;
    geometry=nothing,
    alpha=1.0,
    total_mom=0.0,
    num_dimensions=1
)
    M = num_modes(addr) # this is compile-time information
    if isnothing(geometry)
        if num_dimensions == 1
            geometry = PeriodicBoundaries(M)
        elseif num_dimensions == 2
            sm = round(Int, sqrt(M))
            if sm^2 == M
                geometry = PeriodicBoundaries(sm, sm)
            else
                throw(ArgumentError("Number of modes $M is not a square. Specify geometry."))
            end
        elseif num_dimensions == 3
            cm = round(Int, cbrt(M))
            if cm^3 == M
                geometry = PeriodicBoundaries(cm, cm, cm)
            else
                throw(ArgumentError("Number of modes $M is not a cube. Specify geometry."))
            end
        else
            throw(ArgumentError("Invalid number of dimensions: $num_dimensions"))
        end
    end
    geometry isa PeriodicBoundaries || throw(ArgumentError("Invalid geometry: $geometry"))
    alpha, P = promote(float(alpha), float(total_mom))
    return FroehlichPolaron{P,typeof(P),typeof(addr),typeof(geometry)}(addr, alpha, geometry)
end

function Base.show(io::IO, h::FroehlichPolaron)
    print(io, "FroehlichPolaron($(h.addr); alpha=$(h.alpha), total_mom=$(h.total_mom), ")
    print(io, "geometry=$(h.geometry))")
end

function starting_address(h::FroehlichPolaron)
    return getfield(h, :addr)
end

LOStructure(::Type{<:FroehlichPolaron{<:Real}}) = IsHermitian()

Base.getproperty(h::FroehlichPolaron, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::FroehlichPolaron, ::Val{:addr}) = getfield(h, :addr)
Base.getproperty(h::FroehlichPolaron, ::Val{:alpha}) = getfield(h, :alpha)
Base.getproperty(::FroehlichPolaron{P}, ::Val{:total_mom}) where P = P
Base.getproperty(h::FroehlichPolaron, ::Val{:geometry}) = getfield(h, :geometry)

num_dimensions(h::FroehlichPolaron) = num_dimensions(h.geometry)

# function diagonal_element(h::FroehlichPolaron, addr::OccupationNumberFS)
#     return (
#         (-h.total_mom)^2 +
#         num_particles(addr) +
#         sum(νk(h, k) for k in momentum(h, addr))
#     )
# end

# TODO: Sort out scales for momentum and energy; additional parameters?
# TODO: pre-compute and store grid for p_squared_k (k.e.) and nu_k (coupling strength)
# TODO: compute diagonal and off-diagonal elements
# TODO: rest of AbstractHamiltonian interface
# TODO: write unit tests for FroehlichPolaron
