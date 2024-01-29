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
    T, # eltype
    M, # number of modes
    A<:OccupationNumberFS{M} # address type
} <: AbstractHamiltonian{T}
    addr::A
    v::T
    mass::T
    omega::T
    l::T
    p::T
    ks::SVector{M,T} # values for k
end

function FroehlichPolaron(
    addr::OccupationNumberFS;
    v=1.0,
    mass=1.0,
    omega=1.0,
    l=1.0,
    p=0.0,
)
    M = num_modes(addr) # this is compile-time information
    v, p, mass, omega, l = promote(float(v), float(p), float(mass), float(omega), float(l))
    step = typeof(v)(2π/M)
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    return FroehlichPolaron(addr, v, mass, omega, l, p, ks)
end

function Base.show(io::IO, h::FroehlichPolaron)
    print(io, "FroehlichPolaron($(h.addr); v=$(h.v), mass=$(h.mass), omega=$(h.omega), l=$(h.l), p=$(h.p))")
end

function starting_address(h::FroehlichPolaron)
    return getfield(h, :addr)
end

LOStructure(::Type{<:FroehlichPolaron{<:Real}}) = IsHermitian()

Base.getproperty(h::FroehlichPolaron, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::FroehlichPolaron, ::Val{:ks}) = getfield(h, :ks)
Base.getproperty(h::FroehlichPolaron, ::Val{:addr}) = getfield(h, :addr)
Base.getproperty(h::FroehlichPolaron, ::Val{:v}) = getfield(h, :v)
Base.getproperty(h::FroehlichPolaron, ::Val{:mass}) = getfield(h, :mass)
Base.getproperty(h::FroehlichPolaron, ::Val{:omega}) = getfield(h, :omega)
Base.getproperty(h::FroehlichPolaron, ::Val{:l}) = getfield(h, :l)
Base.getproperty(h::FroehlichPolaron, ::Val{:p}) = getfield(h, :p)

ks(h::FroehlichPolaron) = getfield(h, :ks)

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
