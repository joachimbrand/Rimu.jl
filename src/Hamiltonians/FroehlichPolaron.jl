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
H = \\frac{(P̂_f - P)^2}{m} + ωN̂ - ν Σₖ(âₖ^† + â_{-k})
```

where ``P`` is the total momentum `total_mom`, ``P̂_f = \\frac{M}{L}Σ_k k âₖ^† âₖ`` is the momentum
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
    A<:OccupationNumberFS{M}, # address type
    MC # momentum cutoff indicating type
} <: AbstractHamiltonian{T}
    addr::A
    v::T
    mass::T
    omega::T
    l::T
    p::T
    ks::SVector{M,T} # values for k
    momentum_cutoff::MC
    mode_cutoff::T
end

function FroehlichPolaron(
    addr::OccupationNumberFS;
    v=1.0,
    mass=1.0,
    omega=1.0,
    l=1.0,
    p=0.0,
    momentum_cutoff=nothing,
    mode_cutoff=100.0,
)
    M = num_modes(addr) # this is compile-time information
    if isnothing(momentum_cutoff)
        v, p, mass, omega, l, mode_cutoff = promote(float(v), float(p), float(mass), float(omega), float(l), float(mode_cutoff))
    else
        v, p, mass, omega, l, momentum_cutoff, mode_cutoff = promote(float(v), float(p), float(mass), float(omega), float(l), float(mode_cutoff), float(momentum_cutoff))    
    end

    step = typeof(v)(2π/M)
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    return FroehlichPolaron(addr, v, mass, omega, l, p, ks, momentum_cutoff, mode_cutoff)
end

function Base.show(io::IO, h::FroehlichPolaron)
    println(io, "FroehlichPolaron($(h.addr); v=$(h.v), mass=$(h.mass), omega=$(h.omega), l=$(h.l), p=$(h.p), ")
    println("momentum_cutoff=$(h.momentum_cutoff), mode_cutoff=$(h.mode_cutoff))")
end

function starting_address(h::FroehlichPolaron)
    return h.addr
end

LOStructure(::Type{<:FroehlichPolaron{<:Real}}) = IsHermitian()

# Base.getproperty(h::FroehlichPolaron, s::Symbol) = getproperty(h, Val(s))
# Base.getproperty(h::FroehlichPolaron, ::Val{:ks}) = getfield(h, :ks)
# Base.getproperty(h::FroehlichPolaron, ::Val{:addr}) = getfield(h, :addr)
# Base.getproperty(h::FroehlichPolaron, ::Val{:v}) = getfield(h, :v)
# Base.getproperty(h::FroehlichPolaron, ::Val{:mass}) = getfield(h, :mass)
# Base.getproperty(h::FroehlichPolaron, ::Val{:omega}) = getfield(h, :omega)
# Base.getproperty(h::FroehlichPolaron, ::Val{:l}) = getfield(h, :l)
# Base.getproperty(h::FroehlichPolaron, ::Val{:p}) = getfield(h, :p)

ks(h::FroehlichPolaron) = getfield(h, :ks)

# TODO: Sort out scales for momentum and energy; additional parameters?
# TODO: compute diagonal and off-diagonal elements
# TODO: rest of AbstractHamiltonian interface
# TODO: write unit tests for FroehlichPolaron

function diagonal_element(h::FroehlichPolaron{<:Any,M}, addr::OccupationNumberFS{M}) where {M}
    map = onr(addr)
    p_f = (M/h.l) * dot(h.ks,map)
    return h.omega * num_particles(addr) + (h.p - p_f)^2 / h.mass
end

function num_offdiagonals(::FroehlichPolaron{<:Any,M}, ::OccupationNumberFS{M}) where {M}
    return 2M #num_occupied_modes
end

function get_offdiagonal(h::FroehlichPolaron{<:Any,M,<:Any,<:Nothing}, addr::OccupationNumberFS{M},chosen) where {M}
    return _froehlich_offdiag(h,addr,chosen)
end

function get_offdiagonal(h::FroehlichPolaron{T,M,<:Any,T}, addr::OccupationNumberFS{M},chosen) where {T,M}
    #branch for momentum cutoff
    naddress, value = _froehlich_offdiag(h,addr,chosen)
    
    new_p_tot = dot(h.ks, onr(naddress))
    if new_p_tot > h.momentum_cutoff # check if momentum of new address exceeds momentum_cutoff
        return addr, 0.0
    else
        return naddress, - h.v * value
    end
end

function _froehlich_offdiag(h, addr::OccupationNumberFS{M},chosen) where {M}
    if chosen ≤ M # assign first M indices to creations
        if onr(addr)[chosen] ≥ h.mode_cutoff # check whether occupation exceeds cutoff
            return addr, 0.0
        else
            naddress, value = excitation(addr, (chosen,),())
            return naddress, - h.v * value
        end
    else # remaining indices are destructions

        naddress, value = excitation(addr, (),(chosen-M,))
        return naddress, - h.v * value
    end
end