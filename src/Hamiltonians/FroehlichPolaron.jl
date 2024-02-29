"""
    FroehlichPolaron(address::OccupationNumberFS{M}; kwargs...) <: AbstractHamiltonian


The Froehlich polaron Hamiltonian for a 1D lattice with `M` momentum modes is given by

```math
H = (p̂_f - p)^2/m + ωN̂ - v Σₖ(âₖ^† + â₋ₖ)
```

where ``p`` is the total momentum, ``p̂_f = Σ_k k âₖ^† âₖ`` is the momentum operator for the
bosons, and ``k`` part of the momentum lattice with separation ``2π/l``. ``N̂`` is the number
operator for the bosons.


# Keyword Arguments

* `p=0.0`: the total momentum ``p``.
* `v=1.0`: the coupling strength ``v``.
* `mass=1.0`: the particle mass ``m``.
* `omega=1.0`: the oscillation frequency of the phonons ``ω``.
* `l=1.0`: the box size in real space ``l``. Provides scale parameter of the momentum
    lattice.
* `momentum_cutoff=nothing`: the maximum boson momentum allowed for an address.
* `mode_cutoff`: the maximum number of bosons in each momentum mode. Defaults to the maximum
    value supported by the address type [`OccupationNumberFS`](@ref).

# Examples
```jldoctest
julia> fs = OccupationNumberFS(0,0,0)
OccupationNumberFS{3, UInt8}(0, 0, 0)

julia> ham = FroehlichPolaron(fs; v=0.5)
FroehlichPolaron(fs"|0 0 0⟩{8}"; v=0.5, mass=1.0, omega=1.0, l=1.0, p=0.0, mode_cutoff=255)

julia> dimension(ham)
16777216

julia> dimension(FroehlichPolaron(fs; v=0.5, mode_cutoff=5))
216
```

See also [`OccupationNumberFS`](@ref), [`dimension`](@ref), [`AbstractHamiltonian`](@ref).
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
    mode_cutoff::Int
end

function FroehlichPolaron(
    addr::OccupationNumberFS{M,AT};
    v=1.0,
    mass=1.0,
    omega=1.0,
    l=1.0,
    p=0.0,
    momentum_cutoff=nothing,
    mode_cutoff=nothing,
) where {M,AT}
    if l ≤ 0
        throw(ArgumentError("l must be positive"))
    end

    v, p, mass, omega, l = promote(float(v), float(p), float(mass), float(omega), float(l))

    step = typeof(v)(2π/M)
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = (M/l)*range(start; step = step, length = M)
    ks = SVector{M}(kr)

    if !isnothing(momentum_cutoff)
        momentum_cutoff = typeof(v)(momentum_cutoff)
        momentum = dot(ks,onr(addr))
        if momentum > momentum_cutoff
            throw(ArgumentError("Starting address has momentum $momentum which cannot exceed momentum_cutoff $momentum_cutoff"))
        end
    end

    if isnothing(mode_cutoff)
        mode_cutoff = Int(typemax(AT))
    end
    mode_cutoff = floor(Int, mode_cutoff)::Int
    if _exceed_mode_cutoff(mode_cutoff, addr)
        throw(ArgumentError("Starting address cannot have occupations that exceed mode_cutoff"))
    end
    return FroehlichPolaron(addr, v, mass, omega, l, p, ks, momentum_cutoff, mode_cutoff)
end

function Base.show(io::IO, h::FroehlichPolaron)
    print(io, "FroehlichPolaron(")
    show(IOContext(io, :compact => true), h.addr)
    print("; v=$(h.v), mass=$(h.mass), omega=$(h.omega), l=$(h.l), p=$(h.p), ")
    isnothing(h.momentum_cutoff) || print(io, "momentum_cutoff = $(h.momentum_cutoff), ")
    print(io, "mode_cutoff=$(h.mode_cutoff))")

end

function starting_address(h::FroehlichPolaron)
    return h.addr
end

LOStructure(::Type{<:FroehlichPolaron{<:Real}}) = IsHermitian()

function diagonal_element(h::FroehlichPolaron{<:Any,M}, addr::OccupationNumberFS{M}) where {M}
    map = onr(addr)
    p_f = dot(h.ks,map)
    return h.omega * num_particles(addr) + (h.p - p_f)^2 / h.mass
end

function num_offdiagonals(::FroehlichPolaron{<:Any,M}, ::OccupationNumberFS{M}) where {M}
    return 2M #num_occupied_modes
end

function get_offdiagonal(h::FroehlichPolaron{<:Any,M,<:Any,Nothing}, addr::OccupationNumberFS{M},chosen) where {M}
    # branch that bypasses momentum cutoff
    return _froehlich_offdiag(h, addr, chosen)
end

function get_offdiagonal(h::FroehlichPolaron{T,M,<:Any,T}, addr::OccupationNumberFS{M}, chosen) where {M,T}
    # branch for checking momentum cutoff
    naddress, value = _froehlich_offdiag(h, addr, chosen)

    new_p_tot = dot(h.ks, onr(naddress))
    if (M/h.l) * new_p_tot > h.momentum_cutoff # check if momentum of new address exceeds momentum_cutoff
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
            naddress, value = excitation(addr, (chosen,), ())
            return naddress, - h.v * value
        end
    else # remaining indices are destructions

        naddress, value = excitation(addr, (), (chosen - M,))
        return naddress, - h.v * value
    end
end

function _exceed_mode_cutoff(mode_cutoff, addr::OccupationNumberFS{M}) where {M}
    return any(x -> x > mode_cutoff, onr(addr))
end

function dimension(h::FroehlichPolaron, address)
    # takes into account `mode_cutoff` but not `momentum_cutoff`
    M = num_modes(address)
    n = h.mode_cutoff
    return BigInt(n + 1)^BigInt(M)
end
