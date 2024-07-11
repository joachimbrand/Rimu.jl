#const BoseFS2C{N1,N2,M,S1,S2,N} = CompositeFS{2,N,M,Tuple{BoseFS{N1,M,S1},BoseFS{N2,M,S2}}}

"""
    hubbard_dispersion(k)
Dispersion relation for [`HubbardMom1D`](@ref). Returns `-2cos(k)`.

See also [`continuum_dispersion`](@ref).
"""
hubbard_dispersion(k) = -2cos(k)

"""
    continuum_dispersion(k)
Dispersion relation for [`HubbardMom1D`](@ref). Returns `k^2`.

See also [`hubbard_dispersion`](@ref).
"""
continuum_dispersion(k) = k^2

"""
    HubbardMom1D(address; u=1.0, t=1.0, dispersion=hubbard_dispersion)

Implements a one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} =  \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.
* `dispersion`: defines ``ϵ_k =``` t*dispersion(k)`
    - [`hubbard_dispersion`](@ref): ``ϵ_k = -2t \\cos(k)``
    - [`continuum_dispersion`](@ref): ``ϵ_k = tk^2``

# See also

* [`HubbardReal1D`](@ref)
* [`ExtendedHubbardReal1D`](@ref)
"""
struct HubbardMom1D{T,M,A<:AbstractFockAddress} <: AbstractHamiltonian{T}
    address::A
    u::T
    t::T
    ks::SVector{M,T}  # values for k
    kes::SVector{M,T} # values for kinetic energy
end

function HubbardMom1D(address; u=1.0, t=1.0, dispersion=hubbard_dispersion)
    M = num_modes(address)
    U, T = promote(float(u), float(t))
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    # kes = SVector{M}(-2T*cos.(kr))
    kes = SVector{M}(T .* dispersion.(kr))
    return HubbardMom1D{typeof(U),M,typeof(address)}(address, U, T, ks, kes)
end

function Base.show(io::IO, h::HubbardMom1D)
    compact_addr = repr(h.address, context=:compact => true) # compact print address
    print(io, "HubbardMom1D($(compact_addr); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardMom1D)
    return h.address
end

dimension(::HubbardMom1D, address) = number_conserving_dimension(address)

LOStructure(::Type{<:HubbardMom1D{<:Real}}) = IsHermitian()

@inline function diagonal_element(h::HubbardMom1D, address::SingleComponentFockAddress)
    map = OccupiedModeMap(address)
    M = num_modes(address)
    return dot(h.kes, map) + h.u / 2M * momentum_transfer_diagonal(map)
end
@inline function diagonal_element(h::HubbardMom1D, address::CompositeFS)
    occupied_mode_maps = map(OccupiedModeMap, address.components)
    M = num_modes(address)
    mom_transfer = momentum_transfer_diagonal(occupied_mode_maps)
    kinetic_energies = sum(occupied_mode_maps) do occ
        dot(h.kes, occ)
    end
    return kinetic_energies + h.u / 2M * mom_transfer
end

###
### offdiagonals
###
struct HubbardMom1DOffdiagonals{A,M<:MomentumTransferExcitation{A},T} <: AbstractOffdiagonals{A,T}
    u::T
    excitation::M
end
function offdiagonals(h::HubbardMom1D, address)
    return HubbardMom1DOffdiagonals(
        h.u, MomentumTransferExcitation(address)
    )
end

function Base.getindex(o::HubbardMom1DOffdiagonals, i)
    new_address, value, _ = o.excitation[i]
    return new_address, value * o.u / 2num_modes(new_address)
end
Base.size(o::HubbardMom1DOffdiagonals) = size(o.excitation)

get_offdiagonal(h::HubbardMom1D, address, i) = offdiagonals(h, address)[i]
num_offdiagonals(h::HubbardMom1D, address) = length(offdiagonals(h, address))



###
### momentum
###
struct MomentumMom1D{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    ham::H
end
LOStructure(::Type{MomentumMom1D{H,T}}) where {H,T <: Real} = IsDiagonal()
num_offdiagonals(ham::MomentumMom1D, _) = 0
diagonal_element(mom::MomentumMom1D, address) = mod1(onr(address)⋅ks(mom.ham) + π, 2π) - π
# fold into (-π, π]
starting_address(mom::MomentumMom1D) = starting_address(mom.ham)

momentum(ham::HubbardMom1D) = MomentumMom1D(ham)
