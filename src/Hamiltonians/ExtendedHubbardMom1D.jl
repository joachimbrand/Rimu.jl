"""
    ExtendedHubbardMom1D(
        address; 
        u=1.0, t=1.0, v=1.0, dispersion=hubbard_dispersion, boundary_condition = 0.0
    )

Implements a one-dimensional extended Hubbard chain, also known as the ``t - V`` model, 
in momentum space.

```math
\\hat{H} =  \\sum_{k} ϵ_k n_k + \\frac{1}{2M} \\sum_{kpqr} (u + 2v \\cos(q-p)) a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.
* `boundary_condition`: `θ <: Number`: hopping over the boundary incurs a
    factor ``\\exp(iθ)`` for a hop to the right and ``\\exp(−iθ)`` for a hop to the left.

* `dispersion`: defines ``ϵ_k =``` dispersion(t, k + θ)`
    - [`hubbard_dispersion`](@ref): ``ϵ_k = -2 (\\Re(t) \\cos(k + θ) + \\Im(t) \\sin(k + θ))``
    - [`continuum_dispersion`](@ref): ``ϵ_k = \\Re(t) (k + θ)^2 - 2 \\Im(t) (k + θ)``

# See also

* [`HubbardMom1D`](@ref)
* [`HubbardReal1D`](@ref)
* [`ExtendedHubbardReal1D`](@ref)
"""
struct ExtendedHubbardMom1D{TT,M,AD<:AbstractFockAddress,U,V,T,BOUNDARY_CONDITION} <: AbstractHamiltonian{TT}
    address::AD # default starting address, should have N particles and M modes
    ks::SVector{M,TT} # values for k
    kes::SVector{M,TT} # values for kinetic energy
end

function ExtendedHubbardMom1D(
    address::SingleComponentFockAddress;
    u=1.0, v=1.0, t=1.0, dispersion = hubbard_dispersion, boundary_condition = 0.0
)
    M = num_modes(address)
    U, V, T= promote(float(u), float(v), float(t))
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    kes = SVector{M}(dispersion.(T , kr .+ boundary_condition))
    return ExtendedHubbardMom1D{typeof(U),M,typeof(address),U,V,T,boundary_condition}(address, ks, kes)
end

function Base.show(io::IO, h::ExtendedHubbardMom1D)
    compact_addr = repr(h.address, context=:compact => true) # compact print address
    print(io, "ExtendedHubbardMom1D($(compact_addr); u=$(h.u), v=$(h.v), t=$(h.t), boundary_condition=$(h.boundary_condition))")
end

function starting_address(h::ExtendedHubbardMom1D)
    return h.address
end

dimension(::ExtendedHubbardMom1D, address) = number_conserving_dimension(address)

LOStructure(::Type{<:ExtendedHubbardMom1D{<:Real}}) = IsHermitian()

Base.getproperty(h::ExtendedHubbardMom1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::ExtendedHubbardMom1D, ::Val{:ks}) = getfield(h, :ks)
Base.getproperty(h::ExtendedHubbardMom1D, ::Val{:kes}) = getfield(h, :kes)
Base.getproperty(h::ExtendedHubbardMom1D, ::Val{:address}) = getfield(h, :address)
Base.getproperty(h::ExtendedHubbardMom1D{<:Any,<:Any,<:Any,U}, ::Val{:u}) where {U} = U
Base.getproperty(h::ExtendedHubbardMom1D{<:Any,<:Any,<:Any,<:Any,V}, ::Val{:v}) where {V} = V
Base.getproperty(h::ExtendedHubbardMom1D{<:Any,<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where {T} = T
Base.getproperty(h::ExtendedHubbardMom1D{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,BOUNDARY_CONDITION}, 
    ::Val{:boundary_condition}) where {BOUNDARY_CONDITION} = BOUNDARY_CONDITION

ks(h::ExtendedHubbardMom1D) = getfield(h, :ks)

# standard interface function
function num_offdiagonals(ham::ExtendedHubbardMom1D, address::SingleComponentFockAddress)
    singlies, doublies = num_singly_doubly_occupied_sites(address)
    return num_offdiagonals(ham, address, singlies, doublies)
end

# 4-argument version
@inline function num_offdiagonals(ham::ExtendedHubbardMom1D, ::SingleComponentFockAddress, singlies, doublies)
    M = num_modes(ham)
    return singlies * (singlies - 1) * (M - 2) + doublies * (M - 1)
end

@inline function momentum_transfer_diagonal(
    h::ExtendedHubbardMom1D{<:Any,M,<:BoseFS}, map
) where {M}
    return (h.u/ 2M) * momentum_transfer_diagonal(map) 
    + (h.v/ M) * extended_momentum_transfer_diagonal(map, (2*π)/M) 
end

@inline function momentum_transfer_diagonal(
    h::ExtendedHubbardMom1D{<:Any,M,<:FermiFS}, map
) where {M}
    return (h.v/ M) * extended_momentum_transfer_diagonal(map, M)
end

@inline function diagonal_element(h::ExtendedHubbardMom1D, address::SingleComponentFockAddress)
    map = OccupiedModeMap(address)
    return dot(h.kes, map) + momentum_transfer_diagonal(h, map)
end

@inline function get_offdiagonal(
    ham::ExtendedHubbardMom1D{<:Any,M,A}, address::A, chosen, map=OccupiedModeMap(address)
) where {M,A<:BoseFS}
    address, onproduct,_,_,q = extended_momentum_transfer_excitation(address, chosen, map)
    return address, ham.u/(2*M)*onproduct + ((ham.v * cos((q)*((2*π)/M)))/(M))*onproduct
end

@inline function get_offdiagonal(
    ham::ExtendedHubbardMom1D{<:Any,M,A}, address::A, chosen, map=OccupiedModeMap(address)
) where {M,A<:FermiFS}
    address, onproduct,_,_,q = extended_momentum_transfer_excitation(address, chosen, map)
    return address, -((ham.v * cos((q)*((2*π)/M)))/(M))*onproduct
end
momentum(ham::ExtendedHubbardMom1D) = MomentumMom1D(ham)
