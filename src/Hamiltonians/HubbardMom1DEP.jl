"""
    HubbardMom1DEP(address; u=1.0, t=1.0, v_ho=1.0, dispersion=hubbard_dispersion)

Implements a one-dimensional Bose Hubbard chain in momentum space with harmonic external
potential.

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
struct HubbardMom1DEP{TT,M,AD<:AbstractFockAddress,U,T} <: AbstractHamiltonian{TT}
    add::AD # default starting address, should have N particles and M modes
    ks::SVector{M,TT} # values for k
    kes::SVector{M,TT} # values for kinetic energy
    ep::SVector{M,TT}
end

function HubbardMom1DEP(
    add::Union{BoseFS,FermiFS2C};
    u=1.0, t=1.0, dispersion = hubbard_dispersion, v_ho=1.0,
)
    M = num_modes(add)
    U, T, V = promote(float(u), float(t), float(v_ho))
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

    # Set up potential like in Real1DEP
    is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
    js = shift_lattice(is) # shifted such that js[1] = 0
    real_potential = [V*j^2 for j in js]
    @assert all(isreal, fft(real_potential))
    potential = SVector{M}(real.(fft(real_potential)))

    return HubbardMom1DEP{typeof(U),M,typeof(add),U,T}(add, ks, kes, potential)
end

function Base.show(io::IO, h::HubbardMom1DEP)
    print(io, "HubbardMom1DEP($(h.add); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardMom1DEP)
    return h.add
end

LOStructure(::Type{<:HubbardMom1DEP{<:Real}}) = IsHermitian()

Base.getproperty(h::HubbardMom1DEP, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardMom1DEP, ::Val{:ep}) = getfield(h, :ep)
Base.getproperty(h::HubbardMom1DEP, ::Val{:ks}) = getfield(h, :ks)
Base.getproperty(h::HubbardMom1DEP, ::Val{:kes}) = getfield(h, :kes)
Base.getproperty(h::HubbardMom1DEP, ::Val{:add}) = getfield(h, :add)
Base.getproperty(h::HubbardMom1DEP{<:Any,<:Any,<:Any,U}, ::Val{:u}) where {U} = U
Base.getproperty(h::HubbardMom1DEP{<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where {T} = T

ks(h::HubbardMom1DEP) = getfield(h, :ks)

@inline function momentum_transfer_diagonal(
    h::HubbardMom1DEP{<:Any,M,<:BoseFS}, map
) where {M}
    return h.u / 2M * momentum_transfer_diagonal(map)
end
@inline function momentum_transfer_diagonal(
    h::HubbardMom1DEP{<:Any,M,<:FermiFS2C}, map_a, map_b
) where {M}
    return h.u / 2M * momentum_transfer_diagonal(map_a, map_b)
end

@inline function diagonal_element(h::HubbardMom1DEP, add::BoseFS)
    map = OccupiedModeMap(add)
    return kinetic_energy(h.kes, map) + momentum_transfer_diagonal(h, map) +
        momentum_external_potential_diagonal(h.ep, add, map)
end
@inline function diagonal_element(h::HubbardMom1DEP, add::FermiFS2C)
    map_a = OccupiedModeMap(add.components[1])
    map_b = OccupiedModeMap(add.components[2])
    return kinetic_energy(h.kes, map_a) +
        kinetic_energy(h.kes, map_b) +
        momentum_transfer_diagonal(h, map_a, map_b) +
        momentum_external_potential_diagonal(h.ep, map_a) +
        momentum_external_potential_diagonal(h.ep, map_b)
end

###
### offdiagonals
###
struct OffdiagonalsBoseMom1DEP{
    A<:BoseFS,T,H<:AbstractHamiltonian{T},O<:OccupiedModeMap
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    num_mom::Int
    num_ep::Int
    map::O
end

function offdiagonals(h::HubbardMom1DEP, a::BoseFS)
    M = num_modes(a)
    map = OccupiedModeMap(a)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    num_mom = singlies * (singlies - 1) * (M - 2) + doublies * (M - 1)
    num_ep = singlies * (M - 1)
    return OffdiagonalsBoseMom1DEP(h, a, num_mom, num_ep, map)
end

function Base.getindex(s::OffdiagonalsBoseMom1DEP{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    end
    M = num_modes(s.address)
    if i ≤ s.num_mom
        new_address, onproduct = momentum_transfer_excitation(s.address, i, s.map)
        matrix_element = s.hamiltonian.u/(2*M) * onproduct
    else
        i -= s.num_mom
        new_address, matrix_element = momentum_external_potential_excitation(
            s.hamiltonian.ep, s.address, i, s.map
        )
    end
    return (new_address, matrix_element)
end

Base.size(s::OffdiagonalsBoseMom1DEP) = (s.num_mom + s.num_ep,)
