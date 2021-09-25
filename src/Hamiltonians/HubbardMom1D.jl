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
struct HubbardMom1D{TT,M,AD<:AbstractFockAddress,U,T} <: AbstractHamiltonian{TT}
    add::AD # default starting address, should have N particles and M modes
    ks::SVector{M,TT} # values for k
    kes::SVector{M,TT} # values for kinetic energy
end

function HubbardMom1D(
    add::BoseFS{<:Any,M};
    u=1.0, t=1.0, dispersion = hubbard_dispersion,
) where {M}
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
    return HubbardMom1D{typeof(U),M,typeof(add),U,T}(add, ks, kes)
end

function Base.show(io::IO, h::HubbardMom1D)
    print(io, "HubbardMom1D($(h.add); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardMom1D)
    return h.add
end

LOStructure(::Type{<:HubbardMom1D{<:Real}}) = IsHermitian()

Base.getproperty(h::HubbardMom1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardMom1D, ::Val{:ks}) = getfield(h, :ks)
Base.getproperty(h::HubbardMom1D, ::Val{:kes}) = getfield(h, :kes)
Base.getproperty(h::HubbardMom1D, ::Val{:add}) = getfield(h, :add)
Base.getproperty(h::HubbardMom1D{<:Any,<:Any,<:Any,U}, ::Val{:u}) where {U} = U
Base.getproperty(h::HubbardMom1D{<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where {T} = T

ks(h::HubbardMom1D) = getfield(h, :ks)

"""
    num_singly_doubly_occupied_sites(address)

Returns the number of singly and doubly occupied sites for a bosonic bit string address.

# Example

```jldoctest
julia> Hamiltonians.num_singly_doubly_occupied_sites(BoseFS{3,3}((1, 1, 1)))
(3, 0)
julia> Hamiltonians.num_singly_doubly_occupied_sites(BoseFS{3,3}((2, 0, 1)))
(2, 1)
```
"""
function num_singly_doubly_occupied_sites(b::BoseFS)
    singlies = 0
    doublies = 0
    for (n, _, _) in occupied_modes(b)
        singlies += 1
        doublies += n > 1
    end
    return singlies, doublies
end

function num_singly_doubly_occupied_sites(onrep::AbstractArray)
    # this one is faster by about a factor of 2 if you already have the onrep
    # returns number of singly and doubly occupied sites
    singlies = 0
    doublies = 0
    for n in onrep
        singlies += n > 0
        doublies += n > 1
    end
    return singlies, doublies
end

# standard interface function
function num_offdiagonals(ham::HubbardMom1D, add::BoseFS)
    singlies, doublies = num_singly_doubly_occupied_sites(add)
    return num_offdiagonals(ham, add, singlies, doublies)
end

# 4-argument version
@inline function num_offdiagonals(ham::HubbardMom1D, add::BoseFS, singlies, doublies)
    M = num_modes(ham)
    return singlies * (singlies - 1) * (M - 2) + doublies * (M - 1)
end

"""
    interaction_energy_diagonal(H, onr)

Compute diagonal interaction energy term.

# Example

```jldoctest
julia> a = BoseFS{6,5}((1,2,3,0,0))
BoseFS{6,5}((1, 2, 3, 0, 0))

julia> H = HubbardMom1D(a);


julia> Hamiltonians.interaction_energy_diagonal(H, onr(a))
5.2
```
"""
@inline function interaction_energy_diagonal(
    h::HubbardMom1D{<:Any,M,<:BoseFS}, onrep::StaticVector{M,I}
) where {M,I}
    # now compute diagonal interaction energy
    onproduct = zero(I) # Σ_kp < c^†_p c^†_k c_k c_p >
    # Note: because of the double for-loop, this is more efficient if done with ONR
    for p in 1:M
        iszero(onrep[p]) && continue
        onproduct += onrep[p] * (onrep[p] - one(I))
        for k in 1:p-1
            onproduct += I(4) * onrep[k] * onrep[p]
        end
    end
    return h.u / 2M * onproduct
end

function kinetic_energy(h::HubbardMom1D, add::AbstractFockAddress)
    onrep = onr(add)
    return kinetic_energy(h, onrep)
end

@inline function kinetic_energy(h::HubbardMom1D, onrep::StaticVector)
    return onrep ⋅ h.kes # safe as onrep is Real
end

@inline function diagonal_element(h::HubbardMom1D, add)
    onrep = BitStringAddresses.m_onr(add)
    return diagonal_element(h, onrep)
end

@inline function diagonal_element(h::HubbardMom1D, onrep::StaticVector)
    return kinetic_energy(h, onrep) + interaction_energy_diagonal(h, onrep)
end

@inline function get_offdiagonal(ham::HubbardMom1D, add, chosen)
    get_offdiagonal(ham, add, chosen, num_singly_doubly_occupied_sites(add)...)
end

"""
    momentum_transfer_excitation(add, chosen, singlies, doublies)
Internal function used in [`get_offdiagonal`](@ref) for [`HubbardMom1D`](@ref)
and [`G2Correlator`](@ref). Returns the new address, the onproduct,
and the change in momentum.
"""
@inline function momentum_transfer_excitation(
    add::SingleComponentFockAddress{<:Any,M}, chosen, singlies, doublies
) where {M}
    double = chosen - singlies * (singlies - 1) * (M - 2)

    if double > 0
        # Both moves from the same mode.
        double, mom_change = fldmod1(double, M - 1)
        idx = find_occupied_mode(add, double, 2)
        src_indices = (idx, idx)
    else
        # Moves from different modes.
        pair, mom_change = fldmod1(chosen, M - 2)
        first, second = fldmod1(pair, singlies - 1) # where the holes are to be made
        if second < first # put them in ascending order
            f_hole = second
            s_hole = first
        else
            f_hole = first
            s_hole = second + 1 # as we are counting through all singlies
        end
        src_indices = find_occupied_mode(add, (f_hole, s_hole))
        f_mode, s_mode = src_indices[1].mode, src_indices[2].mode
        if mom_change ≥ s_mode - f_mode
            mom_change += 1 # to avoid putting particles back into the holes
        end
    end
    # For higher dimensions, replace mod1 here with some geometry.
    dst_modes = mod1.(getindex.(src_indices, 2) .+ (mom_change, -mom_change), M)
    dst_indices = find_mode(add, dst_modes)
    return excitation(add, dst_indices, src_indices)..., -mom_change
end

@inline function get_offdiagonal(
    ham::HubbardMom1D{<:Any,M,A}, add, chosen, singlies, doublies
) where {M,A}
    add, onproduct, _ = momentum_transfer_excitation(add, chosen, singlies, doublies)
    return add, ham.u/(2*M)*onproduct
end

###
### offdiagonals
###
"""
    OffdiagonalsBoseMom1D

Specialized [`AbstractOffdiagonals`](@ref) that keeps track of singly and doubly occupied
sites in current address.
"""
struct OffdiagonalsBoseMom1D{
    A<:BoseFS,T,H<:AbstractHamiltonian{T}
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    singlies::Int
    doublies::Int
end

function offdiagonals(h::HubbardMom1D, a::BoseFS)
    singlies, doublies = num_singly_doubly_occupied_sites(a)
    num = num_offdiagonals(h, a, singlies, doublies)
    return OffdiagonalsBoseMom1D(h, a, num, singlies, doublies)
end

function Base.getindex(s::OffdiagonalsBoseMom1D{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i, s.singlies, s.doublies)
    return (new_address, matrix_element)
end

Base.size(s::OffdiagonalsBoseMom1D) = (s.length,)

###
### momentum
###
struct MomentumMom1D{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    ham::H
end
LOStructure(::Type{MomentumMom1D{H,T}}) where {H,T <: Real} = IsHermitian()
num_offdiagonals(ham::MomentumMom1D, add) = 0
diagonal_element(mom::MomentumMom1D, add) = mod1(onr(add)⋅ks(mom.ham) + π, 2π) - π # fold into (-π, π]

momentum(ham::HubbardMom1D) = MomentumMom1D(ham)
