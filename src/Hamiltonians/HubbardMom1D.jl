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
    add::Union{BoseFS,FermiFS2C};
    u=1.0, t=1.0, dispersion = hubbard_dispersion,
)
    M = num_modes(add)
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
@inline function num_offdiagonals(ham::HubbardMom1D, add::FermiFS2C{N1,N2}) where {N1,N2}
    M = num_modes(ham)
    return N1 * N2 * (M - 1)
end

"""
    momentum_transfer_diagonal(H, map::OccupiedModeMap)

Compute diagonal interaction energy term.

# Example

```jldoctest
julia> a = BoseFS{6,5}((1,2,3,0,0))
BoseFS{6,5}((1, 2, 3, 0, 0))

julia> H = HubbardMom1D(a);

julia> Hamiltonians.momentum_transfer_diagonal(H, OccupiedModeMap(a))
5.2
```
"""
@inline function momentum_transfer_diagonal(
    h::HubbardMom1D{<:Any,M,<:BoseFS}, map
) where {M}
    return h.u / 2M * momentum_transfer_diagonal(map)
end
@inline function momentum_transfer_diagonal(
    h::HubbardMom1D{<:Any,M,<:FermiFS2C}, map_a, map_b
) where {M}
    return h.u / 2M * momentum_transfer_diagonal(map_a, map_b)
end

@inline function diagonal_element(h::HubbardMom1D, add::BoseFS)
    map = OccupiedModeMap(add)
    return dot(h.kes, map) + momentum_transfer_diagonal(h, map)
end
@inline function diagonal_element(h::HubbardMom1D, add::FermiFS2C)
    map_a = OccupiedModeMap(add.components[1])
    map_b = OccupiedModeMap(add.components[2])
    return dot(h.kes, map_a) + dot(h.kes, map_b) +
        momentum_transfer_diagonal(h, map_a, map_b)
end

@inline function get_offdiagonal(
    ham::HubbardMom1D{<:Any,M,A}, add::A, chosen, map=OccupiedModeMap(add)
) where {M,A<:BoseFS}
    add, onproduct = momentum_transfer_excitation(add, chosen, map)
    return add, ham.u/(2*M)*onproduct
end
@inline function get_offdiagonal(
    ham::HubbardMom1D{<:Any,M,A}, add::A, chosen,
    map_a=OccupiedModeMap(add.components[1]), map_b=OccupiedModeMap(add.components[2])
) where {M,A<:FermiFS2C}
    add_a, add_b = add.components
    new_add_a, new_add_b, onproduct = momentum_transfer_excitation(
        add_a, add_b, chosen, map_a, map_b
    )
    return CompositeFS(new_add_a, new_add_b), ham.u/M * onproduct
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
    A<:BoseFS,T,H<:AbstractHamiltonian{T},O<:OccupiedModeMap
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    map::O
end

function offdiagonals(h::HubbardMom1D, a::BoseFS)
    map = OccupiedModeMap(a)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)
    num = num_offdiagonals(h, a, singlies, doublies)
    return OffdiagonalsBoseMom1D(h, a, num, map)
end

function Base.getindex(s::OffdiagonalsBoseMom1D{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i, s.map)
    return (new_address, matrix_element)
end

Base.size(s::OffdiagonalsBoseMom1D) = (s.length,)

struct OffdiagonalsFermiMom1D2C{
    F<:FermiFS2C,T,H<:AbstractHamiltonian{T},O1,O2
} <: AbstractOffdiagonals{F,T}
    hamiltonian::H
    address::F
    length::Int
    map_a::O1
    map_b::O2
end

function offdiagonals(h::HubbardMom1D, f::FermiFS2C)
    comp_a, comp_b = f.components
    map_a = OccupiedModeMap(comp_a)
    map_b = OccupiedModeMap(comp_b)
    num = num_offdiagonals(h, f)
    return OffdiagonalsFermiMom1D2C(h, f, num, map_a, map_b)
end

Base.size(s::OffdiagonalsFermiMom1D2C) = (s.length,)

function Base.getindex(s::OffdiagonalsFermiMom1D2C{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        i ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(
        s.hamiltonian, s.address, i, s.map_a, s.map_b
    )
    return (new_address, matrix_element)
end

###
### momentum
###
struct MomentumMom1D{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    ham::H
end
LOStructure(::Type{MomentumMom1D{H,T}}) where {H,T <: Real} = IsDiagonal()
num_offdiagonals(ham::MomentumMom1D, add) = 0
diagonal_element(mom::MomentumMom1D, add) = mod1(onr(add)⋅ks(mom.ham) + π, 2π) - π
# fold into (-π, π]
starting_address(mom::MomentumMom1D) = starting_address(mom.ham)

momentum(ham::HubbardMom1D) = MomentumMom1D(ham)
