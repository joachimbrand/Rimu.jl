"""
    momentum_space_harmonic_potential(M::Integer, v_ho::Real) -> w

Set up a harmonic potential for use with momentum space Hamiltonians:
```math
\\begin{aligned}
w(k) & =  \\mathrm{DFT}[V_{ext}]_{k}  ,\\\\
V_\\mathrm{ext}(x) &= v_\\mathrm{ho} \\,x^2 ,
\\end{aligned}
```
where
``\\mathrm{DFT}[…]_k`` is a discrete Fourier transform performed by `fft()[k%M + 1]`.
"""
function momentum_space_harmonic_potential(M::Integer, v::Real)
    v = float(v)
    # Set up potential like in Real1DEP
    is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
    js = shift_lattice(is) # shifted such that js[1] = 0
    real_potential = [v*j^2 for j in js]
    mom_potential = fft(real_potential)
    # This should never fail for a harmonic oscillator, but it's best to check just in case.
    for x in mom_potential
        @assert iszero(real(x)) || abs(imag(x) / real(x)) < sqrt(eps(v))
    end
    # Make sure it's completely symmetric. It should be, but round-off errors can sometimes
    # make it non-symmetric.
    for i in 1:M÷2
        mom_potential[M - i + 1] = mom_potential[i + 1]
    end
    return SVector{M}(real.(mom_potential))
end

"""
    HubbardMom1DEP(address; u=1.0, t=1.0, v_ho=1.0, dispersion=hubbard_dispersion)

Implements a one-dimensional Bose Hubbard chain in momentum space with harmonic external
potential.

```math
Ĥ = \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}
            + V̂_\\mathrm{ho} ,
```
where
```math
\\begin{aligned}
V̂_\\mathrm{ho} & = \\sum_{p,q}  \\mathrm{DFT}[V_{ext}]_{p-q} \\,
                    a^†_{p} a_q ,\\\\
V_\\mathrm{ext}(x) &= v_\\mathrm{ho} \\,x^2 ,
\\end{aligned}
```
is an external harmonic potential in momentum space,
``\\mathrm{DFT}[…]_k`` is a discrete Fourier transform performed by `fft()[k%M + 1]`, and
`M == num_modes(address)`.

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.
* `dispersion`: defines ``ϵ_k =``` t*dispersion(k)`
    - [`hubbard_dispersion`](@ref): ``ϵ_k = -2t \\cos(k)``
    - [`continuum_dispersion`](@ref): ``ϵ_k = tk^2``
* `v_ho`: strength of the external harmonic oscillator potential ``v_\\mathrm{ho}``.

See also [`HubbardMom1D`](@ref), [`HubbardReal1DEP`](@ref),
[`Transcorrelated1D`](@ref), [`Hamiltonians`](@ref).
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
    kes = SVector{M}(T .* dispersion.(kr))

    potential = momentum_space_harmonic_potential(M, V)

    return HubbardMom1DEP{typeof(U),M,typeof(add),U,T}(add, ks, kes, potential)
end

function Base.show(io::IO, h::HubbardMom1DEP)
    print(io, "HubbardMom1DEP($(h.add); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardMom1DEP)
    return h.add
end
function get_offdiagonal(h::HubbardMom1DEP{<:Any,<:Any,F}, add::F, i) where {F}
    return offdiagonals(h, add)[i]
end
function num_offdiagonals(h::HubbardMom1DEP{<:Any,<:Any,F}, add::F) where {F}
    return length(offdiagonals(h, add))
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
    return dot(h.kes, map) + momentum_transfer_diagonal(h, map) +
        momentum_external_potential_diagonal(h.ep, add, map)
end
@inline function diagonal_element(h::HubbardMom1DEP, add::FermiFS2C)
    c1, c2 = add.components
    map_a = OccupiedModeMap(c1)
    map_b = OccupiedModeMap(c2)
    return dot(h.kes, map_a) + dot(h.kes, map_b) +
        momentum_transfer_diagonal(h, map_a, map_b) +
        momentum_external_potential_diagonal(h.ep, c1, map_a) +
        momentum_external_potential_diagonal(h.ep, c2, map_b)
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

struct OffdiagonalsFermiMom1D2CEP{
    A<:FermiFS2C,T,H<:AbstractHamiltonian{T},O1,O2
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    num_mom::Int
    num_ep_a::Int
    num_ep_b::Int
    map_a::O1
    map_b::O2
end

function offdiagonals(h::HubbardMom1DEP, a::FermiFS2C)
    M = num_modes(a)
    comp_a, comp_b = a.components
    N1 = num_particles(comp_a)
    N2 = num_particles(comp_b)
    map_a = OccupiedModeMap(comp_a)
    map_b = OccupiedModeMap(comp_b)
    num_mom = N1 * N2 * (M - 1)
    num_ep_a = N1 * (M - 1)
    num_ep_b = N2 * (M - 1)

    return OffdiagonalsFermiMom1D2CEP(h, a, num_mom, num_ep_a, num_ep_b, map_a, map_b)
end

Base.size(s::OffdiagonalsFermiMom1D2CEP) = (s.num_mom + s.num_ep_a + s.num_ep_b,)

function Base.getindex(s::OffdiagonalsFermiMom1D2CEP{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ length(s) || throw(BoundsError(s, i))
    end
    c1, c2 = s.address.components
    M = num_modes(s.address)
    if i ≤ s.num_mom
        new_c1, new_c2, onproduct = momentum_transfer_excitation(c1, c2, i, s.map_a, s.map_b)
        new_address = CompositeFS(new_c1, new_c2)
        matrix_element = s.hamiltonian.u/M * onproduct
    elseif i ≤ s.num_mom + s.num_ep_a
        i -= s.num_mom
        new_c1, matrix_element = momentum_external_potential_excitation(
            s.hamiltonian.ep, c1, i, s.map_a
        )
        new_address = CompositeFS(new_c1, c2)
    else
        i -= s.num_mom + s.num_ep_a
        new_c2, matrix_element = momentum_external_potential_excitation(
            s.hamiltonian.ep, c2, i, s.map_b
        )
        new_address = CompositeFS(c1, new_c2)
    end
    return (new_address, matrix_element)
end
