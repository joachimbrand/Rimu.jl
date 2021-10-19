# TODO: disabling three-body terms
"""
    Transcorrelated1D(address; t=1.0, u=1.0, v_ho=1.0, cutoff=1)

Implements a transcorrelated Hamiltonian for two component fermionic addresses with optional
harmonic potential:

```math
\\tilde{H} = J\\sum_{kσ}k^2 n_{k,σ} +
              \\sum_{pqkσσ'} T_{pqk} a^†_{p-k,σ} a^†_{q+k,σ'} a_{q,σ'} a_{p,σ} +
              \\sum_{pqskk'σσ'} Q_{kk'}a^†_{p-k,σ} a^†_{q+k,σ} a^†_{s+k-k',σ'}
                                       a_{s,σ'} a_{q,σ} a_{p,σ},
```

where

```math
\\tilde{u}(k) = {-2/k^2 \\mathrm{\\ if\\ } |k| ≥ k_c; 0 \\mathrm{\\ otherwise\\ }}

W(k) = \\sum_{k′} (k - k′)k′ \\tilde{u}(k′)\\tilde{u}(k - k′),

T_{pqk} = \\frac{v}{M} + \\frac{2t}{M}(k^2\\tilde{u}(k) - (p - q)k\\tilde{u}(k) +
\\frac{W(k)}{M}),

Q_{kl} = -\\frac{t}{M^2}kl \\tilde{u}(k)\\tilde{u}(l).
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `v`: the interaction parameter. Default: 1
* `t`: the hopping strength. Default: 1
* `v_ho`: strength of the external harmonic oscillator potential ``ϵ_i = v_{ho}
  i^2``. Default: 0
* `cutoff`: a high `cutoff` reduces the number of three-body terms in the
  Hamiltonian. Default: 1
* `three_body_term`: Use the three body term of the Hamiltonian. If set to false, generating
  three body excitations is skipped. Default: true

# See also

* [`HubbardMom1D`](@ref)
* [`HubbardMom1DEP`](@ref)
"""
struct Transcorrelated1D{
    M,F<:CompositeFS{2,<:Any,M,<:Tuple{FermiFS,FermiFS}}, # TODO: relax this to allow bosons
    P<:Union{Nothing,SVector{M,Float64}}
} <: AbstractHamiltonian{Float64}
    address::F
    cutoff::Int
    v::Float64
    t::Float64
    v_ho::Float64
    ks::SVector{M,Float64}
    kes::SVector{M,Float64}
    ws::SVector{M,Float64}
    us::SVector{M,Float64}
    potential::P
    three_body_term::Bool
end

function Transcorrelated1D(address; t=1.0, v=1.0, v_ho=0.0, cutoff=1, three_body_term=true)
    M = num_modes(address)
    cutoff < 1 && error("`cutoff` must be a positive integer")
    ks = SVector{M}(i_to_k.(1:M, M))
    kes = t .* ks.^2
    ws = SVector{M}(w_function.(0:M-1, cutoff, M, v, t))
    us = SVector{M}(correlation_factor.(1:M, cutoff, M))
    if iszero(v_ho)
        potential = nothing
    else
        potential = momentum_space_harmonic_potential(M, v_ho)
    end

    return Transcorrelated1D{M,typeof(address),typeof(potential)}(
        address, cutoff, float(v), float(t), float(v_ho), ks, kes, ws, us, potential,
        three_body_term
    )
end

function Base.show(io::IO, h::Transcorrelated1D)
    print(io, "Transcorrelated1D(", starting_address(h), ", t=$(h.t), v=$(h.v)")
    if !iszero(h.v_ho)
        print(io, ", v_ho=$(h.v_ho))")
    else
        print(io, ")")
    end
end

LOStructure(::Type{<:Transcorrelated1D}) = AdjointUnknown()

function get_offdiagonal(h::Transcorrelated1D{<:Any,F}, add::F, i) where {F}
    return offdiagonals(h, add)[i]
end
function num_offdiagonals(h::Transcorrelated1D{<:Any,F}, add::F) where {F}
    return length(offdiagonals(h, add))
end

# Note: n starts at 0 in the middle of the address, and can be positive or negative
@inline i_to_k(i, M) = i_to_n(i, M) * 2π/M
@inline i_to_n(i, M) = i - M ÷ 2 - isodd(M)
@inline n_to_i(n, M) = n + M ÷ 2 + isodd(M)
@inline n_to_k(n, M) = i_to_k(n_to_i(n, M), M)

"""
    correlation_factor(n, cutoff)

Compute the (dimensionless) correlation factor multiplied by ``k``, ``k\\tilde{u}(k)``.

```math
k \\tilde{u}(k) = {-2/k \\mathrm{\\ if\\ } |k| ≥ k_c; 0 \\mathrm{\\ otherwise\\ }}
```

where ``k = π + 2πn/M``.
"""
function correlation_factor(n, cutoff, M)
    if abs(n) ≥ cutoff
        -1/2n_to_k(n, M)
    else
        0.0
    end
end
function correlation_factor(h::Transcorrelated1D{M}, n) where {M}
    absn = abs(n)
    if absn > 0
        return sign(n) * h.us[absn]
    else
        return 0.0
    end
end

"""
    w_function(i::Integer, nc::Integer, M, v, t)
    w_function(h::Transcorrelated1D, i::Integer)

Compute the (dimensionless) function ``\tilde{W}(k) = t^2/u^2 W(k)``.

```math
W(k) = \\sum_{k′} (k - k′)k′ \\tilde{u}(k′)\\tilde{u}(k - k′)
```
"""
function w_function(n, nc, M, v, t)
    prefactor = -1 / (8π^2)
    n = abs(n)

    if n == 0
        x = π^2/6 - sum(1/(np^2) for np in 1:nc-1; init=0.0)
    elseif 2nc > n > 0
        x = 1/n * sum(1/np for np in nc:n+nc-1)
    elseif n ≥ nc
        x = 1/n * sum(1/np for np in nc:n+nc-1) -
            1/2 * sum(1/(np * (n - np)) for np in nc:n-nc)
    end

    return prefactor * x
end
w_function(h::Transcorrelated1D, i) = h.ws[abs(i) + 1]

"""
    t_function(h::Transcorrelated1D, p, q, k)

Compute

```math
T_{pqk} = \\frac{v}{M} + \\frac{2t}{M}(k^2\\tilde{u}(k) - (p - q)k\\tilde{u}(k) +
\\frac{W(k)}{M})
```

where ``k\\tilde{u}(k)`` is the [`correlation_factor`](@ref) and ``W(k)`` is the
[`w_function`](@ref).
"""
function t_function(h::Transcorrelated1D{M}, p, q, k) where {M}
    @unpack t, v = h
    k_pi = n_to_k(k, M)
    pmq_pi = n_to_k(p - q, M)
    cor_k = correlation_factor(h, k)
    return v/M + 2v/M * (cor_k * k_pi - cor_k * pmq_pi) +
        2v^2/t * w_function(h, k)
end

"""
    q_function(h::Transcorrelated1D, k, l)

Compute

```math
Q_{kl} = -\\frac{t}{M^2}kl \\tilde{u}(k)\\tilde{u}(l),
```

where ``k\\tilde{u}(k)`` is the [`correlation_factor`](@ref).
"""
function q_function(h::Transcorrelated1D{M}, k, l) where {M}
    @unpack t, v = h
    cor_k = correlation_factor(h, k)
    cor_l = correlation_factor(h, l)

    return -v^2/(t * M^2) * cor_k * cor_l
end

starting_address(ham::Transcorrelated1D) = ham.address

function momentum_transfer_diagonal(h::Transcorrelated1D{M}, map1, map2) where {M}
    @unpack v, t = h
    return momentum_transfer_diagonal(map1, map2) * (v/M + 2v^2/t * w_function(h, 0)) / 2
end

"""
    transcorrelated_diagonal(h::Transcorrelated1D, map1, map2)

The diagonal part of [`transcorrelated_three_body_excitation`](@ref).
"""
function transcorrelated_diagonal(h::Transcorrelated1D{M}, map1, map2) where {M}
    value = 0.0
    for p in 1:length(map1)
        for q in 1:p-1
            k = map1[p].mode - map1[q].mode
            qkk = q_function(h, -k, k)
            value += 2 * qkk * length(map2)
        end
    end
    return value
end

function diagonal_element(h::Transcorrelated1D{<:Any,F}, add::F) where {F}
    c1, c2 = add.components
    map1 = OccupiedModeMap(c1)
    map2 = OccupiedModeMap(c2)

    value = kinetic_energy(h.kes, map1) +
        kinetic_energy(h.kes, map2) +
        momentum_transfer_diagonal(h, map1, map2)
    if h.three_body_term
        value += transcorrelated_diagonal(h, map1, map2) +
            transcorrelated_diagonal(h, map2, map1)
    end
    if !isnothing(h.potential)
        value += momentum_external_potential_diagonal(h.potential, c1, map1) +
            momentum_external_potential_diagonal(h.potential, c2, map2)
    end

    return value
end

struct Transcorrelated1DOffdiagonals{H,A,O1,O2}<:AbstractOffdiagonals{A,Float64}
    hamiltonian::H
    address::A
    map1::O1
    map2::O2
    length::Int
end

function offdiagonals(h::Transcorrelated1D{M,F}, add::F) where {M,F}
    offdiags = Tuple{F,Float64}[]
    c1, c2 = add.components
    map1 = OccupiedModeMap(c1)
    map2 = OccupiedModeMap(c2)
    N1 = length(map1)
    N2 = length(map2)
    n_mom = N1 * N2 * (M - 1)

    three_body_term = h.three_body_term
    n_trans1 = three_body_term ? N1 * (N1 - 1) * N2 * M * M : 0
    n_trans2 = three_body_term ? N2 * (N2 - 1) * N1 * M * M : 0

    if !isnothing(h.potential)
        n_pot1 = N1 * (M - 1)
        n_pot2 = N2 * (M - 1)
    else
        n_pot1 = 0
        n_pot2 = 0
    end
    len = n_mom + n_trans1 + n_trans2 + n_pot1 + n_pot2
    return Transcorrelated1DOffdiagonals(h, add, map1, map2, len)
end

Base.size(od::Transcorrelated1DOffdiagonals) = (od.length,)

function Base.getindex(od::Transcorrelated1DOffdiagonals, i)
    @unpack address, map1, map2 = od
    c1, c2 = address.components
    h = od.hamiltonian
    N1 = length(map1)
    N2 = length(map2)
    M = num_modes(c1)

    n_mom = N1 * N2 * (M - 1)
    three_body_term = od.hamiltonian.three_body_term
    n_trans1 = three_body_term ? N1 * (N1 - 1) * N2 * M * M : 0
    n_trans2 = three_body_term ? N2 * (N2 - 1) * N1 * M * M : 0

    # This should be efficient as it depends on the type of the potential
    if !isnothing(od.hamiltonian.potential)
        n_pot1 = N1 * (M - 1)
        n_pot2 = N2 * (M - 1)
    else
        n_pot1 = 0
        n_pot2 = 0
    end

    # Fallback on zero values
    new_c = CompositeFS(c1, c2)

    if i ≤ n_mom
        # Momentum transfer
        new_c1, new_c2, value, p, q, k = momentum_transfer_excitation(
            c1, c2, i, map1, map2; fold=false
        )
        if !iszero(value)
            @assert new_c1 ≠ c1 || new_c2 ≠ c2
            value *= t_function(h, p, q, k)
            new_c = CompositeFS(new_c1, new_c2)
        end
    elseif i ≤ n_mom + n_trans1
        # Transcorrelated excitation from first to second component
        i -= n_mom

        new_c1, new_c2, value, k, l = transcorrelated_three_body_excitation(
            c1, c2, i, map1, map2
        )
        value *= q_function(h, k, l)
        if !iszero(value)
            @assert new_c1 ≠ c1 || new_c2 ≠ c2
            new_c = CompositeFS(new_c1, new_c2)
        end
    elseif i ≤ n_mom + n_trans1 + n_trans2
        # Transcorrelated excitation from second to first component
        i -= n_mom + n_trans1

        new_c2, new_c1, value, k, l = transcorrelated_three_body_excitation(
            c2, c1, i, map2, map1
        )
        value *= q_function(h, k, l)
        if !iszero(value)
            @assert new_c1 ≠ c1 || new_c2 ≠ c2
            new_c = CompositeFS(new_c1, new_c2)
        end
    elseif i ≤ n_mom + n_trans1 + n_trans2 + n_pot1
        # Potential acting on first component
        i -= n_mom + n_trans1 + n_trans2

        new_c1, value = momentum_external_potential_excitation(
            od.hamiltonian.potential, c1, i, map1
        )
        if !iszero(value)
            new_c = CompositeFS(new_c1, c2)
        end
    elseif i ≤ n_mom + n_trans1 + n_trans2 + n_pot1 + n_pot2
        # Potential acting on second component
        i -= n_mom + n_trans1 + n_trans2 + n_pot1

        new_c2, value = momentum_external_potential_excitation(
            od.hamiltonian.potential, c2, i, map2
        )
        if !iszero(value)
            new_c = CompositeFS(c1, new_c2)
        end
    else
        throw(BoundsError(od, i))
    end
    return new_c, value
end
