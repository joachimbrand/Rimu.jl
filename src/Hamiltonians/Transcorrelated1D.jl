# TODO: disabling three-body terms
"""
    Transcorrelated1D(address; t=1.0, v=1.0, v_ho=0.0, cutoff=1, three_body_term=true)

Implements a transcorrelated Hamiltonian for contact interactions in one dimensional
momentum space from [Jeszenski *et al.* (2018)](http://arxiv.org/abs/1806.11268).
Currently limited to two component fermionic addresses.

```math
\\begin{aligned}

\\tilde{H} &= t \\sum_{kσ}k^2 n_{k,σ} \\\\
    &\\quad + \\sum_{pqkσσ'} T_{pqk} a^†_{p-k,σ} a^†_{q+k,σ'} a_{q,σ'} a_{p,σ} \\\\
    &\\quad + \\sum_{pqskk'σσ'} Q_{kk'}a^†_{p-k,σ} a^†_{q+k,σ} a^†_{s+k-k',σ'}
                                       a_{s,σ'} a_{q,σ} a_{p,σ} \\\\
    &\\quad + V̂_\\mathrm{ho}
\\end{aligned}
```

where

```math
\\begin{aligned}
\\tilde{u}(k) &= \\begin{cases} -\\frac{2}{k^2} &\\mathrm{if\\ } |k| ≥ k_c\\\\
0 & \\mathrm{otherwise}
\\end{cases}
\\\\

T_{pqk} &= \\frac{v}{M} + \\frac{2v}{M}\\left[k^2\\tilde{u}(k)
          - (p - q)k\\tilde{u}(k)\\right] + \\frac{2v^2}{t}W(k)\\\\
W(k) &= \\frac{1}{M^2}\\sum_{q} (k - q)q\\, \\tilde{u}(q)\\,\\tilde{u}(k - q) \\\\
Q_{kl} &= -\\frac{v^2}{t M^2}k \\tilde{u}(k)\\,l\\tilde{u}(l),
\\end{aligned}
```

# Arguments

* `address`: The starting address, defines number of particles and sites.
* `v`: The interaction parameter.
* `t`: The kinetic energy prefactor.
* `v_ho`: Strength of the external harmonic oscillator potential ``V̂_\\mathrm{ho}``.
  See [`HubbardMom1DEP`](@ref).
* `cutoff` controls ``k_c`` in equations above. Note: skipping generating
  off-diagonal elements below the cutoff is not implemented - zero-valued elements
  are returned instead.
* `three_body_term`: If set to false, generating three body excitations is skipped.
  Note: when disabling three body terms, cutoff should be set to a higher value for
  best results.

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
    ks::SVector{M,Float64} # wave numbers
    kes::SVector{M,Float64} # single-particle dispersion
    ws::SVector{M,Float64} # pre-computed W(k)
    us::SVector{M,Float64} # correlation factor
    potential::P # external potential
    three_body_term::Bool
end

function Transcorrelated1D(address; t=1.0, v=1.0, v_ho=0.0, cutoff=1, three_body_term=true)
    M = num_modes(address)
    cutoff < 1 && throw(ArgumentError("`cutoff` must be a positive integer"))
    ks = SVector{M}(i_to_k.(1:M, M))
    kes = t .* ks.^2
    ws = SVector{M}(w_function.(0:M-1, cutoff))
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

dimension(::Transcorrelated1D, address) = number_conserving_dimension(address)

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
    correlation_factor(n, cutoff, M)

Compute the (dimensionless) correlation factor multiplied by ``k``:

```math
\\begin{aligned}
k \\tilde{u}(k) &= \\begin{cases} -\\frac{2}{k} &\\mathrm{if\\ } |k| ≥ k_c \\\\
0 & \\mathrm{otherwise}
\\end{cases}
\\end{aligned}
```

where ``k = π + 2πn/M`` and `k_c = π + 2π/M * cutoff`.

See also [`Transcorrelated1D`](@ref).
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
    w_function(n::Integer, nc::Integer)
    w_function(h::Transcorrelated1D, n::Integer)

Compute the (dimensionless) function
```math
W(k) = \\frac{1}{M^2}\\sum_{q} (k - q)q\\, \\tilde{u}(q)\\,\\tilde{u}(k - q) .
```
where ``k = π + 2π\\mathtt{n}/M``,  ``k_c = π + 2π\\mathtt{nc}/M``,
and ``k\\tilde{u}(k)`` is the [`correlation_factor`](@ref).

See also [`Transcorrelated1D`](@ref).
"""
function w_function(n::Integer, nc::Integer)
    prefactor = -1 / (8π^2)
    n = abs(n)

    if n == 0
        x = π^2/6 - sum(1/(np^2) for np in 1:nc-1; init=0.0)
    elseif 2nc > n > 0
        x = 1/n * sum(1/np for np in nc:n+nc-1)
    else
        x = 1/n * sum(1/np for np in nc:n+nc-1) -
            1/2 * sum(1/(np * (n - np)) for np in nc:n-nc)
    end

    return prefactor * x
end
w_function(h::Transcorrelated1D, n::Integer) = h.ws[abs(n) + 1]

"""
    t_function(h::Transcorrelated1D, p, q, k)

Compute

```math
T_{pqk} = \\frac{v}{M} + \\frac{2v}{M}\\left[k^2\\tilde{u}(k)
          - (p - q)k\\tilde{u}(k)\\right] + \\frac{2v^2}{t}W(k)
```

where ``k\\tilde{u}(k)`` is the [`correlation_factor`](@ref) and ``W(k)`` is the
[`w_function`](@ref).

See also [`Transcorrelated1D`](@ref).
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
Q_{kl} = -\\frac{v^2}{t M^2}k \\tilde{u}(k)\\,l\\tilde{u}(l),
```

where ``k\\tilde{u}(k)`` is the [`correlation_factor`](@ref).

See also [`Transcorrelated1D`](@ref).
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

    value = dot(h.kes, map1) + dot(h.kes, map2) +
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
    C = typeof(od.address)
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
    new_c = C(c1, c2)

    if i ≤ n_mom
        # Momentum transfer
        new_c1, new_c2, value, p, q, k = momentum_transfer_excitation(
            c1, c2, i, map1, map2; fold=false
        )
        if !iszero(value)
            @assert new_c1 ≠ c1 || new_c2 ≠ c2
            value *= t_function(h, p, q, k)
            new_c = C(new_c1, new_c2)
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
            new_c = C(new_c1, new_c2)
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
            new_c = C(new_c1, new_c2)
        end
    elseif i ≤ n_mom + n_trans1 + n_trans2 + n_pot1
        # Potential acting on first component
        i -= n_mom + n_trans1 + n_trans2

        new_c1, value = momentum_external_potential_excitation(
            od.hamiltonian.potential, c1, i, map1
        )
        if !iszero(value)
            new_c = C(new_c1, c2)
        end
    elseif i ≤ n_mom + n_trans1 + n_trans2 + n_pot1 + n_pot2
        # Potential acting on second component
        i -= n_mom + n_trans1 + n_trans2 + n_pot1

        new_c2, value = momentum_external_potential_excitation(
            od.hamiltonian.potential, c2, i, map2
        )
        if !iszero(value)
            new_c = C(c1, new_c2)
        end
    else
        throw(BoundsError(od, i))
    end
    return new_c, value
end
