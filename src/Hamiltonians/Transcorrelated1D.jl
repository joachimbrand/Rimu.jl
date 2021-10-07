struct Transcorrelated1D{
    M,F<:CompositeFS{2,<:Any,M,<:Tuple{FermiFS,FermiFS}} # TODO: relax this to allow bosons
} <: AbstractHamiltonian{Float64}
    address::F
    cutoff::Int
    v::Float64
    t::Float64
    ks::SVector{M,Float64}
    ws::SVector{M,Float64}
    us::SVector{M,Float64}
end

function Transcorrelated1D(address; t=1.0, v=1.0, cutoff=1)
    M = num_modes(address)
    cutoff < 1 && error("`cutoff` must be a positive integer")
    ks = SVector{M}(i_to_k.(1:M, M))
    ws = SVector{M}(w_function.(0:M-1, cutoff, M, v, t))
    us = SVector{M}(correlation_factor.(1:M, cutoff, M))

    return Transcorrelated1D{M,typeof(address)}(address, cutoff, v, t, ks, ws, us)
end

function Base.show(io::IO, h::Transcorrelated1D)
    print(io, "Transcorrelated1D(", starting_address(h), ", t=$(h.t), v=$(h.v))")
end

# Note: n starts at 0 in the middle of the address, and can be positive or negative
@inline i_to_k(i, M) = i_to_n(i, M) * 2π/M
@inline i_to_n(i, M) = i - M ÷ 2 - isodd(M)
@inline n_to_i(n, M) = n + M ÷ 2 + isodd(M)
@inline n_to_k(n, M) = i_to_k(n_to_i(n, M), M)
@inline cutoff_to_kc(cutoff, M) = cutoff * 2π/M

"""
    correlation_factor(n, cutoff)

Compute the (dimensionless) correlation factor multiplied by ``k``, ``k\\tilde{u}(k)``.

```math
k \\tilde{u}(k) = {-2/k \\mathrm{\\ if\\ } |k| ≥ k_c; 0 \\mathrm{\\ otherwise\\ }}
```

where ``k = π + 2πn/M``.
"""
function correlation_factor(n, cutoff, M)
    return ifelse(abs(n) ≥ cutoff, -1/2n_to_k(n, M), 0.0)
end
function correlation_factor(h::Transcorrelated1D{M}, n) where {M}
    return correlation_factor(n, h.cutoff, M)
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
```math
T_{pqk}
```
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
Compute the

```math
Q_{kl}
```
"""
function q_function(h::Transcorrelated1D{M}, k, l) where {M}
    @unpack t, v = h
    cor_k = correlation_factor(h, k)
    cor_l = correlation_factor(h, l)

    return -v^2/(t * M^2) * cor_k * cor_l
end

starting_address(ham::Transcorrelated1D) = ham.address

"""
First term,

```math
t \\sum_{kσ} k^2 n_{k,σ}
```
"""
function kinetic_energy(h::Transcorrelated1D{M}, onr1, onr2) where {M}
    return h.t * sum(h.ks[i]^2 * (onr1[i] + onr2[i]) for i in 1:M)
end

"""
Diagonal contribution from second term (where ``k = 0``),

```math
\\sum_{pqσσ'} T_{pq0} n_{p,σ} n_{q,σ'}
```

and third term where ``k = k' = p - q`` (two ways).

```math
\\sum_{pqsσσ'} Q_{kk} n_{p,σ} n_{q,σ} n_{s,σ'}
```
"""
# Fermion version
function interaction_energy_diagonal(h::Transcorrelated1D{M}, onr1, onr2) where {M}
    @unpack v, t = h
    N2 = sum(onr2)
    onproduct = 0
    for p in 1:M
        onproduct += N2 * onr1[p]
    end
    return onproduct * (v/M + 2v^2/t * w_function(h, 0))
end
function transcorrelated_diagonal(h::Transcorrelated1D{M}, onr1, onr2) where {M}
    value = 0.0
    for p in 1:M
        onr1[p] == 0 && continue
        for q in 1:p-1
            onr1[q] == 0 && continue
            k = p - q
            qkk = q_function(h, -k, k)
            @assert onr1[p] == onr1[q] == 1
            for s in 1:M
                # factor of 2 because we skipped half of the loop
                value += 2 * qkk * onr2[s]
            end
        end
    end
    return value
end

function diagonal_element(h::Transcorrelated1D{<:Any,F}, add::F) where {F}
    onr1, onr2 = onr(add)
    return kinetic_energy(h, onr1, onr2) + interaction_energy_diagonal(h, onr1, onr2) +
        transcorrelated_diagonal(h, onr1, onr2) + transcorrelated_diagonal(h, onr2, onr1)
end

function offdiagonals(h::Transcorrelated1D{M,F}, add::F) where {M,F}
    offdiags = Tuple{F,Float64}[]
    c1, c2 = add.components
    map1 = OccupiedModeMap(c1)
    map2 = OccupiedModeMap(c2)
    N1 = length(map1)
    N2 = length(map2)

    # Second term
    for i in 1:N1*N2*(M - 1)
        new_c1, new_c2, value, p, q, k = momentum_transfer_excitation(
            c1, c2, i, map1, map2; fold=false
        )
        iszero(value) && continue
        @assert new_c1 ≠ c1 || new_c2 ≠ c2
        value *= t_function(h, p, q, k)
        new_c = CompositeFS(new_c1, new_c2)
        push!(offdiags, (new_c, value))
    end

    # Third term
    for i in 1:N1 * N1 * N2 * M * M
        new_c1, new_c2, value, k, l = transcorrelated_three_body_excitation(
            c1, c2, i, map1, map2
        )
        value *= q_function(h, k, l)
        iszero(value) && continue
        @assert new_c1 ≠ c1 || new_c2 ≠ c2
        new_c = CompositeFS(new_c1, new_c2)
        push!(offdiags, (new_c, value))
    end
    for i in 1:N2 * N2 * N1 * M * M
        new_c2, new_c1, value, k, l = transcorrelated_three_body_excitation(
            c2, c1, i, map2, map1
        )
        value *= q_function(h, k, l)
        iszero(value) && continue
        @assert new_c1 ≠ c1 || new_c2 ≠ c2
        new_c = CompositeFS(new_c1, new_c2)
        push!(offdiags, (new_c, value))
    end

    return offdiags
end
