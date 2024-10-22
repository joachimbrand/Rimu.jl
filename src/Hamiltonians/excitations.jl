using Rimu
using Rimu.Hamiltonians: hopnextneighbour

const BoseOccupiedModeMap{N} = OccupiedModeMap{N,BitStringAddresses.BoseFSIndex}
const FermiOccupiedModeMap{N} = OccupiedModeMap{N,BitStringAddresses.FermiFSIndex}

"""
    momentum_transfer_excitation(add, chosen, map; fold=true) -> nadd, α, p, q, -k
    momentum_transfer_excitation(add_a, add_b, chosen, map_a, map_b; fold=true)

Apply the momentum transfer operator to Fock address (or pair of addresses) `add` (or
`add_a`, `add_b`):

```math
a^†_{p + k} a^†_{q - k} a_q a_p |\\mathtt{add}⟩
```

The `fold` argument controls whether the terms `p + k` and `q - k` are done modulo M. If
not, zero is returned when either of those terms is less than 1 or larger than M.
It is expected that `map == OccupiedModeMap(add)`.

Return the new address(es), the value, modes `p` and `q`, and the momentum change `-k`.

See [`excitation`](@ref), [`OccupiedModeMap`](@ref).
"""
@inline function momentum_transfer_excitation(
    add::SingleComponentFockAddress, chosen, map::OccupiedModeMap; fold=true
)
    M = num_modes(add)
    singlies = length(map) # number of at least singly occupied modes

    double = chosen - singlies * (singlies - 1) * (M - 2)

    if double > 0
        # Both moves from the same mode.
        double, mom_change = fldmod1(double, M - 1)
        idx = first(map) # placeholder
        for i in map
            double -= i.occnum ≥ 2
            if double == 0
                idx = i
                break
            end
        end
        src_indices = (idx, idx)
    else
        # Moves from different modes.
        pair, mom_change = fldmod1(chosen, M - 2)
        fst, snd = fldmod1(pair, singlies - 1) # where the holes are to be made
        if snd < fst # put them in ascending order
            f_hole = snd
            s_hole = fst
        else
            f_hole = fst
            s_hole = snd + 1 # as we are counting through all singlies
        end
        src_indices = (map[f_hole], map[s_hole])
        f_mode = src_indices[1].mode
        s_mode = src_indices[2].mode
        if mom_change ≥ s_mode - f_mode
            mom_change += 1 # to avoid putting particles back into the holes
        end
    end
    # For higher dimensions, replace mod1 here with some geometry.
    src_modes = (src_indices[1].mode, src_indices[2].mode)
    dst_modes = (src_modes[1] + mom_change, src_modes[2] - mom_change)
    if fold
        dst_modes = (mod1(dst_modes[1], M), mod1(dst_modes[2], M))
    elseif !(1 ≤ dst_modes[1] ≤ M && 1 ≤ dst_modes[2] ≤ M)
        # Using a positive momentum change would have folded, so we try to use its negative
        # equivalent.
        mom_change = mom_change - M
        dst_modes = (src_modes[1] + mom_change, src_modes[2] - mom_change)
        if !(1 ≤ dst_modes[1] ≤ M && 1 ≤ dst_modes[2] ≤ M)
            return add, 0.0, src_modes..., -mom_change
        end
    end
    dst_indices = find_mode(add, dst_modes)
    return excitation(add, dst_indices, src_indices)..., src_modes..., -mom_change
end

@inline function momentum_transfer_excitation(
    add_a, add_b, chosen, map_a, map_b; fold=true
)
    M = num_modes(add_a)
    M == num_modes(add_b) || throw(ArgumentError("Addresses must have the same number of modes"))

    src_a, remainder = fldmod1(chosen, (M - 1) * length(map_b))
    dst_a, src_b = fldmod1(remainder, length(map_b))

    src_a_index = map_a[src_a]
    src_b_index = map_b[src_b]
    src_a_mode = src_a_index.mode

    if dst_a ≥ src_a_mode
        dst_a += 1 # to skip the src_a
    end
    mom_change = dst_a - src_a_mode # change in momentun
    dst_a_index = find_mode(add_a, dst_a)
    src_b_mode = src_b_index.mode
    dst_b = src_b_mode - mom_change

    # Additional info returned with result:
    params = (src_a_mode, src_b_mode, -mom_change)

    if fold
        dst_a = mod1(dst_a, M)
        dst_b = mod1(dst_b, M) # enforce periodic boundary condition
    elseif !(0 < dst_a ≤ M) || !(0 < dst_b ≤ M)
        return add_a, add_b, 0.0, params...
    end

    dst_b_index = find_mode(add_b, dst_b)

    new_add_a, val_a = excitation(add_a, (dst_a_index,), (src_a_index,))
    new_add_b, val_b = excitation(add_b, (dst_b_index,), (src_b_index,))

    return new_add_a, new_add_b, val_a * val_b, params...
end

"""
    momentum_transfer_diagonal(map)

The diagonal part of onsite [`momentum_transfer_excitation`](@ref).
"""
function momentum_transfer_diagonal(map::BoseOccupiedModeMap)
    onproduct = 0
    for i in 1:length(map)
        occ_i = map[i].occnum
        onproduct += occ_i * (occ_i - 1)
        for j in 1:i-1
            occ_j = map[j].occnum
            onproduct += 4 * occ_i * occ_j
        end
    end
    return float(onproduct)
end

"""
    extended_momentum_transfer_diagonal(map, step)

The diagonal part of nearest neighbour term [`momentum_transfer_excitation`](@ref) in [`ExtendedHubbardMom1D`](@ref).
Where `step` is the separation of single-particle momenta in the momentum grid.
"""
function extended_momentum_transfer_diagonal(map::OccupiedModeMap, step)
    onproduct = 0
    for i in 1:length(map)
        occ_i = map[i].occnum
        onproduct += occ_i * (occ_i - 1)
        for j in 1:i-1
            occ_j = map[j].occnum
            onproduct += 2*occ_i * occ_j * (1 - cos((map[j].mode - map[i].mode)*step))
        end
    end
    return float(onproduct)
end

function momentum_transfer_diagonal(
    map_a::FermiOccupiedModeMap, map_b::FermiOccupiedModeMap
)
    onproduct = 0
    n1 = length(map_a)
    n2 = length(map_b)

    return float(2 * n1 * n2)
end

"""
    transcorrelated_three_body_excitation(add↑, add↓, i, map↑, map↓)
    -> nadd↑, nadd↓, value, k, l

Apply the following operator to two addresses:

```math
a^†_{p+k,↑} a^†_{q+l,↑} a^†_{s-k-l,↓} a_{s,↓} a_{q,↑} a_{p,↑} |\\mathtt{add↑}⟩⊗|\\mathtt{add↓}⟩
```

The index `i` enumerates the possible non-zero terms and determines ``p, q, s, k, l``.
It is expected that `map↑, map↓ == OccupiedModeMap(add↑), OccupiedModeMap(add↓)`.

Return new addresses, prefactor `value`, `k`, and `l`. Note: If either `k` or `l` are zero,
or the excitation is diagonal, the function returns `value == 0`.

See [`transcorrelated_diagonal`](@ref), [`Transcorrelated1D`](@ref).
"""
function transcorrelated_three_body_excitation(add_a, add_b, i, map_a, map_b)
    N1 = length(map_a)
    N2 = length(map_b)
    M = num_modes(add_a)
    M == num_modes(add_b) || throw(ArgumentError("Addresses must have the same number of modes"))

    if add_a isa FermiFS # TODO: this is better done in the same way as momentum transfer
        p, q, s, p_k, q_l = Tuple(CartesianIndices((N1, N1 - 1, N2, M, M))[i])
        if q ≥ p
            q += 1
        end
    else
        p, q, s, p_k, q_l = Tuple(CartesianIndices((N1, N1, N2, M, M))[i])
    end

    p_index = map_a[p]
    q_index = map_a[q]
    s_index = map_b[s]

    k = p_index.mode - p_k
    l = q_l - q_index.mode
    s_kl = s_index.mode + k - l

    if k == 0 || l == 0
        # Zero because Q_kl == 0
        return add_a, add_b, 0.0, k,l
    elseif p_index.mode == q_l && q_index.mode == p_k
        # Diagonal
        return add_a, add_b, 0.0, k,l
    elseif s_kl > M || s_kl < 1
        # Out of bounds
        return add_a, add_b, 0.0, k,l
    end
    p_k_index, q_l_index = find_mode(add_a, (p_k, q_l))
    s_kl_index = find_mode(add_b, s_kl)
    new_add_a, val1 = excitation(add_a, (p_k_index, q_l_index), (q_index, p_index))
    new_add_b, val2 = excitation(add_b, (s_kl_index,), (s_index,))

    return new_add_a, new_add_b, val1 * val2, k,l
end

"""
    momentum_external_potential_excitation(ep, add, i, map::OccupiedModeMap) -> nadd, α

The momentum space version of an external potential. `ep` may be a discrete Fourier
transform of a real-space potential.

```math
\\sum_{k,q} \\mathtt{ep}_k a^†_{q+k} a_{q} |\\mathtt{add}⟩
```

Return the new address `nadd` and value `α` of the `i`th of the
`(num_modes(add)-1) * num_occupied_modes(add)` terms in the sum, excluding the diagonal
term `∝ |add⟩`. It is expected that `map == OccupiedModeMap(add)`.

See [`momentum_external_potential_diagonal`](@ref), [`OccupiedModeMap`](@ref),
[`num_occupied_modes`](@ref), [`num_modes`](@ref).
"""
function momentum_external_potential_excitation(ep, add, i, map::OccupiedModeMap)
    M = num_modes(add)
    p, q = fldmod1(i, M - 1) # i == (p-1)*(M-1) + q
    p_index = map[p] # p-th occupied mode in add
    q += q ≥ p_index.mode # leave out diagonal matrix element
    q_index = find_mode(add, q) # q-th mode in add (not counting p)
    k = p_index.mode - q # change in momentum
    factor = ep[mod(k, M) + 1]
    new_add, value = excitation(add, (q_index,), (p_index,)) # a_q^† a_p |add⟩
    return new_add, value * factor
end

"""
    momentum_external_potential_diagonal(ep, add, map::OccupiedModeMap)

The diagonal part of [`momentum_external_potential_excitation`](@ref).
"""
function momentum_external_potential_diagonal(ep, add, map::OccupiedModeMap)
    onproduct = sum(map) do index
        index.occnum
    end
    return onproduct * ep[1]
end
