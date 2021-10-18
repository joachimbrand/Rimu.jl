using Rimu
using Rimu.Hamiltonians: hopnextneighbour

const BoseOccupiedModeMap{N} = OccupiedModeMap{N,BitStringAddresses.BoseFSIndex}
const FermiOccupiedModeMap{N} = OccupiedModeMap{N,BitStringAddresses.FermiFSIndex}

"""
    momentum_transfer_excitation(add, chosen, map; fold=true)
    momentum_transfer_excitation(add_a, add_b, chosen, map_a, map_b; fold=true)

Apply the momentum transfer operator to Fock address (or pair of addresses) `add` (or
`add_a`, `add_b`):

```math
a^†_{p + k} a^†{q - k} a_q a_p
```

The `fold` argument controls whether the terms `p + k` and `q - k` are done modulo M. If
not, zero is returned when either of those terms is less than 1 or larger than M.

Return the new address(es), the value, modes `p` and `q`, and the momentum change `-k`.
"""
@inline function momentum_transfer_excitation(
    add::BoseFS, chosen, map::OccupiedModeMap; fold=true
)
    M = num_modes(add)
    singlies = length(map)
    doublies = count(i -> i.occnum ≥ 2, map)

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
        src_indices = getindex.(Ref(map), (f_hole, s_hole))
        f_mode, s_mode = src_indices[1].mode, src_indices[2].mode
        if mom_change ≥ s_mode - f_mode
            mom_change += 1 # to avoid putting particles back into the holes
        end
    end
    # For higher dimensions, replace mod1 here with some geometry.
    src_modes = getproperty.(src_indices, :mode)
    dst_modes = src_modes .+ (mom_change, -mom_change)
    if fold
        dst_modes = mod1.(dst_modes, M)
    elseif !all(1 .≤ dst_modes .≤ M)
        return add, 0.0, src_modes..., -mom_change
    end
    dst_indices = find_mode(add, dst_modes)
    return excitation(add, dst_indices, src_indices)..., src_modes..., -mom_change
end

function momentum_transfer_excitation(add::FermiFS, chosen::Integer, map; fold=true)
    return add, 0.0, 0, 0, 0
end

@inline function momentum_transfer_excitation(
    add_a, add_b, chosen, map_a, map_b; fold=true
)
    M = num_modes(add_a)

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

The diagonal part of [`momentum_transfer_excitation`](@ref).
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
function momentum_transfer_diagonal(
    map_a::FermiOccupiedModeMap, map_b::FermiOccupiedModeMap
)
    onproduct = 0
    n1 = length(map_a)
    n2 = length(map_b)

    return float(2 * n1 * n2)
end
function kinetic_energy(kes, map)
    value = 0.0
    for index in map
        value += kes[index.mode] * index.occnum
    end
    return value
end

"""
    transcorrelated_three_body_excitation(f1::FermiFS{N1,M}, f2::FermiFS{N2,M}, i)

Apply the following operator to two addresses:

```math
a^†_{p+k,1} a^†_{q+l,1} a^†_{s-k-l,2} a_{s,2} a_{q,1} a_{p,1}
```

Return new addresses, value, `k`, and `l`. Note: if either `k` or `l` are zero, the
function returns zero.
"""
function transcorrelated_three_body_excitation(add_a, add_b, i, map_a, map_b)
    N1 = length(map_a)
    N2 = length(map_b)
    M = num_modes(add_a)

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
    s_kl_index = find_mode(add_a, s_kl)
    new_add_a, val1 = excitation(add_a, (p_k_index, q_l_index), (q_index, p_index))
    new_add_b, val2 = excitation(add_b, (s_kl_index,), (s_index,))

    return new_add_a, new_add_b, val1 * val2, k,l
end

"""
    momentum_external_potential_excitation(ep, add, i, map)

The momentum space version of an external potential. `ep` must be the output a DFT of a
potential.

```math
ep[p - q mod M + 1] a^†_{p} a_{q}
```

Return the new address, and the value.
"""
function momentum_external_potential_excitation(ep, add, i, map)
    M = num_modes(add)
    p, q = fldmod1(i, M - 1)
    p_index = map[p]
    q += q ≥ p_index.mode
    q_index = find_mode(add, q)
    k = p_index.mode - q # change in momentum
    factor = 1/M * ep[mod(k, M) + 1]
    new_add, value = excitation(add, (q_index,), (p_index,))
    return new_add, value * factor
end

"""
    momentum_external_potential_diagonal(ep, add, map)

The diagonal part of [`momentum_external_potential_excitation`](@ref).
"""
function momentum_external_potential_diagonal(::Nothing, add, map)
    return 0.0
end
function momentum_external_potential_diagonal(ep, add, map)
    M = num_modes(add)
    onproduct = sum(map) do index
        index.occnum
    end
    return onproduct * 1/M * ep[1]
end
