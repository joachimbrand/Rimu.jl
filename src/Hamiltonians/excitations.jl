using Rimu
using Rimu.Hamiltonians: hopnextneighbour

"""
    real_space_excitation(add, chosen, map, geometry)

Apply a real space hop operator:

```math
a^†_{p} a_q
```

where `p` and `q` are neighbour sites in `geometry`.
"""
@inline function real_space_excitation(add, chosen, map, geometry)
    neighbours = num_neighbours(geometry)
    particle, neigh = fldmod1(chosen, neighbours)
    src_index = map[particle]
    neigh = neighbour_site(geometry, src_index.mode, neigh)
    if neigh == 0
        return add, 0.0
    else
        dst_index = find_mode(add, neigh)
        return excitation(add, (dst_index,), (src_index,))
    end
end

"""
    momentum_transfer_excitation(add, chosen, map; fold=true)
    momentum_transfer_excitation(add1, add2, chosen, map1, map2; fold=true)

Apply the momentum transfer operator to Fock address (or pair of addresses) `add` (or
`add1`, `add2`):

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
    transcorrelated_three_body_excitation(f1::FermiFS{N1,M}, f2::FermiFS{N2,M}, i)

Apply the following operator to two addresses:

```math
a^†_{p+k,1} a^†_{q+l,1} a^†_{s-k-l,2} a_{s,2} a_{q,1} a_{p,1}
```

Return new addresses, value, `k`, and `l`. Note: if either `k` or `l` are zero, the
function returns zero.
"""
function transcorrelated_three_body_excitation(add1, add2, i, map1, map2)
    N1 = length(map1)
    N2 = length(map2)
    M = num_modes(add1)

    if add1 isa FermiFS # TODO: this is better done in the same way as momentum transfer
        p, q, s, p_k, q_l = Tuple(CartesianIndices((N1, N1 - 1, N2, M, M))[i])
        if q ≥ p
            q += 1
        end
    else
        p, q, s, p_k, q_l = Tuple(CartesianIndices((N1, N1, N2, M, M))[i])
    end

    p_index = map1[p]
    q_index = map1[q]
    s_index = map2[s]

    k = p_index.mode - p_k
    l = q_l - q_index.mode
    s_kl = s_index.mode + k - l

    if k == 0 || l == 0
        # Zero because Q_kl == 0
        return add1, add2, 0.0, k,l
    elseif p_index.mode == q_l && q_index.mode == p_k
        # Diagonal
        return add1, add2, 0.0, k,l
    elseif s_kl > M || s_kl < 1
        # Out of bounds
        return add1, add2, 0.0, k,l
    end
    p_k_index, q_l_index = find_mode(add1, (p_k, q_l))
    s_kl_index = find_mode(add1, s_kl)
    new_add1, val1 = excitation(add1, (p_k_index, q_l_index), (q_index, p_index))
    new_add2, val2 = excitation(add2, (s_kl_index,), (s_index,))

    return new_add1, new_add2, val1 * val2, k,l
end
