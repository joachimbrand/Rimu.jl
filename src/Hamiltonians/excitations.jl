using Rimu
using Rimu.Hamiltonians: hopnextneighbour

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
    momentum_transfer_excitation(add, chosen, singlies, doublies)
Internal function used in [`get_offdiagonal`](@ref) for [`HubbardMom1D`](@ref)
and [`G2Correlator`](@ref). Returns the new address, the onproduct,
and the change in momentum.
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
    dst_modes = getproperty.(src_indices, :mode) .+ (mom_change, -mom_change)
    if fold
        dst_modes = mod1.(dst_modes, M)
    elseif !all(1 .≤ dst_modes .≤ M)
        return add, 0.0, -mom_change
    end
    dst_indices = find_mode(add, dst_modes)
    return excitation(add, dst_indices, src_indices)..., -mom_change
end

function momentum_transfer_excitation(add::FermiFS, chosen, map; fold=true)
    return add, 0.0, 0
end

@inline function momentum_transfer_excitation(
    add_a, add_b, chosen, map_a, map_b; fold=true
)
    M = num_modes(add_a)

    src_a, remainder = fldmod1(chosen, (M - 1) * length(map_b))
    p, src_b = fldmod1(remainder, length(map_b))

    src_a_index = map_a[src_a]
    src_b_index = map_b[src_b]
    r = src_a_index.mode

    if p ≥ r
        p += 1 # to skip the src_a
    end
    ΔP = p - r # change in momentun
    dst_a_index = find_mode(add_a, p)
    q = src_b_index.mode

    if fold
        s = mod1(q - ΔP, M)
        p = mod1(p, M) # enforce periodic boundary condition
    elseif s > M || p > M || s < 1 || p < 1
        return add_a, add_b, 0.0, q, s
    end

    dst_b_index = find_mode(add_b, s)

    new_add_a, val_a = excitation(add_a, (dst_a_index,), (src_a_index,))
    new_add_b, val_b = excitation(add_b, (dst_b_index,), (src_b_index,))

    return new_add_a, new_add_b, val_a * val_b, p, q, s
end

"""
    transcorrelated_three_body_excitation(f1::FermiFS{N1,M}, f2::FermiFS{N2,M}, i)

Apply the following operator to two addresses:

```math
a^†_{p+k,1} a^†_{q+l,1} a^†_{s-k-l,2} a_{s,2} a_{q,1} a_{p,1}
```

Return new addresses, value, `k`, and `l`.
"""
function transcorrelated_three_body_excitation(
    f1::FermiFS{N1,M}, f2::FermiFS{N2,M}, i
) where {N1,N2,M}
    p, q, s, p_k, q_l = Tuple(CartesianIndices((N1, N1, N2, M, M))[i])

    p_index, q_index = find_occupied_mode(f1, (p, q))
    k = p_index - p_k
    l = q_l - q_index
    s_index = find_occupied_mode(f2, s)
    s_kl = s_index + k - l

    if k == 0 || l == 0
        # Zero because Q_kl == 0
        return f1, f2, 0.0, k,l
    elseif p_index == q_l && q_index == p_k
        # Diagonal
        return f1, f2, 0.0, k,l
    elseif s_kl > M || s_kl < 1
        # Out of bounds
        return f1, f2, 0.0, k,l
    end
    p_k_index, q_l_index = find_mode(f1, (p_k, q_l))
    s_kl_index = find_mode(f1, s_kl)
    new_f1, val1 = excitation(f1, (p_k_index, q_l_index), (q_index, p_index))
    new_f2, val2 = excitation(f2, (s_kl_index,), (s_index,))

    return new_f1, new_f2, val1 * val2, k,l
end
