"""
    HOCartesian(addr; g = 1.0) <: AbstractHamiltonian

Hamiltonian for particles interacting with contact interactions with a harmonic
oscillator external potential. The underlying single-particle basis states are the
eigenstates of the external potential. `g` is the value of the bare interaction strength.
"""
struct HOCartesian{A,T} <: AbstractHamiltonian{T}
    addr::A
    g::T
    energies::Vector{Float64}   # noninteracting single particle energies
end
function HOCartesian(addr; g=1.0)
    g = float(g)
    addr isa BoseFS || throw(ArgumentError("Only `BoseFS` is currently supported."))
    energies = [0.5 + i for i in 0:(num_modes(addr)-1)]
    return HOCartesian(addr, g, energies)
end

starting_address(h::HOCartesian) = h.addr

"""
    num_even_parity_excitations(addr::BoseFS{<:Any,M}[, pairs]) where {M}

Return the number of possible even parity excitations.
"""
function num_even_parity_excitations(addr)
    return num_even_parity_excitations(addr, OccupiedPairsMap(addr))
end
function num_even_parity_excitations(addr::BoseFS{<:Any,M}, pairs) where {M}
    count = 0
    even_pairs = num_even_pairs(M)
    odd_pairs = M * (M + 1) ÷ 2 - even_pairs
    for (i, j) in pairs
        count += ifelse(iseven((i.mode - 1) * (j.mode - 1)), even_pairs, odd_pairs)
    end
    return count
end

"""
    even_parity_excitations(addr::BoseFS, pairs)

Return the number of pairs `i, j` for particle creation for each
of the particle destructions in `pairs` `k, l` such that the two-particle excitation
``a^\\dag_i a^\\dag_j a_l a_k`` has even parity.
"""
@inline function even_parity_excitations(::BoseFS{<:Any,M}, pairs) where {M}
    return even_parity_excitations(M, pairs)
end
@inline function even_parity_excitations(M, pairs)
    even_pairs = num_even_pairs(M)
    odd_pairs = M * (M + 1) ÷ 2 - even_pairs
    possible_excitations = map(pairs) do (i,j)
        ifelse(iseven((i.mode - 1) * (j.mode - 1)), even_pairs, odd_pairs)
    end
    return possible_excitations
end

"""
    num_even_pairs(m)

Return the number of even parity pairs `(i,j)` of bosons in `m` modes, where
`0 ≤ i ≤ j < m`.
"""
@inline function num_even_pairs(m::Integer)
    count = 0
    for i = 0:(m-1), j = i:(m-1)
        count += iseven(i + j)
    end
    return count
end

import Rimu.Interfaces: num_offdiagonals

num_offdiagonals(::HOCartesian, addr) = num_even_parity_excitations(addr)

get_offdiagonal(h::HOCartesian, addr::BoseFS, chosen) = offdiagonals(h, addr)[chosen]

offdiagonals(h::HOCartesian, addr) = HOCartesianOffdiagonals(h, addr)

"""
    HOCartesianOffdiagonals(h::HOCartesian, addr) <: AbstractOffdiagonals

Iterator over new address and matrix element for reachable off-diagonal matrix elements of
`HOCartesian` operator `h` from address `addr`.
"""
struct HOCartesianOffdiagonals{A,T,H<:HOCartesian{A,T},P,E} <: AbstractOffdiagonals{A,T}
    ham::H
    addr::A
    num_pairs::Int # number of `occupied_pairs` and `even_parity_pairs`
    occupied_pairs::P # overfilled StaticVector
    even_parity_pairs::E # overfilled StaticVector
    length::Int
end
function HOCartesianOffdiagonals(h::HOCartesian, addr)
    o_pairs = OccupiedPairsMap(addr)
    num_pairs = length(o_pairs)
    e_pairs = even_parity_excitations(addr, o_pairs.pairs)
    num = 0 # compute number of even parity pairs
    for i in 1:num_pairs
        @inbounds num += e_pairs[i]
    end
    return HOCartesianOffdiagonals(h, addr, num_pairs, o_pairs.pairs, e_pairs, num)
end
Base.size(od::HOCartesianOffdiagonals) = (od.length,)

# jump directly to the relevant excitation, returning four `(BoseFS/FermiFS)Index` objects
function _get_excitation_indices(od::HOCartesianOffdiagonals, i::Integer)
    pair_index = 1
    while i > od.even_parity_pairs[pair_index]
        i -= od.even_parity_pairs[pair_index]
        pair_index += 1
    end
    # now i is the index of the excitation within the pair
    # and pair_index is the index of the pair
    # we need to find the excitation
    kk, ll = od.occupied_pairs[pair_index]
    m = num_modes(od.addr)
    count = 0
    for k = 0:(m-1), l = k:(m-1)
        if iseven(k + l)
            # (k, l) == (kk.mode - 1, ll.mode - 1) && continue # skip diagonal excitation
            count += 1
            if count == i # found the excitation
                ii, jj = find_mode(od.addr, (k+1, l+1)) # mode count is 1-based
                return ii, jj, od.occupied_pairs[pair_index]...
            end
        end
    end
    throw(ArgumentError("should not reach here: pair_index = $pair_index, i = $i"))
end

function Base.getindex(od::HOCartesianOffdiagonals, i::Integer)
    @boundscheck 1 ≤ i ≤ od.length || throw(BoundsError(od, i))
    ii, jj, kk, ll = _get_excitation_indices(od, i)
    # return (ii, jj), (kk, ll)
    naddr, val = excitation(od.addr, (ii, jj), (kk, ll))
    val *= (1 + (ii ≠ jj)) * (1 + (kk ≠ ll)) # account for jj ≤ ii and ll ≤ kk
    val *= (ii, jj) != (kk, ll) # return zero if excitation is diagonal
    val *= od.ham.g/2 * four_oscillator_integral_general(ii.mode, jj.mode, kk.mode, ll.mode)
    return naddr, val
end

@inline function noninteracting_energy(h::HOCartesian, omm::BoseOccupiedModeMap)
    return dot(h.energies, omm)
end

@inline function diagonal_element(h::HOCartesian, addr::BoseFS)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm) + diagonal_interactions(h, omm)
end

@inline function diagonal_interactions(h::HOCartesian, omm::BoseOccupiedModeMap)
    pairs = OccupiedPairsMap(omm)
    energy = sum(pairs) do (ii, jj)
        four_oscillator_integral_general(ii.mode, jj.mode, ii.mode, jj.mode) *
            # account for ii ≤ jj
            ii.occnum * ifelse(ii == jj, ii.occnum - 1, jj.occnum * 4)
    end
    return h.g/2 * energy
end

# TODO: write tests
