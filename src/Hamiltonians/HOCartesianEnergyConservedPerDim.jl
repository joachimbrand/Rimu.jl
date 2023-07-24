# TO-DO: optimise if large arguments are used
"""
    four_oscillator_integral_1D(i, j, k, l; max_level = typemax(Int))

Integral of four one-dimensional harmonic oscillator functions, 
```math
    \\mathcal{I}(i,j,k,l) = \\int_{-\\infty}^\\infty dx \\, 
    \\phi_i(x) \\phi_j(x) \\phi_k(x) \\phi_l(x)
```
assuming ``i+j = k+l``. State indices `i,j,k,l` start at `0` for the groundstate.
Closed form taken from [Papenbrock (2002)](https://doi.org/10.1103/PhysRevA.65.033606).
Used in [`HOCartesianEnergyConservedPerDim`](@ref).
"""
function four_oscillator_integral_1D(i, j, k, l; max_level = typemax(Int))
    !all(0 .≤ (i,j,k,l) .≤ max_level) && return 0.0
    min_kl = min(k, l)
    # SpecialFunctions.gamma(x) is faster than factorial(big(x))
    p = sqrt(gamma(k + 1) * gamma(l + 1)) / sqrt(2 * gamma(i + 1) * gamma(j + 1)) / pi^2
    s = sum(gamma(t + 1/2) * gamma(i - t + 1/2) * gamma(j - t + 1/2) / (gamma(t + 1) * gamma(k - t + 1) * gamma(l - t + 1)) for t in 0 : min_kl)
    
    return p * s
end

"""
    largest_two_point_box(i, j, dims::Tuple) -> ranges

For a box defined by `dims` containing two points `i` and `j`, find the set of all positions 
such that shifting `i` to that position with an opposite shift to `j` leaves both points inbounds.

Arguments `i` and `j` are linear indices, `dims` is a `Tuple` defining the inbounds area.
"""
function largest_two_point_box(i::Int, j::Int, dims::NTuple{D,Int}) where {D}
    cart = CartesianIndices(dims)
    state_i = Tuple(cart[i])

    state_j = Tuple(cart[j])

    ranges = ntuple(d -> largest_two_point_interval(state_i[d], state_j[d], dims[d]), Val(D))
    
    box_size = prod(length.(ranges))
    return ranges, box_size
end

"""
    find_chosen_pair_moves(omm::OccupiedModeMap, c, S) -> p_i, p_j, c, box_ranges

Find size of valid moves for chosen pair of indices in `omm`. Returns two valid indices `p_i` and `p_j`
for the initial pair and a tuple of ranges `box_ranges` defining the subbox of valid moves. 
The index for the chosen move `c` is updated to be valid for this box.
Arguments are the `OccupiedModeMap` `omm`, the chosen move index `c`
and the size of the basis grid `S`.
"""
function find_chosen_pair_moves(omm::OccupiedModeMap, c, S::Tuple)
    for i in eachindex(omm)
        p_i = omm[i]
        if p_i.occnum > 1
            box_ranges, box_size = largest_two_point_box(p_i.mode, p_i.mode, S)
            if c - box_size < 1
                return p_i, p_i, box_ranges, c
            else
                c -= box_size 
            end
        end
        for j in 1:i-1
            p_j = omm[j]
            box_ranges, box_size = largest_two_point_box(p_i.mode, p_j.mode, S)
            if c - box_size < 1
                return p_i, p_j, box_ranges, c
            else
                c -= box_size 
            end
        end
    end
    error("chosen pair not found")
end

"""
    HOCartesianEnergyConservedPerDim(addr; S, η, g = 1.0, interaction_only = false)

Implements a bosonic harmonic oscillator in Cartesian basis with contact interactions 
```math
\\hat{H} = \\sum_{i} ϵ_i n_i + \\frac{g}{2}\\sum_{ijkl} V_{ijkl} a^†_i a^†_j a_k a_l,
```
with the additional restriction that the interactions only couple states with the same
energy in each dimension separately. See [`HOCartesianEnergyConserved`](@ref) for a model that 
conserves total energy.

For a ``D``-dimensional harmonic oscillator indices ``\\mathbf{i}, \\mathbf{j}, \\ldots`` 
are ``D``-tuples. The energy scale is defined by the first dimension i.e. ``\\hbar \\omega_x`` 
so that single particle energies are 
```math
    \\frac{\\epsilon_\\mathbf{i}}{\\hbar \\omega_x} = (i_x + 1/2) + \\eta_y (i_y+1/2) + \\ldots.
```
The factors ``\\eta_y, \\ldots`` allow for anisotropic trapping geometries and are assumed to 
be greater than `1` so that ``\\omega_x`` is the smallest trapping frequency.

Matrix elements ``V_{\\mathbf{ijkl}}`` are for a contact interaction calculated in this basis using 
first-order degenerate perturbation theory.
```math
    V_{\\mathbf{ijkl}} = \\prod_{d \\in x, y,\\ldots} \\mathcal{I}(i_d,j_d,k_d,l_d) 
        \\delta_{i_d + j_d}^{k_d + l_d},
```
where the ``\\delta``-function indicates that the noninteracting energy is conserved along each
dimension.
The integral ``\\mathcal{I}(a,b,c,d)`` is of four one dimensional harmonic oscillator 
basis functions with the restriction that energy is conserved in each dimension, 
see [`four_oscillator_integral_1D`](@ref).

# Arguments

* `addr`: the starting address, defines number of particles and total number of modes.
* `S`: Tuple of the number of levels in each dimension, including the groundstate. Defaults 
    to a 1D spectrum with number of levels matching modes of `addr`. Will be sorted to 
    make the first dimension the largest.
* `η`: Define a custom aspect ratio for the trapping potential strengths, instead of deriving
    from `S .- 1`. The values are always scaled relative to the first dimension, which sets 
    the energy scale of the system, ``\\hbar\\omega_x``.
* `g`: the (isotropic) interparticle interaction parameter. The value of `g` is assumed 
    to be in trap units.
* `interaction_only`: if set to `true` then the noninteracting single-particle terms are 
    ignored. Useful if only energy shifts due to interactions are required.


"""
struct HOCartesianEnergyConservedPerDim{
    D,  # number of dimensions
    A<:BoseFS
} <: AbstractHamiltonian{Float64}
    addr::A
    S::NTuple{D,Int64}
    aspect1::NTuple{D,Float64}
    energies::Vector{Float64} # noninteracting single particle energies
    vtable::Array{Float64,3}  # interaction coefficients
    u::Float64
end

function HOCartesianEnergyConservedPerDim(
        addr::BoseFS; 
        S = (num_modes(addr),),
        η = nothing, 
        g = 1.0,
        interaction_only = false
    )
    D = length(S)
    P = prod(S)
    P == num_modes(addr) || throw(ArgumentError("number of modes does not match starting address"))

    if isnothing(η)
        aspect1 = float.(box_to_aspect(S))
    elseif length(η) == D
        aspect1 = Tuple(η ./ η[1])
    elseif length(η) == 1 && isreal(η)
        aspect1 = ntuple(i -> (i == 1 ? 1.0 : η), D)
    else
        throw(ArgumentError("Invalid aspect ratio parameter η."))
    end

    if interaction_only
        energies = zeros(P)
    else
        states = CartesianIndices(S)    # 1-indexed
        energies = reshape(map(x -> dot(aspect1, Tuple(x) .- 1/2), states), P)
    end

    # the aspect ratio appears from a change of variable when calculating the interaction integrals
    u = g / (2 * sqrt(prod(aspect1)))

    M = maximum(S)
    bigrange = 0:M-1
    # case n = M is the same as n = 0
    vmat = reshape(
                [four_oscillator_integral_1D(i-n, j+n, j, i; max_level = M-1) 
                    for i in bigrange, j in bigrange, n in bigrange],
                    M,M,M)

    return HOCartesianEnergyConservedPerDim{D,typeof(addr)}(addr, S, aspect1, energies, vmat, u)
end

function Base.show(io::IO, h::HOCartesianEnergyConservedPerDim)
    flag = iszero(h.energies)
    # invert the scaling of u parameter
    g = h.u * 2 * sqrt(prod(h.aspect1))
    print(io, "HOCartesianEnergyConservedPerDim($(h.addr); S=$(h.S), η=$(h.aspect1), g=$g, interaction_only=$flag)")
end

Base.:(==)(H::HOCartesianEnergyConservedPerDim, G::HOCartesianEnergyConservedPerDim) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

function starting_address(h::HOCartesianEnergyConservedPerDim)
    return h.addr
end

LOStructure(::Type{<:HOCartesianEnergyConservedPerDim}) = IsHermitian()


### DIAGONAL ELEMENTS ###
function energy_transfer_diagonal(h::HOCartesianEnergyConservedPerDim{D}, omm::BoseOccupiedModeMap) where {D}
    result = 0.0
    states = CartesianIndices(h.S)    # 1-indexed
    
    for i in eachindex(omm)
        mode_i, occ_i = omm[i].mode, omm[i].occnum
        idx_i = Tuple(states[mode_i])
        if occ_i > 1
            # use i in 1:M indexing for accessing table
            val = occ_i * (occ_i - 1)
            result += prod(h.vtable[idx_i[d],idx_i[d],1] for d in 1:D) * val
        end
        for j in 1:i-1
            mode_j, occ_j = omm[j].mode, omm[j].occnum
            idx_j = Tuple(states[mode_j])
            val = 4 * occ_i * occ_j
            result += prod(h.vtable[idx_i[d],idx_j[d],1] for d in 1:D) * val
        end        
    end
    return result * h.u
end

noninteracting_energy(h::HOCartesianEnergyConservedPerDim, omm::BoseOccupiedModeMap) = dot(h.energies, omm)
@inline function noninteracting_energy(h::HOCartesianEnergyConservedPerDim, addr::BoseFS)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm)
end
# fast method for finding blocks
noninteracting_energy(h::HOCartesianEnergyConservedPerDim, t::Union{Vector{Int64},NTuple{N,Int64}}) where {N} = sum(h.energies[j] for j in t)

@inline function diagonal_element(h::HOCartesianEnergyConservedPerDim, addr::BoseFS)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm) + energy_transfer_diagonal(h, omm)
end

### OFFDIAGONAL ELEMENTS ###

# includes swap moves and trivial moves
# To-Do: optimise these out for FCIQMC
function num_offdiagonals(h::HOCartesianEnergyConservedPerDim, addr::BoseFS)
    S = h.S
    omm = OccupiedModeMap(addr)
    noffs = 0

    for i in eachindex(omm)
        p_i = omm[i]
        if p_i.occnum > 1
            _, valid_box_size  = largest_two_point_box(p_i.mode, p_i.mode, S)
            noffs += valid_box_size 
        end
        for j in 1:i-1
            p_j = omm[j]
            _, valid_box_size  = largest_two_point_box(p_i.mode, p_j.mode, S)
            noffs += valid_box_size 
        end
    end
    return noffs
end


"""
    energy_transfer_offdiagonal(S, addr, chosen, omm = OccupiedModeMap(addr))
        -> new_add, val, mode_i, mode_j, mode_l

Return the new address `new_add`, the prefactor `val`, the initial particle modes
`mode_i` and `mode_j` and the new mode for `i`th particle, `mode_l`. The other new mode
`mode_k` is implicit by energy conservation.
"""
function energy_transfer_offdiagonal(
        S::Tuple, 
        addr::BoseFS, 
        chosen::Int, 
        omm::BoseOccupiedModeMap = OccupiedModeMap(addr)
    )
    # find size of valid moves for each pair
    particle_i, particle_j, valid_box_ranges, chosen = find_chosen_pair_moves(omm, chosen, S)
    mode_i = particle_i.mode
    mode_j = particle_j.mode

    # This is probably not optimal
    mode_l = LinearIndices(CartesianIndices(S))[CartesianIndices(valid_box_ranges)[chosen]]
    # discard swap moves and self moves
    if mode_l == mode_j || mode_l == mode_i
        return addr, 0.0, 0, 0, 0, 0
    end
    mode_Δn = mode_l - mode_i
    mode_k = mode_j - mode_Δn
    particle_k = find_mode(addr, mode_k)
    particle_l = find_mode(addr, mode_l)

    new_add, val = excitation(addr, (particle_l, particle_k), (particle_j, particle_i))

    return new_add, val, mode_i, mode_j, mode_l
end

function get_offdiagonal(
        h::HOCartesianEnergyConservedPerDim{D,A}, 
        addr::A, 
        chosen::Int, 
        omm::BoseOccupiedModeMap = OccupiedModeMap(addr)
    ) where {D,A}

    S = h.S
    
    new_add, val, i, j, l = energy_transfer_offdiagonal(S, addr, chosen, omm)

    if val ≠ 0.0    # is this a safe check? maybe check if Δns == (0,...)
        states = CartesianIndices(S)    # 1-indexed
        idx_i = Tuple(states[i])
        idx_j = Tuple(states[j])
        idx_l = Tuple(states[l])

        # sort indices to match table of values
        idx_i_sort = ntuple(d -> idx_i[d] > idx_l[d] ? idx_i[d] : idx_j[d], Val(D))
        idx_j_sort = ntuple(d -> idx_i[d] > idx_l[d] ? idx_j[d] : idx_i[d], Val(D))
        idx_Δns = ntuple(d -> abs(idx_i[d] - idx_l[d]) + 1, Val(D))

        val *= prod(h.vtable[a,b,c] for (a,b,c) in zip(idx_i_sort, idx_j_sort, idx_Δns))

        # account for swap of (i,j)
        val *= (1 + (i ≠ j)) * h.u
    end
    return new_add, val
end

###
### offdiagonals
###
"""
    HOCartSeparableOffdiagonals

Specialized [`AbstractOffdiagonals`](@ref) that keeps track of singly and doubly occupied
sites in current address.
"""
struct HOCartSeparableOffdiagonals{
    A<:BoseFS,T,H<:AbstractHamiltonian{T},O<:OccupiedModeMap
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    map::O
end

function offdiagonals(h::HOCartesianEnergyConservedPerDim, addr::BoseFS)
    omm = OccupiedModeMap(addr)
    num = num_offdiagonals(h, addr)
    return HOCartSeparableOffdiagonals(h, addr, num, omm)
end

function Base.getindex(s::HOCartSeparableOffdiagonals{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i, s.map)
    return (new_address, matrix_element)
end

Base.size(s::HOCartSeparableOffdiagonals) = (s.length,)
