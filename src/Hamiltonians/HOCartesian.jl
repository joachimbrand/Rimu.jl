"""
    four_oscillator_integral_general(i, j, k, l; max_level = typemax(Int))

Integral of four one-dimensional harmonic oscillator functions, following Titchmarsh (1948).
State indices `i,j,k,l` start at `0` for the groundstate.
"""
function four_oscillator_integral_general(i, j, k, l; max_level = typemax(Int))
    all(0 .≤ (i, j, k, l) .≤ max_level) || return 0.0
    iseven(i + j + k + l) || return 0.0

    # enforce correct symmetry and k ≥ l
    i, j, k, l = sort(SVector(i, j, k, l); rev = true)
    
    a = i + j - k + l + 1    
    b = i - j + k - l + 1
    c = -i + j + k - l + 1
    d = -i - j + k - l + 1
    
    p = sqrt(2 * gamma(i + 1) * gamma(j + 1) * gamma(k + 1) * gamma(l + 1)) * pi^2
    q = gamma(a/2) * gamma(b/2) * gamma(c/2) * gamma(k + 1) / gamma(k - l + 1)
    
    f = _₃F₂(float(-l), b/2, c/2, float(1 + k - l), d/2, 1.0)
    
    return f * q / p
end

"""
    largest_two_point_interval(i, j, [v=1,] w) -> range

For two points `i` and `j` on a range `v:w`, find the largest subinterval of sites 
such that moving `i` to one of those sites and moving `j` by an equal but opposite
amount leaves both within `v:w`.

`i` and `j` may be `Rational`s, but `v` and `w` must be `Integer`s.
"""
function largest_two_point_interval(i::Int, j::Int, v::Int, w::Int)
    i in v:w && j in v:w || throw("interval out of bounds: i=$i, j=$j, v=$v, w=$w")
    left_edge, right_edge = two_point_interval_bounds(i, j, v, w)
    return left_edge:right_edge
end
function largest_two_point_interval(i, j, v::Int, w::Int)
    v ≤ i ≤ w && v ≤ j ≤ w || throw("interval out of bounds: i=$i, j=$j, v=$v, w=$w")
    left_edge, right_edge = two_point_interval_bounds(i, j, v, w)
    return ceil(Int, left_edge):floor(Int, right_edge)
end
# largest_two_point_interval(i::Int, j::Rational, v, w) = largest_two_point_interval(i//1, j, v, w)   # are these necessary?
# largest_two_point_interval(i::Rational, j::Int, v, w) = largest_two_point_interval(i, j//1, v, w)
largest_two_point_interval(i, j, w) = largest_two_point_interval(i, j, 1, w)

function two_point_interval_bounds(i, j, v, w)
    if i ≤ j
        left_gap = min(i - v, w - j)
        left_edge = i - left_gap
        right_gap = min(w - i, j - v)
        right_edge = i + right_gap
    else
        left_gap = min(j - v, w - i)
        left_edge = j - left_gap
        right_gap = min(w - j, i - v)
        right_edge = j + right_gap
    end
    return left_edge, right_edge
end

# useful conversion of aspect ratio to a factor for getting box size
aspect_to_box(η::NTuple{N,Int}) where {N} = Int.(lcm(η...) .// η)
box_to_aspect(S) = (S[1] - 1) ./ (S .- 1)
box_to_aspect(S::NTuple{N,Int}) where {N} = (S[1] - 1) .// (S .- 1)

@inline function mode_to_energy(i, S, aspect)
    # aspect = box_to_aspect(S)
    states = CartesianIndices(S)
    # account for 1-indexing here in case aspect ratio is fractional
    idx = Tuple(states[i]) .- 1
    E = dot(idx, aspect)
    # shift back to 1-indexing
    return E + 1
end

"""
    find_Ebounds(i, j, S)

Find the range of single particle energies that a particle in mode `i` can move to while 
preserving total energy of particles `i` and `j`, and constrained to box `S`.
"""
function find_Ebounds(i, j, S, aspect)
    
    E_i = mode_to_energy(i, S, aspect)
    E_j = mode_to_energy(j, S, aspect)

    # shift back to 1-indexing
    return extrema(largest_two_point_interval(E_i, E_j, S[1]))..., E_i + E_j
end

"""
    OccupiedPairsMap(omm::OccupiedModeMap) <: AbstractVector

Get all pairs of particles in `omm`. Pairs involving multiply-occupied modes 
are counted once, (including self pairing).
This is an eager iterator whose elements are a tuple of particle indices.

See [`OccupiedModeMap`](@ref).
"""
struct OccupiedPairsMap{N,T} <: AbstractVector{T}
    pairs::SVector{N,T}
    length::Int
end

function OccupiedPairsMap(addr::SingleComponentFockAddress{N}) where {N}
    omm = OccupiedModeMap(addr)
    T = eltype(omm)
    P = N * (N - 1) ÷ 2
    pairs = MVector{P,Tuple{T,T}}(undef)
    L = 0
    for i in eachindex(omm)
        p_i = omm[i]
        if p_i.occnum > 1
            L += 1
            @inbounds pairs[L] = (p_i, p_i)
        end
        for j in 1:i-1
            p_j = omm[j]
            L += 1
            @inbounds pairs[L] = (p_i, p_j)
        end
    end
    
    return OccupiedPairsMap(SVector(pairs), L)
end

Base.size(opm::OccupiedPairsMap) = (opm.length,)
function Base.getindex(opm::OccupiedPairsMap, i)
    @boundscheck 1 ≤ i ≤ opm.length || throw(BoundsError(opm, i))
    return opm.pairs[i]
end

"""
    HOCartesian(addr; S, η, g = 1.0, interaction_only = false)

Implements a harmonic oscillator in Cartesian basis with contact interactions.
```math
\\hat{H} = \\sum_{i} \\epsilon_i n_i + \\frac{g}{2}\\sum_{ijkl} V_{ijkl} a^†_i a^†_j a_k a_l
```
Indices ``i, \\ldots`` are ``D``-tuples for a ``D``-dimensional harmonic oscillator. 
The energy scale is defined by the first dimension i.e. ``\\hbar \\omega_x`` so that 
single particle energies are ``\\epsilon_i = (i_x + 1/2) + \\eta_y (i_y+1/2) + \\ldots``.
The factors ``\\eta_y, \\ldots`` allow for anisotropic trapping geometries and are assumed to 
be greater than `1` so that ``\\omega_x`` is the smallest trapping frequency.
Matrix elements ``V_{ijkl}`` are for a contact interaction calculated in this basis using 
first-order perturbation theory. All states with the same noninteracting energy are connected
by this interaction.

# Arguments

* `addr`: the starting address, defines number of particles and total number of modes.
* `S`: Tuple of the number of levels in each dimension, including the groundstate. The 
    allowed couplings between states is defined by the aspect ratio of `S .- 1`. Defaults 
    to a 1D spectrum with number of levels matching modes of `addr`. Will be sorted to make 
    the first dimension the largest.
* `η`: Define a custom aspect ratio for the trapping potential strengths, instead of deriving
    from `S .- 1`. This will only affect the single particle energy scale and not the 
    interactions. The values are always scaled relative to the first dimension, which sets 
    the energy scale of the system, ``\\hbar\\omega_x``.
* `g`: the (isotropic) interparticle interaction parameter. The value of `g` is assumed 
    to be in trap units.
* `interaction_only`: if set to `true` then the noninteracting single-particle terms are 
    ignored. Useful if only energy shifts due to interactions are required.

See also [`HOCartesianSeparable`](@ref).

# Warning: 
`num_offdiagonals` is a bad estimate for this Hamiltonian. Take care when building 
a matrix (use option `col_hint` with [`BasisSetRep`](@ref)) or using QMC methods.
"""
struct HOCartesian{
    D,  # number of dimensions
    A<:BoseFS,
    T
} <: AbstractHamiltonian{Float64}
    addr::A
    S::NTuple{D,Int64}
    aspect::NTuple{D,T}
    aspect1::NTuple{D,Float64}  # (possibly custom) ratios of trapping frequencies for single-particle energy scale
    energies::Vector{Float64}   # noninteracting single particle energies
    vtable::Array{Float64,4}    # interaction coefficients
    u::Float64
end

function HOCartesian(
        addr::BoseFS; 
        S = (num_modes(addr),),
        η = nothing, 
        g = 1.0,
        interaction_only = false
    )
    # x dimension defines the energy scale, and should be the smallest frequency i.e. largest spatial dimension
    D = length(S)
    S_sort = tuple(sort(collect(S); rev = true)...)
    S == S_sort || @warn("dimensions have been reordered")
    
    P = prod(S)
    P == num_modes(addr) || throw(ArgumentError("number of modes does not match starting address"))

    aspect = box_to_aspect(S)
    if all(denominator.(aspect) .== 1)
        aspect = Int.(aspect)
    end

    if isnothing(η)
        aspect1 = float.(aspect)
    elseif length(η) == D
        aspect1 = S == S_sort ? η : η[sortperm(collect(S); rev = true)]
    elseif length(η) == 1 && isreal(η) && η ≥ 1
        aspect1 = ntuple(d -> (d == 1 ? 1.0 : η), D)
    else
        throw(ArgumentError("Invalid aspect ratio parameter η."))
    end

    if interaction_only
        energies = zeros(P)
    else
        states = CartesianIndices(S)    # 1-indexed
        # energies = map(x -> dot(aspect, Tuple(x) .- 1/2), states)
        energies = reshape(map(x -> dot(aspect1, Tuple(x) .- 1/2), states), P)
    end

    u = sqrt(prod(aspect1)) * g / 2

    max_level = S_sort[1] - 1
    r = 0:max_level
    vmat = [four_oscillator_integral_general(i, j, k, l; max_level) for i in r, j in r, k in r, l in r]

    return HOCartesian{D,typeof(addr),eltype(aspect)}(addr, S_sort, aspect, aspect1, energies, vmat, u)
end

function Base.show(io::IO, h::HOCartesian)
    print(io, "HOCartesian($(h.addr); S=$(h.S), η=$(h.aspect1), u=$(h.u))")
end

function starting_address(h::HOCartesian)
    return h.addr
end

LOStructure(::Type{<:HOCartesian}) = IsHermitian()

# Base.getproperty(h::HOCartesian, s::Symbol) = getproperty(h, Val(s))
# Base.getproperty(h::HOCartesian, ::Val{:ks}) = getfield(h, :ks)
# Base.getproperty(h::HOCartesian, ::Val{:kes}) = getfield(h, :kes)
# Base.getproperty(h::HOCartesian, ::Val{:addr}) = getfield(h, :addr)
# Base.getproperty(h::HOCartesian, ::Val{:vtable}) = getfield(h, :vtable)
# Base.getproperty(h::HOCartesian{<:Any,<:Any,<:Any,<:Any,U}, ::Val{:u}) where {U} = U
# Base.getproperty(h::HOCartesian{<:Any,<:Any,<:Any,W}, ::Val{:w}) where {W} = W
# Base.getproperty(h::HOCartesian{<:Any,D}, ::Val{:dim}) where {D} = D


### DIAGONAL ELEMENTS ###
function energy_transfer_diagonal(h::HOCartesian{D}, omm::BoseOccupiedModeMap) where {D}
    result = 0.0
    states = CartesianIndices(h.S)    # 1-indexed
    
    for i in eachindex(omm)
        mode_i, occ_i = omm[i].mode, omm[i].occnum
        idx_i = Tuple(states[mode_i])
        if occ_i > 1
            # use i in 1:M indexing for accessing table
            val = occ_i * (occ_i - 1)
            result += prod(h.vtable[idx_i[d],idx_i[d],idx_i[d],idx_i[d]] for d in 1:D) * val
        end
        for j in 1:i-1
            mode_j, occ_j = omm[j].mode, omm[j].occnum
            idx_j = Tuple(states[mode_j])
            val = 4 * occ_i * occ_j
            result += prod(h.vtable[idx_i[d],idx_j[d],idx_i[d],idx_j[d]] for d in 1:D) * val
        end        
    end
    return result * h.u
end

noninteracting_energy(h::HOCartesian, omm::BoseOccupiedModeMap) = dot(h.energies, omm)
@inline function noninteracting_energy(h::HOCartesian, addr::BoseFS)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm)
end
# fast method for finding blocks
noninteracting_energy(h::HOCartesian, t::Union{Vector{Int64},NTuple{N,Int64}}) where {N} = sum(h.energies[j] for j in t)

@inline function diagonal_element(h::HOCartesian, addr::BoseFS)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm) + energy_transfer_diagonal(h, omm)
end

### OFFDIAGONAL ELEMENTS ###

# crude upper bound
# num_offdiagonals(h::HOCartesian, addr::BoseFS) = num_particles(addr) * prod(h.S)
num_offdiagonals(h::HOCartesian, addr::BoseFS) = dimension(h) - 1

"""
    HOCartOffdiagonals

Specialized [`AbstractOffdiagonals`](@ref) iterates over valid offdiagonal moves that
are connected by the interaction matrix.
"""
struct HOCartOffdiagonals{
    A<:BoseFS,T,H<:AbstractHamiltonian{T},P<:OccupiedPairsMap
}# <: AbstractOffdiagonals{A,T}
    ham::H
    addr::A
    pairs::P
    length::Int
end

function offdiagonals(h::HOCartesian, addr::BoseFS)
    num = num_offdiagonals(h, addr)
    pairs = OccupiedPairsMap(addr)
    return HOCartOffdiagonals(h, addr, pairs, num)
end

# This is the dumb way to loop through valid states.
# It should take arguments that define where to begin so that the loop can be restarted later 
# Better way would 'jump ahead' when a mode index goes outside the valid energy range.
# For that I would need dummy indices and probably while loops
function loop_over_modes(k_start, l_start, S, aspect, Emin, Emax, Etot)
    P = prod(S)
    for k in k_start:P
        E_k = mode_to_energy(k, S, aspect)
        Emin ≤ E_k ≤ Emax || continue
        l1 = k == k_start ? l_start : 1  # second loop should only restart in the middle until the first loop iterates
        for l in l1:k
            E_l = mode_to_energy(l, S, aspect)
            Emin ≤ E_l ≤ Emax || continue
            if E_l + E_k == Etot
                return k, l
            end
        end
    end
    return 0, 0     # fail case
end

function loop_over_particles(S, aspect, pairs, start)
    pair_index, k_start, l_start = start

    p_i, p_j = pairs[pair_index]

    mode_i = p_i.mode
    mode_j = p_j.mode
    mode_k = k_start
    mode_l = l_start
    while true
        Emin, Emax, Etot = find_Ebounds(mode_i, mode_j, S, aspect)      # this could be saved in the iteration state

        mode_k, mode_l = loop_over_modes(k_start, l_start, S, aspect, Emin, Emax, Etot)
        mode_k > 0 && mode_l > 0 && break   # no valid move found

        # move to next pair of particles
        if pair_index ≥ length(pairs)
            # end of iteration
            return pair_index, (mode_i, mode_j, mode_k, mode_l), true
        end
        pair_index += 1
        # update looping indices        
        p_i, p_j = pairs[pair_index]
        mode_i = p_i.mode
        mode_j = p_j.mode
        # start with checking lowest possible states in first dimension (x)
        k_start = 1
        l_start = 1
    end

    return pair_index, (mode_i, mode_j, mode_k, mode_l), false
end

function Base.iterate(off::HOCartOffdiagonals, iter_state = (1,1,1))
    S = off.ham.S
    aspect = off.ham.aspect
    addr = off.addr
    pairs = off.pairs
    length(off.pairs) == 0 && return nothing
    pair_index, modes, end_of_loop = loop_over_particles(S, aspect, pairs, iter_state)

    # end of iteration
    end_of_loop && return nothing

    i, j, k, l = modes
    new_iter_state = (pair_index, k, l + 1)  # if l + 1 is out of bounds then `loop_over_states` can handle it
    # check for swap and self moves but do not discard
    if (k,l) == (i,j) || (k,l) == (j,i)
        return (addr, 0.0), new_iter_state
    end
    # can I make a parity check here?
    p_i, p_j = pairs[pair_index]
    p_k = find_mode(addr, k)
    p_l = find_mode(addr, l)
    new_add, val = excitation(addr, (p_l, p_k), (p_j, p_i))

    if val ≠ 0.0
        states = CartesianIndices(S)    # 1-indexed
        ho_indices = ntuple(x -> Tuple(states[modes[x]]), 4)
        val *= prod(off.ham.vtable[a,b,c,d] for (a,b,c,d) in zip(ho_indices...))
        # account for j ≤ i and l ≤ k 
        val *= (1 + (i ≠ j)) * (1 + (k ≠ l)) * off.ham.u
    end

    return (new_add, val), new_iter_state
end

Base.IteratorSize(::HOCartOffdiagonals) = Base.SizeUnknown()
Base.eltype(::HOCartOffdiagonals{A,T}) where {A,T} = Tuple{A,T}

# function Base.getindex(s::HOCartOffdiagonals{A,T}, i)::Tuple{A,T} where {A,T}
#     @boundscheck begin
#         1 ≤ i ≤ s.length || throw(BoundsError(s, i))
#     end
#     new_address, matrix_element = get_offdiagonal(s.ham, s.addr, i, s.map)
#     return (new_address, matrix_element)
# end
# Base.size(s::HOCartOffdiagonals) = (s.length,)