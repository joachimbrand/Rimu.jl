"""
    log_abs_oscillator_zero(n)

    Compute the logarithm of the absolute value of the ``n^\\mathrm{th}`` 1D 
    harmonic oscillator function evaluated at the origin. The overall sign is
    determined when the matrix element is evaluated.
"""
function log_abs_oscillator_zero(n)
    isodd(n) && return -Inf, 1.0
    x, _ = SpecialFunctions.logabsgamma((1-n)/2)
    y, _ = SpecialFunctions.logabsgamma(n+1)
    result = log(pi)/4 + (n/2)*log(2) - x - y/2
    return result
end

"""
    ho_2brel_interaction(S, table, omm_i::OccupiedModeMap, omm_j::OccupiedModeMap)

Returns the product of two one-dimensional harmonic oscillator functions 
evaluated at the origin, 
```math
    v_{ij} = \\phi_i(0) \\phi_j(0)
```
Indices `i,j` start at `0` for the groundstate, must be even, and are bounded by
the entries of `S` which defines the Cartesian HO modes.
The values ``\\phi_i(0)`` are precomputed by [`HOCartesian2BosonRelative`](@ref) 
and passed in as the vector `table`.
The result is summed over all occupied modes in `omm_i` and `omm_j`.
"""
function ho_2brel_interaction(S, table, omm_i::OccupiedModeMap, omm_j::OccupiedModeMap)
    states = CartesianIndices(S)
    result = 0.0
    # there should only be one particle but the sum is kept for consistency
    for p_i in omm_i, p_j in omm_j
        occ_i, mode_i = p_i
        occ_j, mode_j = p_j
        ho_indices = (Tuple(states[mode_i])..., Tuple(states[mode_j])...) .- 1
        if all(iseven.(ho_indices))
            doubly_evens = count(iseven.(ho_indices .÷ 2))
            sign = (-1)^doubly_evens
            result += sign * exp(sum(table[k÷2 + 1] for k in ho_indices)) * occ_i * occ_j
        end
    end
    return result
end

"""
    HOCartesian2BosonRelative(addr; S, Sx, ηs, g = 1.0, interaction_only = false)

Implements the relative Hamiltonian of two bosons in a harmonic oscillator in Cartesian basis 
with contact interactions 
```math
\\hat{H} = \\sum_\\mathbf{i} ϵ_\\mathbf{i} + g\\sum_\\mathbf{ij} V_\\mathbf{ij} a^†_\\mathbf{i} a_\\mathbf{i},
```
For a ``D``-dimensional harmonic oscillator indices ``\\mathbf{i}, \\mathbf{j}, \\ldots`` 
are ``D``-tuples. The energy scale is defined by the first dimension i.e. ``\\hbar \\omega_x`` 
so that single particle energies are 
```math
    \\frac{\\epsilon_\\mathbf{i}}{\\hbar \\omega_x} = (i_x + 1/2) + \\eta_y (i_y+1/2) + \\ldots.
```
The factors ``\\eta_y, \\ldots`` allow for anisotropic trapping geometries and are assumed to 
be greater than `1` so that ``\\omega_x`` is the smallest trapping frequency.

Matrix elements ``V_{\\mathbf{ij}}`` are for a contact interaction calculated in this basis
```math
    V_{\\mathbf{ij}} = \\prod_{d \\in x, y,\\ldots} \\psi_{i_d}(0) \\psi_{j_d}(0).
```

# Arguments

* `addr`: **not used** the starting address is required for the [`Hamiltonians`](@ref) interface,
    but is overwritten for this Hamiltonians. A warning will be issued if the input address has the wrong
    number of particles (1) or modes (`prod(S)`). This warning can be disabled with keyword
    argument `warn_address = false`.
* `S`: Tuple of the number of levels in each dimension, including the groundstate. The first 
    dimension should be the largest and sets the energy scale of the system, ``\\hbar\\omega_x``.
    The aspect ratios are determined from the other elements of `S`.
    Defaults to a 1D spectrum with number of levels matching modes of `addr`.
* `Sx`, `ηs`: Alternatively, provide the number of modes (including the groundstate) in the 
    first dimension `Sx` and `D-1` aspect ratios for the other dimensions `ηs = (η_y,...)`. The elements 
    of `ηs` should be at least `1.0`. Providing `Sx` and `ηs` will redefine `S = (Sx, Sy,...)`.
* `g`: the (isotropic) interparticle interaction parameter. The value of `g` is assumed 
    to be in trap units.
* `interaction_only`: if set to `true` then the noninteracting single-particle terms are 
    ignored. Useful if only energy shifts due to interactions are required.

See also [`HOCartesianEnergyConserved`](@ref) and[`HOCartesianEnergyConservedPerDim`](@ref).
"""
struct HOCartesian2BosonRelative{
    D,  # number of dimensions
    A<:BoseFS
} <: AbstractHamiltonian{Float64}
    addr::A
    S::NTuple{D,Int64}
    aspect::NTuple{D,Float64}
    vtable::Vector{Float64}  # interaction coefficients
    g::Float64
    interaction_only::Bool
end

function HOCartesian2BosonRelative(
        addr; # ignored for this Hamiltonian
        S = (num_modes(addr),),
        Sx = nothing,
        ηs = nothing, 
        g = 1.0,
        interaction_only = false,
        warn_address = true
    )
    if isnothing(ηs) && isnothing(Sx)
        S[1] ≠ maximum(S) && throw(ArgumentError("Aspect ratios must be greater than 1.0"))
        aspect = float(box_to_aspect(S))
        D = length(S)
    else
        any(ηs .< 1.0) && throw(ArgumentError("Aspect ratios must be greater than 1.0"))
        D = length(ηs) + 1
        Srest = map(x -> floor(Int, (Sx - 1) / x + 1), ηs)
        S = (Sx, Srest...)
        aspect = (1.0, float.(ηs)...)
    end
    P = prod(S)

    # address type is fixed so it is overwritten here to ensure it works
    warn_address && !(addr isa BoseFS{1,P}) && @warn "Input address is being overwritten"
    addr = BoseFS(P, 1 => 1)

    M = maximum(S) - 1
    v_vec = [log_abs_oscillator_zero(i) for i in 0:2:M]

    return HOCartesian2BosonRelative{D,typeof(addr)}(addr, S, aspect, v_vec, g, interaction_only)
end

function Base.show(io::IO, H::HOCartesian2BosonRelative)
    if length(H.S) == 1
        print(io, "HOCartesian2BosonRelative($(H.addr); S=$(H.S), g=$(H.g), interaction_only=$(H.interaction_only))")
    else
        print(io, "HOCartesian2BosonRelative($(H.addr); Sx=$(H.S[1]), ηs=$(H.aspect[2:end]), g=$(H.g), interaction_only=$(H.interaction_only))")
    end
end

Base.:(==)(H::HOCartesian2BosonRelative, G::HOCartesian2BosonRelative) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(H::HOCartesian2BosonRelative) = H.addr

LOStructure(::Type{<:HOCartesian2BosonRelative}) = IsHermitian()

### DIAGONAL ELEMENTS ###
@inline function noninteracting_energy(S, aspect, omm::BoseOccupiedModeMap)
    states = CartesianIndices(S)
    return sum(omm) do p
        indices = Tuple(states[p.mode]) .- 1
        dot(indices, aspect) + sum(aspect)/2
    end
end
noninteracting_energy(H::HOCartesian2BosonRelative, addr) = noninteracting_energy(H.S, H.aspect, OccupiedModeMap(addr))

@inline function diagonal_element(H::HOCartesian2BosonRelative, addr)
    omm = OccupiedModeMap(addr)
    u = H.g / sqrt(sum(H.aspect))
    result = u * ho_2brel_interaction(H.S, H.vtable, omm, omm)
    if !H.interaction_only
        result += noninteracting_energy(H, addr)
    end
    return result
end

### OFFDIAGONAL ELEMENTS ###
# all addresses couple
num_offdiagonals(H::HOCartesian2BosonRelative, addr) = prod(H.S) - 1

function get_offdiagonal(
        H::HOCartesian2BosonRelative, 
        addr, 
        chosen::Int, 
        omm::OccupiedModeMap = OccupiedModeMap(addr)
    )
    P = prod(H.S)
    chosen ≥ P - 1 && return addr, 0.0

    # skip self-move
    p_i = omm[1]
    if chosen ≥ p_i.mode
        chosen += 2     # conserve parity
    end

    p_j = find_mode(addr, chosen)
    new_addr, val = excitation(addr, (p_j,), (p_i,))

    u = H.g / sqrt(sum(H.aspect))
    interaction = ho_2brel_interaction(H.S, H.vtable, omm, OccupiedModeMap(new_addr))

    return new_addr, val * interaction * u
end

###
### offdiagonals
###
"""
    HOCart2bRelOffdiagonals
"""
struct HOCart2bRelOffdiagonals{
    A<:BoseFS,T,H<:AbstractHamiltonian{T},O<:OccupiedModeMap
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    map::O
end

function offdiagonals(H::HOCartesian2BosonRelative, addr)
    omm = OccupiedModeMap(addr)
    num = num_offdiagonals(H, addr)
    return HOCart2bRelOffdiagonals(H, addr, num, omm)
end

function Base.getindex(s::HOCart2bRelOffdiagonals{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i, s.map)
    return (new_address, matrix_element)
end

Base.size(s::HOCart2bRelOffdiagonals) = (s.length,)
