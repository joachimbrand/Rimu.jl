"""
    log_abs_oscillator_zero(n)

Compute the logarithm of the absolute value of the ``n^\\mathrm{th}`` 1D 
harmonic oscillator function evaluated at the origin. The overall sign is
determined when the matrix element is evaluated in [`ho_delta_potential`](@ref).
"""
function log_abs_oscillator_zero(n)
    isodd(n) && return -Inf, 1.0
    x, _ = SpecialFunctions.logabsgamma((1-n)/2)
    y, _ = SpecialFunctions.logabsgamma(n+1)
    result = log(pi)/4 + (n/2)*log(2) - x - y/2
    return result
end

"""
    ho_delta_potential(S, omm_i::OccupiedModeMap, omm_j::OccupiedModeMap; vals)

Returns the product of two one-dimensional harmonic oscillator functions 
evaluated at the origin, 
```math
    v_{ij} = \\phi_i(0) \\phi_j(0)
```
Indices `i,j` start at `0` for the groundstate, must be even, and are bounded by
the entries of `S` which defines the Cartesian HO modes.
The values ``\\phi_i(0)`` are precomputed by [`HOCartesianCentralImpurity`](@ref) 
and passed in as the vector `vals`.
The result is summed over all occupied modes in `omm_i` and `omm_j`.
"""
function ho_delta_potential(S, 
    omm_i::OccupiedModeMap, 
    omm_j::OccupiedModeMap; 
    vals = [log_abs_oscillator_zero(i) for i in 0:2:(maximum(S)-1)]
    )
    states = CartesianIndices(S)
    result = 0.0
    for p_i in omm_i, p_j in omm_j
        occ_i, mode_i = p_i
        occ_j, mode_j = p_j
        ho_indices = (Tuple(states[mode_i])..., Tuple(states[mode_j])...) .- 1
        if all(iseven.(ho_indices))
            doubly_evens = count(iseven.(ho_indices .÷ 2))
            sign = (-1)^doubly_evens
            result += sign * exp(sum(vals[k÷2 + 1] for k in ho_indices)) * occ_i * occ_j
        end
    end
    return result
end

"""
    HOCartesianCentralImpurity(addr; kwargs...)

Hamiltonian of non-interacting particles in an arbitrary harmonic trap with a delta-function 
potential at the centre, with strength `g`,
```math
\\hat{H}_\\mathrm{rel} = \\sum_\\mathbf{i} ϵ_\\mathbf{i} n_\\mathbf{i} 
    + g\\sum_\\mathbf{ij} V_\\mathbf{ij} a^†_\\mathbf{i} a_\\mathbf{j}.
```
For a ``D``-dimensional harmonic oscillator indices ``\\mathbf{i}, \\mathbf{j}, \\ldots`` 
are ``D``-tuples. The energy scale is defined by the first dimension i.e. ``\\hbar \\omega_x`` 
so that single particle energies are 
```math
    \\frac{\\epsilon_\\mathbf{i}}{\\hbar \\omega_x} = (i_x + 1/2) + \\eta_y (i_y+1/2) + \\ldots.
```
The factors ``\\eta_y, \\ldots`` allow for anisotropic trapping geometries and are assumed to 
be greater than `1` so that ``\\omega_x`` is the smallest trapping frequency.

Matrix elements ``V_{\\mathbf{ij}}`` are for a delta function potential calculated in this basis
```math
    V_{\\mathbf{ij}} = \\prod_{d \\in x, y,\\ldots} \\psi_{i_d}(0) \\psi_{j_d}(0).
```
Only even parity states feel this impurity, so all ``i_d`` are even. Note that the matrix 
representation of this Hamiltonian is completely dense in the even-parity subspace.

# Arguments

* `addr`: the starting address, defines number of particles and total number of modes.
* `S`: Tuple of the number of levels in each dimension, including the groundstate. The first 
    dimension should be the largest and sets the energy scale of the system, ``\\hbar\\omega_x``.
    The aspect ratios are determined from the other elements of `S`.
    Defaults to a 1D spectrum with number of levels matching modes of `addr`.
* `M`, `ηs`: Alternatively, provide the maximum mode number in the 
    first dimension `Mx` and `D-1` aspect ratios for the other dimensions `ηs = (η_y, ...)`. 
    The elements of `ηs` should be at least `1.0`. Providing `Mx` and `ηs` will redefine 
    `S = (M + 1, M/η_y + 1, ...)`.
* `g`: the strength of the delta impurity in (``x``-dimension) trap units.
* `impurity_only`: if set to `true` then the trap energy terms are ignored. Useful if 
    only energy shifts due to the impurity are required.

See also [`HOCartesianContactInteractions`](@ref) and[`HOCartesianEnergyConservedPerDim`](@ref).
"""
struct HOCartesianCentralImpurity{
    D,  # number of dimensions
    A
} <: AbstractHamiltonian{Float64}
    addr::A
    S::NTuple{D,Int64}
    aspect::NTuple{D,Float64}
    vtable::Vector{Float64}  # interaction coefficients
    g::Float64
    impurity_only::Bool
end

function HOCartesianCentralImpurity(
        addr::SingleComponentFockAddress; 
        S = (num_modes(addr),),
        M = nothing,
        ηs = nothing, 
        g = 1.0,
        impurity_only = false
    )
    if isnothing(ηs) && isnothing(Sx)
        M = S[1] - 1
        M + 1 ≠ maximum(S) && throw(ArgumentError("First dimension must have the most states"))
        aspect = float.(box_to_aspect(S))
        D = length(S)        
    else
        any(ηs .< 1.0) && throw(ArgumentError("Aspect ratios must be greater than 1.0"))
        D = length(ηs) + 1
        Srest = map(x -> floor(Int, M / x + 1), ηs)
        S = (M + 1, Srest...)
        aspect = (1.0, float.(ηs)...)
    end
    num_modes(addr) == prod(S) || throw(ArgumentError("number of modes does not match starting address"))

    v_vec = [log_abs_oscillator_zero(i) for i in 0:2:M]

    return HOCartesianCentralImpurity{D,typeof(addr)}(addr, S, aspect, v_vec, g, impurity_only)
end

function Base.show(io::IO, H::HOCartesianCentralImpurity)
    if length(H.S) == 1
        print(io, "HOCartesianCentralImpurity($(H.addr); S=$(H.S), g=$(H.g), impurity_only=$(H.impurity_only))")
    else
        print(io, "HOCartesianCentralImpurity($(H.addr); M=$(H.S[1]-1), ηs=$(H.aspect[2:end]), g=$(H.g), impurity_only=$(H.impurity_only))")
    end
end

Base.:(==)(H::HOCartesianCentralImpurity, G::HOCartesianCentralImpurity) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(H::HOCartesianCentralImpurity) = H.addr

LOStructure(::Type{<:HOCartesianCentralImpurity}) = IsHermitian()

### DIAGONAL ELEMENTS ###
@inline function noninteracting_energy(S, aspect, omm::BoseOccupiedModeMap)
    states = CartesianIndices(S)
    return sum(omm) do p
        indices = Tuple(states[p.mode]) .- 1
        dot(indices, aspect) + sum(aspect)/2
    end
end
noninteracting_energy(H::HOCartesianCentralImpurity, addr) = noninteracting_energy(H.S, H.aspect, OccupiedModeMap(addr))

@inline function diagonal_element(H::HOCartesianCentralImpurity, addr)
    omm = OccupiedModeMap(addr)
    u = H.g / sqrt(prod(H.aspect))
    result = u * ho_delta_potential(H.S, omm, omm; vals = H.vtable)
    if !H.impurity_only
        result += noninteracting_energy(H, addr)
    end
    return result
end

### OFFDIAGONAL ELEMENTS ###
# all addresses couple
num_offdiagonals(H::HOCartesianCentralImpurity, addr) = prod(H.S) - 1

function get_offdiagonal(
        H::HOCartesianCentralImpurity, 
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

    u = H.g / sqrt(prod(H.aspect))
    impurity = ho_delta_potential(H.S, omm, OccupiedModeMap(new_addr); vals = H.vtable)

    return new_addr, val * impurity * u
end

###
### offdiagonals
###
"""
    HOCart2bRelOffdiagonals

Specialized [`AbstractOffdiagonals`](@ref) for [`HOCartesianCentralImpurity`](@ref).
"""
struct HOCart2bRelOffdiagonals{
    A<:BoseFS,T,H<:AbstractHamiltonian{T},O<:OccupiedModeMap
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    map::O
end

function offdiagonals(H::HOCartesianCentralImpurity, addr)
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
