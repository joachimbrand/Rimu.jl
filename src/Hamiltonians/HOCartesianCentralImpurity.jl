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
    ho_delta_potential(S, i, j; [vals])

Returns the matrix element of a delta potential at the centre of a trap, i.e.
the  product of two harmonic oscillator functions evaluated at the origin,
```math
    v_{ij} = \\phi_{\\mathbf{n}_i}(0) \\phi_{\\mathbf{n}_j}(0)
```
which is only non-zero for even-parity states. The `i`th single particle state
corresponds to a ``D``-tuple of harmonic oscillator indices ``\\mathbf{n}_i``.
`S` defines the bounds of Cartesian harmonic oscillator indices for each dimension.
The optional keyword argument `vals` allows passing pre-computed values of
``\\phi_i(0)`` to speed-up the calculation. The values can be calculated with
[`log_abs_oscillator_zero`](@ref).

See also [`HOCartesianCentralImpurity`](@ref).
"""
function ho_delta_potential(S, i, j;
    vals = [log_abs_oscillator_zero(k) for k in 0:2:(maximum(S)-1)]
    )
    states = CartesianIndices(S)
    ho_indices = (Tuple(states[i])..., Tuple(states[j])...) .- 1
    if all(iseven.(ho_indices))
        doubly_evens = count(iseven.(ho_indices .÷ 2))
        sign = (-1)^doubly_evens
        return sign * exp(sum(vals[k÷2 + 1] for k in ho_indices))
    else
        return 0.0
    end
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
representation of this Hamiltonian for a single particle is completely dense in the even-parity
subspace.

# Arguments

* `addr`: the starting address, defines number of particles and total number of modes.
* `max_nx = num_modes(addr) - 1`: the maximum harmonic oscillator index number in the ``x``-dimension.
    Must be even. Index number for the harmonic oscillator groundstate is `0`.
* `ηs = ()`: a tuple of aspect ratios for the remaining dimensions `(η_y, ...)`. Should be empty
    for a 1D trap or contain values greater than `1.0`. The maximum index
    in other dimensions will be the largest even number less than `M/η_y`.
* `S = nothing`: Instead of `max_nx`, manually set the number of levels in each dimension,
    including the groundstate. Must be a `Tuple` of `Int`s.
* `g = 1.0`: the strength of the delta impurity in (``x``-dimension) trap units.
* `impurity_only=false`: if set to `true` then the trap energy terms are ignored. Useful if
    only energy shifts due to the impurity are required.

!!! warning
        Due to use of `SpecialFunctions` with large arguments the matrix representation of
        this Hamiltonian may not be strictly symmetric, but is approximately symmetric within
        machine precision.

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
        max_nx::Int = num_modes(addr) - 1,
        S = nothing,
        ηs = (),
        g = 1.0,
        impurity_only = false
    )
    any(ηs .< 1.0) && throw(ArgumentError("Aspect ratios must be greater than 1.0"))
    if isnothing(S)
        iseven(max_nx) || throw(ArgumentError("max_nx must be even"))
        D = length(ηs) + 1
        Srest = map(x -> 2floor(Int, max_nx÷2 / x) + 1, ηs)
        S = (max_nx + 1, Srest...)
    else
        D = length(S)
    end
    aspect = (1.0, float.(ηs)...)

    num_modes(addr) == prod(S) || throw(ArgumentError("number of modes does not match starting address"))

    v_vec = [log_abs_oscillator_zero(i) for i in 0:2:S[1]-1]

    return HOCartesianCentralImpurity{D,typeof(addr)}(addr, S, aspect, v_vec, g, impurity_only)
end

function Base.show(io::IO, H::HOCartesianCentralImpurity)
    compact_addr = repr(H.addr, context=:compact => true) # compact print address
    print(io, "HOCartesianCentralImpurity($(compact_addr); max_nx=$(H.S[1]-1), ηs=$(H.aspect[2:end]), g=$(H.g), impurity_only=$(H.impurity_only))")
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
    result = u * sum(p -> ho_delta_potential(H.S, p.mode, p.mode; vals = H.vtable), omm)
    if !H.impurity_only
        result += noninteracting_energy(H, addr)
    end
    return result
end

### OFFDIAGONAL ELEMENTS ###
get_offdiagonal(H::HOCartesianCentralImpurity, addr, i) = offdiagonals(H, addr)[i]
num_offdiagonals(H::HOCartesianCentralImpurity, addr) = (num_modes(addr) - 1) * length(occupied_modes(addr))

###
### offdiagonals
###
"""
    HOCartImpurityOffdiagonals

Specialized [`AbstractOffdiagonals`](@ref) for [`HOCartesianCentralImpurity`](@ref).
"""
struct HOCartImpurityOffdiagonals{
    A<:BoseFS,O<:OccupiedModeMap,H<:HOCartesianCentralImpurity
} <: AbstractOffdiagonals{A,Float64}
    ham::H
    address::A
    omm::O
    u::Float64
    length::Int
end

function offdiagonals(H::HOCartesianCentralImpurity, addr::SingleComponentFockAddress)
    omm = OccupiedModeMap(addr)
    num_offs = num_offdiagonals(H, addr)
    u = H.g / sqrt(prod(H.aspect))
    return HOCartImpurityOffdiagonals(H, addr, omm, u, num_offs)
end

function Base.getindex(offs::HOCartImpurityOffdiagonals, chosen)
    addr = offs.address
    omm = offs.omm
    S = offs.ham.S
    vals = offs.ham.vtable
    u = offs.u

    P = num_modes(addr)
    i, j = fldmod1(chosen, P - 1)
    index_i = omm[i]
    j += j ≥ index_i.mode   # skip self move
    index_j = find_mode(addr, j)
    new_addr, val = excitation(addr, (index_j,), (index_i,))

    impurity = ho_delta_potential(S, index_i.mode, index_j.mode; vals)

    return new_addr, val * impurity * u
end

Base.size(s::HOCartImpurityOffdiagonals) = (s.length,)
