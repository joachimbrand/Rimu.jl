# Counting HO modes
# slow but accurate method
function count_ho_modes_loop(S, aspect)
    P = prod(S)
    Emax = S[1]
    @assert Emax == maximum(S)  
    return count(
        map(1:P) do k
            E_k = mode_to_energy(k, S, aspect)
            E_k ≤ Emax
        end
    )
end

"""
    points_in_triangle(a, b)

    Given orthogonal side lengths `a` and `b` of a right triangle, returns 
    ``I+B`` where ``I`` is number of interior points and ``B`` is number 
    of points on the boundary.

    Pick's theorem:
    ```math
        A = I + B/2 - 1
    ```
    for a polygon with integer vertices and area ``A``.    
"""
function points_in_triangle(a, b)
    # double to ensure integer arithmetic
    double_area = a*b
    boundary = a + b + gcd(a, b)
    interior = (double_area + 2 - boundary) ÷ 2

    return interior + boundary
end

# faster methods - dispatch on dimension
count_ho_modes(S::NTuple{1,Int}, aspect) = S[1]

function count_ho_modes(S::NTuple{2,Int}, aspect)
    Mx, My = S
    ηy = aspect[2]
    My_upper = ceil(Int, Mx / ηy)

    return points_in_triangle(Mx, My_upper)
end

function count_ho_modes(S::NTuple{3,Int}, aspect)
    Mx, My, Mz = S .- 1
    ηy, ηz = aspect[2:3]
    
    result = 0
    for k in 0 : Mz
        Mx_upper = ceil(Int, Mx - k * ηz)
        My_upper = ceil(Int, Mx_upper / ηy)
        result += points_in_triangle(Mx_upper, My_upper)
    end
    return result
end

"""
    find_ho_mode(S, aspect, j, mode_in)

Loop over modes in grid defined by `S` and `aspect` until `j`th valid
mode is found. Skips over `mode_in`, which is the diagonal move
"""
function find_ho_mode(S, aspect, j, mode_in)
    P = prod(S)
    Emax = S[1]
    mode_j = 0
    for k in 1:P
        # skip self move
        k == mode_in && continue
        E_j = mode_to_energy(k, S, aspect)
        if E_j ≤ Emax 
            mode_j += 1
        end
        mode_j == j && return k
    end
    return 0     # fail case
end


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

# interaction
function ho_2brel_interaction(S, u, table, omm_i::OccupiedModeMap, omm_j::OccupiedModeMap)
    states = CartesianIndices(S)
    result = 0.0
    # there should only be one particle but the loop is kept for consistency
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
    return result * u
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

* `addr`: the starting address, defines number of particles and total number of modes.
* `S`: Tuple of the number of levels in each dimension, including the groundstate. The first 
    dimension should be the largest and sets the energy scale of the system, ``\\hbar\\omega_x``.
    The aspect ratios are determined from the other elements of `S`.
    Defaults to a 1D spectrum with number of levels matching modes of `addr`.
* `Sx`, `ηs`: Alternatively, provide the number of modes (including the groundstate) in the 
    first dimension `Sx` and `D-1` aspect ratios for the other dimensions `ηs`. The elements 
    of `ηs` should be at least 1 and can be `Float`s or `Rational`s. Providing `Sx` and `ηs` 
    will redefine `S = (Sx, Sy,...)`.
* `g`: the (isotropic) interparticle interaction parameter. The value of `g` is assumed 
    to be in trap units.
* `interaction_only`: if set to `true` then the noninteracting single-particle terms are 
    ignored. Useful if only energy shifts due to interactions are required.
"""
struct HOCartesian2BosonRelative{
    D,  # number of dimensions
    A<:BoseFS
} <: AbstractHamiltonian{Float64}
    addr::A
    S::NTuple{D,Int64}
    aspect::NTuple{D,Float64}
    num_offs::Int
    energies::Vector{Float64} # noninteracting single particle energies
    vtable::Vector{Float64}  # interaction coefficients
    u::Float64
end

function HOCartesian2BosonRelative(
        addr::BoseFS; # ignored for this Hamiltonian
        S = (num_modes(addr),),
        Sx = nothing,
        ηs = nothing, 
        g = 1.0,
        interaction_only = false
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
    addr = BoseFS(P, 1 => 1)
    # all addresses couple
    num_offs = count_ho_modes_loop(S, aspect) - 1

    if interaction_only
        energies = zeros(P)
    else
        states = CartesianIndices(S)    # 1-indexed
        energies = reshape(map(x -> dot(aspect, Tuple(x) .- 1/2), states), P)
    end

    # the aspect ratio appears from a change of variable when calculating the interaction integrals
    # this interaction is a one-body term
    u = g / sqrt(prod(aspect))

    M = maximum(S) - 1
    v_vec = [log_abs_oscillator_zero(i) for i in 0:2:M]

    return HOCartesian2BosonRelative{D,typeof(addr)}(addr, S, aspect, num_offs, energies, v_vec, u)
end

function Base.show(io::IO, h::HOCartesian2BosonRelative)
    flag = iszero(h.energies)
    # invert the scaling of u parameter
    g = h.u * sqrt(prod(h.aspect))
    print(io, "HOCartesian2BosonRelative($(h.addr); Sx=$(h.S[1]), η=$(h.aspect), g=$g, interaction_only=$flag)")
end

Base.:(==)(H::HOCartesian2BosonRelative, G::HOCartesian2BosonRelative) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(h::HOCartesian2BosonRelative) = h.addr

LOStructure(::Type{<:HOCartesian2BosonRelative}) = IsHermitian()

### DIAGONAL ELEMENTS ###
noninteracting_energy(h::HOCartesian2BosonRelative, omm::BoseOccupiedModeMap) = dot(h.energies, omm)
@inline function noninteracting_energy(h::HOCartesian2BosonRelative, addr)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm)
end
# fast method for finding blocks
noninteracting_energy(h::HOCartesian2BosonRelative, t::Union{Vector{Int64},NTuple{N,Int64}}) where {N} = sum(h.energies[j] for j in t)

@inline function diagonal_element(h::HOCartesian2BosonRelative, addr)
    omm = OccupiedModeMap(addr)
    return noninteracting_energy(h, omm) + ho_2brel_interaction(h.S, h.u, h.vtable, omm, omm)
end

### OFFDIAGONAL ELEMENTS ###
num_offdiagonals(h::HOCartesian2BosonRelative, addr) = h.num_offs

function get_offdiagonal(
        h::HOCartesian2BosonRelative, 
        addr, 
        chosen::Int, 
        omm::OccupiedModeMap = OccupiedModeMap(addr)
    )
    p_i = omm[1]
    mode_j = find_ho_mode(h.S, h.aspect, chosen, p_i.mode)
    # no move found:
    mode_j == 0 && return addr, 0.0

    p_j = find_mode(addr, mode_j)
    new_addr, val = excitation(addr, (p_j,), (p_i,))

    interaction = ho_2brel_interaction(h.S, h.u, h.vtable, omm, OccupiedModeMap(new_addr))

    return new_addr, val * interaction
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

function offdiagonals(h::HOCartesian2BosonRelative, addr)
    omm = OccupiedModeMap(addr)
    num = num_offdiagonals(h, addr)
    return HOCart2bRelOffdiagonals(h, addr, num, omm)
end

function Base.getindex(s::HOCart2bRelOffdiagonals{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i, s.map)
    return (new_address, matrix_element)
end

Base.size(s::HOCart2bRelOffdiagonals) = (s.length,)
