# TO-DO: add geometry for higher dimensions.
"""
    G2RealCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Two-body operator for density-density correlation between sites separated by `d`
with `0 ≤ d < M`.
```math
    \\hat{G}^{(2)}(d) = \\frac{1}{M} \\sum_i^M \\hat{n}_i (\\hat{n}_{i+d} - \\delta_{0d}).
```
Assumes a one-dimensional lattice with periodic boundary conditions where
```math
    \\hat{G}^{(2)}(-M/2 \\leq d < 0) = \\hat{G}^{(2)}(|d|),
```
```math
    \\hat{G}^{(2)}(M/2 < d < M) = \\hat{G}^{(2)}(M - d),
```
and normalisation
```math
    \\sum_{d=0}^{M-1} \\langle \\hat{G}^{(2)}(d) \\rangle = \\frac{N (N-1)}{M}.
```

For multicomponent basis, calculates correlations between all particles equally, 
equivalent to stacking all components into a single Fock state.

# Arguments
- `d::Integer`: distance between sites.

# See also

* [`HubbardReal1D`](@ref)
* [`G2MomCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2RealCorrelator{D} <: AbstractHamiltonian{Float64}
end

G2RealCorrelator(d::Int) = G2RealCorrelator{d}()

function Base.show(io::IO, ::G2RealCorrelator{D}) where {D}
    print(io, "G2RealCorrelator($D)")
end

LOStructure(::Type{<:G2RealCorrelator}) = IsDiagonal()

function diagonal_element(::G2RealCorrelator{0}, add::SingleComponentFockAddress)
    M = num_modes(add)
    v = onr(add)
    return dot(v, v .- 1) / M
end
function diagonal_element(::G2RealCorrelator{D}, add::SingleComponentFockAddress) where {D}
    M = num_modes(add)
    d = mod(D, M)
    v = onr(add)
    result = 0
    for i in eachindex(v)
        result += v[i] * v[mod1(i + d, M)]
    end
    return result / M
end

function diagonal_element(::G2RealCorrelator{0}, add::CompositeFS)
    M = num_modes(add)
    v = sum(map(onr, add.components))
    return dot(v, v .- 1) / M
end
function diagonal_element(::G2RealCorrelator{D}, add::CompositeFS) where {D}
    M = num_modes(add)
    d = mod(D, M)
    v = sum(map(onr, add.components))
    result = 0
    for i in eachindex(v)
        result += v[i] * v[mod1(i + d, M)]
    end
    return result / M
end

num_offdiagonals(::G2RealCorrelator, ::SingleComponentFockAddress) = 0
num_offdiagonals(::G2RealCorrelator, ::CompositeFS) = 0

# not needed:
# get_offdiagonal(::G2RealCorrelator, add)
# starting_address(::G2RealCorrelator)

# Methods that need to be implemented:
#   •  starting_address(::AbstractHamiltonian) - not needed
"""
    G2MomCorrelator(d::Int,c=:cross) <: AbstractHamiltonian{ComplexF64}

Two-body correlation operator representing the density-density
correlation at distance `d` of a two component system
in a momentum-space Fock-state basis.
It returns a `Complex` value.

Correlation across two components:
```math
\\hat{G}^{(2)}(d) = \\frac{1}{M}\\sum_{spqr=1}^M e^{-id(p-q)2π/M} a^†_{s} b^†_{p}  b_q a_r δ_{s+p,q+r}
```
Correlation within a single component:
```math
\\hat{G}^{(2)}(d) = \\frac{1}{M}\\sum_{spqr=1}^M e^{-id(p-q)2π/M} a^†_{s} a^†_{p}  a_q a_r δ_{s+p,q+r}
```

The diagonal element, where `(p-q)=0`, is
```math
\\frac{1}{M}\\sum_{k,p=1}^M a^†_{k} b^†_{p}  b_p a_k .
```

# Arguments
- `d::Integer`: the distance between two particles.
- `c`: possible instructions: `:cross`: default instruction, computing correlation between particles across two components;
  `:first`: computing correlation between particles within the first component;
  `:second`: computing correlation between particles within the second component.
  These are the only defined instructions, using anything else will produce errors.

# To use on a one-component system

For a system with only one component, e.g. with `BoseFS`, the second argument `c` is irrelevant
and can be any of the above instructions, one could simply skip this argument and let it be the default value.

# See also

* [`BoseHubbardMom1D2C`](@ref)
* [`BoseFS2C`](@ref)
* [`G2RealCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2MomCorrelator{C} <: AbstractHamiltonian{ComplexF64}
    d::Int

    function G2MomCorrelator(d,c=:cross)
        if c == :first
            return new{1}(d)
        elseif c == :second
            return new{2}(d)
        elseif c == :cross
            return new{3}(d)
        else
            throw(ArgumentError("Unknown instruction for G2MomCorrelator!"))
        end
    end
end
@deprecate G2Correlator(d,c) G2MomCorrelator(d,c)
@deprecate G2Correlator(d) G2MomCorrelator(d)

function Base.show(io::IO, g::G2MomCorrelator{C}) where C
    if C == 1
        print(io, "G2MomCorrelator($(g.d),:first)")
    elseif C == 2
        print(io, "G2MomCorrelator($(g.d),:second)")
    elseif C == 3
        print(io, "G2MomCorrelator($(g.d),:cross)")
    end
end


num_offdiagonals(g::G2MomCorrelator{1},add::BoseFS2C) = num_offdiagonals(g, add.bsa)
num_offdiagonals(g::G2MomCorrelator{2},add::BoseFS2C) = num_offdiagonals(g, add.bsb)

function num_offdiagonals(g::G2MomCorrelator{3}, add::BoseFS2C)
    m = num_modes(add)
    sa = num_occupied_modes(add.bsa)
    sb = num_occupied_modes(add.bsb)
    return sa*(m-1)*sb
    # number of excitations that can be made
end

function num_offdiagonals(g::G2MomCorrelator, add::BoseFS)
    m = num_modes(add)
    singlies, doublies = num_singly_doubly_occupied_sites(add)
    return singlies*(singlies-1)*(m - 2) + doublies*(m - 1)
end

diagonal_element(g::G2MomCorrelator{1}, add::BoseFS2C) = diagonal_element(g, add.bsa)
diagonal_element(g::G2MomCorrelator{2}, add::BoseFS2C) = diagonal_element(g, add.bsb)

function diagonal_element(g::G2MomCorrelator{3}, add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    onrep_a = onr(add.bsa)
    onrep_b = onr(add.bsb)
    gd = 0
    for p in 1:M
        iszero(onrep_b[p]) && continue
        for k in 1:M
            gd += onrep_a[k]*onrep_b[p] # b†_p b_p a†_k a_k
        end
    end
    return ComplexF64(gd/M)
end

function diagonal_element(g::G2MomCorrelator, add::BoseFS{N,M,A}) where {N,M,A}
    onrep = onr(add)
    gd = 0
    for p in 1:M
        iszero(onrep[p]) && continue
        for k in 1:M
            gd += onrep[k]*onrep[p] # a†_p a_p a†_k a_k
        end
    end
    return ComplexF64(gd/M)
end

function get_offdiagonal(g::G2MomCorrelator{1}, add::A, chosen)::Tuple{A,ComplexF64} where A<:BoseFS2C
    new_bsa, elem = get_offdiagonal(g, add.bsa, chosen)
    return A(new_bsa,add.bsb), elem
end

function get_offdiagonal(g::G2MomCorrelator{2}, add::A, chosen)::Tuple{A,ComplexF64} where A<:BoseFS2C
    new_bsb, elem = get_offdiagonal(g, add.bsb, chosen)
    return A(add.bsa, new_bsb), elem
end

function get_offdiagonal(
    g::G2MomCorrelator{3},
    add::A,
    chosen,
    ma=OccupiedModeMap(add.bsa),
    mb=OccupiedModeMap(add.bsb),
)::Tuple{A,ComplexF64} where {A<:BoseFS2C}

    m = num_modes(add)
    new_bsa, new_bsb, gamma, _, _, Δp = momentum_transfer_excitation(
        add.bsa, add.bsb, chosen, ma, mb
    )
    gd = exp(-im*g.d*Δp*2π/m)*gamma
    return A(new_bsa, new_bsb), ComplexF64(gd/m)
end

function get_offdiagonal(
    g::G2MomCorrelator,
    add::A,
    chosen,
)::Tuple{A,ComplexF64} where {A<:BoseFS}
    M = num_modes(add)
    new_add, gamma, Δp = momentum_transfer_excitation(add, chosen, OccupiedModeMap(add))
    gd = exp(-im*g.d*Δp*2π/M)*gamma
    return new_add, ComplexF64(gd/M)
end

"""
    SuperfluidCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Two-body operator for superfluid correlation between sites separated by `d` with `0 ≤ d < m`

```math
    \\hat{C}_{\\text{SF}}(d) = \\frac{1}{M} \\sum_{i}^{M} a_{i}^{\\dagger} a_{i + d}
```
Assumes a one-dimensional lattice with periodic boundary conditions.

SuperfluidCorrelator must be wrapped in 'AllOverlaps'

# Arguments
- `d::Integer`: distance between sites.

# Implementation
Follow below for an example on how 'SuperfluidCorrelator' is used. Implementation is similar to that of 'G2RealCorrelator'
- [G2 Correlator Example](https://joachimbrand.github.io/Rimu.jl/previews/PR227/generated/G2-example.html)


# See also 

* [`HubbardReal1D`](@ref)
* [`G2RealCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)

"""
struct SuperfluidCorrelator{D} <: AbstractHamiltonian{Float64}
end

SuperfluidCorrelator(d::Int) = SuperfluidCorrelator{d}()

function Base.show(io::IO, ::SuperfluidCorrelator{D}) where {D}
    print(io, "SuperfluidCorrelator($D)")
end

function num_offdiagonals(::SuperfluidCorrelator, add::SingleComponentFockAddress)
    return num_occupied_modes(add)
end

function get_offdiagonal(::SuperfluidCorrelator{D}, add::SingleComponentFockAddress, chosen) where {D}
    src = find_occupied_mode(add, chosen)
    dst = find_mode(add, mod1(src.mode + D, num_modes(add)))
    address, value = excitation(add, (dst,), (src,))
    return address, value / num_modes(add)
end

function diagonal_element(::SuperfluidCorrelator{0}, add::SingleComponentFockAddress)
    return num_particles(add) / num_modes(add)
end
function diagonal_element(::SuperfluidCorrelator{D}, add::SingleComponentFockAddress) where {D}
    return 0.0
end


"""
    StringCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Two-body operator for string correlation between sites separated by `d` with `0 ≤ d < m`

```math
    \\hat{C}_{\\text{string}}(d) = \\frac{1}{M} \\sum_{i}^{M} \\delta n_j (e^{i \\pi \\sum_{j \\leq k < j + d} \\delta n_k}) \\delta n_{j+d}
```
where ``\\delta \\hat{n}_j = \\hat{n}_j - \\bar{n}`` is the boson number deviation from the mean filling number. 

Assumes a one-dimensional lattice with periodic boundary conditions. 

# Arguments
- `d::Integer`: distance between sites. 

# See also 

* [`HubbardReal1D`](@ref)
* [`G2RealCorrelator`](@ref)
* [`SuperfluidCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)

"""
struct StringCorrelator{D} <: AbstractHamiltonian{Float64}
end

StringCorrelator(d::Int) = StringCorrelator{d}()

function Base.show(io::IO, ::StringCorrelator{D}) where {D}
    print(io, "StringCorrelator($D)")
end

LOStructure(::Type{<:StringCorrelator}) = IsDiagonal()

function diagonal_element(::StringCorrelator{0}, add::SingleComponentFockAddress)
    M = num_modes(add)
    N = num_particles(add)
    n̄ = N/M
    v = onr(add)

    result = 0  
    for i in eachindex(v)
        result += (v[i]- n̄)^2
    end

    return result / M
end

# function diagonal_element(::StringCorrelator{D}, add::SingleComponentFockAddress) where {D}
#     M = num_modes(add)
#     N = num_particles(add)
#     n̄ = N/M
#     d = mod(D, M)
#     v = onr(add)

#     result = 0  
#     for i in eachindex(v)
#         phase_sum = sum( (v[mod1(k,M)] - n̄) for k in i:1:(i+d-1) )

#         result += (v[i]- n̄) * exp(pi * im * phase_sum) * (v[mod1(i + d, M)]-n̄)
#     end

#     if M == N
#         return real(result) /M
#     end

#     return result / M
# end

num_offdiagonals(::StringCorrelator, ::SingleComponentFockAddress) = 0

function diagonal_element(::StringCorrelator{D}, add::SingleComponentFockAddress) where {D}
    M = num_modes(add)
    N = num_particles(add)
    d = mod(D, M)
    v = onr(add)

    if iszero(N % M)
        n̄ = N ÷ M
        
        result = 0.  
        for i in eachindex(v)
            phase_sum = sum( (v[mod1(k,M)] - n̄) for k in i:1:(i+d-1) )

            result += (v[i]- n̄) * (-1)^(phase_sum) * (v[mod1(i + d, M)]-n̄)
        end
    
        return result / M

    else
        n̄ = N/M

        result = ComplexF64(0)  
        for i in eachindex(v)
            phase_sum = sum( (v[mod1(k,M)] - n̄) for k in i:1:(i+d-1) )
    
            result += (v[i]- n̄) * exp(pi * im * phase_sum) * (v[mod1(i + d, M)]-n̄)
        end
    
        return result / M
    end
end