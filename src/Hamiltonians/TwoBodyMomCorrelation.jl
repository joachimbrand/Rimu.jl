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

"""
    diagonal_element(g::G2MomCorrelator, add::BoseFS2C{NA,NB,M,AA,AB})

The diagonal element in [`G2MomCorrelator`](@ref), where `(p-q)=0`, hence
it becomes
```math
\\frac{1}{M}\\sum_{k,p=1}^M a^†_{k} b^†_{p}  b_p a_k .
```
# See also

* [`G2MomCorrelator`](@ref)
"""
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
