# Methods that need to be implemented:
#   •  starting_address(::AbstractHamiltonian) - not needed
"""
    TwoBodyCorrelation(d::Int)

Implements the two-body correlation operator computing the correlation function
between the impurity and a boson from the Bose gas as a function of the distance `d`, which can take
`Integer` values from `0` to `M` (the number of modes).
It currently only works on [`BoseFS2C`](@ref) on Fock states in momentum space. It returns a `Complex` value.

```math
\\hat{G}_2^{(d)} = \\frac{1}{M}\\sum_{spqr}^M e^{-id(p-q)2π/M} a^†_{s} b^†_{p}  b_q a_r δ_{s+p,q+r}
```

# Arguments

* `d`: the distance between the impurity and a boson from the Bose gas.

# See also

* [`BoseHubbardMom1D2C`](@ref)
* [`BoseFS2C`](@ref)

"""
struct TwoBodyCorrelation <: AbstractHamiltonian{ComplexF64}
    d::Int
end

function num_offdiagonals(g::TwoBodyCorrelation, add::BoseFS2C)
    M = num_modes(add)
    sa = numberoccupiedsites(add.bsa)
    sb = numberoccupiedsites(add.bsb)
    return sa*(M-1)*sb
    # number of excitations that can be made
end

"""
    diagonal_element(g::TwoBodyCorrelation, add::BoseFS2C{NA,NB,M,AA,AB})

The diagonal element in [`TwoBodyCorrelation`](@ref), where `(p-q)=0`, hence
it becomes
```math
\\frac{1}{M}\\sum_{spqr}^M a^†_{k} b^†_{p}  b_p a_k .
```
# See also

* [`TwoBodyCorrelation`](@ref)
"""
function diagonal_element(g::TwoBodyCorrelation, add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
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

function get_offdiagonal(g::TwoBodyCorrelation, add::BoseFS2C{NA,NB,M,AA,AB}, chosen) where {NA,NB,M,AA,AB}
    sa = numberoccupiedsites(add.bsa)
    sb = numberoccupiedsites(add.bsb)
    new_bsa, new_bsb, onproduct_a, onproduct_b, p, q = hop_across_two_addresses(add.bsa, add.bsb, chosen, sa, sb)
    new_add = BoseFS2C(new_bsa, new_bsb)
    gamma = sqrt(onproduct_a*onproduct_b)
    gd = exp(-im*g.d*(p-q)*2π/M)*gamma
    return new_add, gd/M
end
