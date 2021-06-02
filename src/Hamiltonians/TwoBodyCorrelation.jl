# Methods that need to be implemented:
#   •  starting_address(::AbstractHamiltonian) - not needed
"""
    G2Correlator(d::Int) <: AbstractHamiltonian{ComplexF64}

Two-body correlation operator representing the inter-component density-
density correlation at distance `d` of a two component system 
in a momentum-space Fock-state basis. 
It returns a `Complex` value.
It currently only works on [`BoseFS2C`](@ref).


```math
\\hat{G}_2(d) = \\frac{1}{M}\\sum_{spqr=1}^M e^{-id(p-q)2π/M} a^†_{s} b^†_{p}  b_q a_r δ_{s+p,q+r}
```

# See also

* [`BoseHubbardMom1D2C`](@ref)
* [`BoseFS2C`](@ref)
* [`AbstractHamiltonian`](@ref)

"""
struct G2Correlator <: AbstractHamiltonian{ComplexF64}
    d::Int
end

function num_offdiagonals(g::G2Correlator, add::BoseFS2C)
    M = num_modes(add)
    sa = numberoccupiedsites(add.bsa)
    sb = numberoccupiedsites(add.bsb)
    return sa*(M-1)*sb
    # number of excitations that can be made
end

"""
    diagonal_element(g::G2Correlator, add::BoseFS2C{NA,NB,M,AA,AB})

The diagonal element in [`G2Correlator`](@ref), where `(p-q)=0`, hence
it becomes
```math
\\frac{1}{M}\\sum_{k,p=1}^M a^†_{k} b^†_{p}  b_p a_k .
```
# See also

* [`G2Correlator`](@ref)
"""
function diagonal_element(g::G2Correlator, add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
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

function get_offdiagonal(g::G2Correlator, add::BoseFS2C{NA,NB,M,AA,AB}, chosen) where {NA,NB,M,AA,AB}
    sa = numberoccupiedsites(add.bsa)
    sb = numberoccupiedsites(add.bsb)
    new_bsa, new_bsb, onproduct_a, onproduct_b, p, q = hop_across_two_addresses(add.bsa, add.bsb, chosen, sa, sb)
    new_add = BoseFS2C(new_bsa, new_bsb)
    gamma = sqrt(onproduct_a*onproduct_b)
    gd = exp(-im*g.d*(p-q)*2π/M)*gamma
    return new_add, gd/M
end
