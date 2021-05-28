# Methods that need to be implemented:
#   •  starting_address(::AbstractHamiltonian) - not needed

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
