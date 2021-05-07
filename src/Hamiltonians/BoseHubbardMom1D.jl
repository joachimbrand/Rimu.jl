@with_kw struct BoseHubbardMom1D{T, AD} <: AbstractHamiltonian{T}
    n::Int = 6    # number of bosons
    m::Int = 6    # number of lattice sites
    u::T = 1.0    # interaction strength
    t::T = 1.0    # hopping strength
    # AT::Type = BSAdd64 # address type
    add::AD       # starting address
end

@doc """
    ham = BoseHubbardMom1D(;[n=6, m=6, u=1.0, t=1.0], add = add)
    ham = BoseHubbardMom1D(add; u=1.0, t=1.0)

Implements a one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}\\\\
ϵ_k = -2t \\cos(k)
```

!!! warning
    This Hamiltonian is deprecated. Please use [`HubbardMom1D`](@ref) instead.

# Arguments
- `n::Int`: the number of bosons
- `m::Int`: the number of lattice sites
- `u::Float64`: the interaction parameter
- `t::Float64`: the hopping strength
- `AT::Type`: the address type

""" BoseHubbardMom1D

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardMom1D{T, AD}}) where {T <: Real, AD} = Hermitian()

"""
    BoseHubbardMom1D(add::AbstractFockAddress; u=1.0, t=1.0)
Set up the `BoseHubbardMom1D` with the correct particle and mode number and
address type. Parameters `u` and `t` can be passed as keyword arguments.
"""
function BoseHubbardMom1D(add::BSA; u=1.0, t=1.0) where BSA <: AbstractFockAddress
    U, T = promote(float(u), float(t))
    n = num_particles(add)
    m = num_modes(add)
    return BoseHubbardMom1D{typeof(T),typeof(add)}(n,m,U,T,add)
end

starting_address(ham::BoseHubbardMom1D) = ham.add

momentum(ham::BoseHubbardMom1D) = MomentumMom1D(ham)

function num_offdiagonals(ham::BoseHubbardMom1D, add)
    singlies, doublies = num_singly_doubly_occupied_sites(add)
    return singlies*(singlies-1)*(ham.m - 2) + doublies*(ham.m - 1)
    # number of excitations that can be made
end

function get_offdiagonal(ham::BoseHubbardMom1D, add::ADDRESS, chosen) where ADDRESS
    onr = BitStringAddresses.m_onr(add) # get occupation number representation as a mutable array
    singlies, doublies = num_singly_doubly_occupied_sites(add)
    onproduct = 1
    k = p = q = 0
    double = chosen - singlies*(singlies-1)*(ham.m - 2)
    # start by making holes as the action of two annihilation operators
    if double > 0 # need to choose doubly occupied site for double hole
        # c_p c_p
        double, q = fldmod1(double, ham.m-1)
        # double is location of double
        # q is momentum transfer
        for (i, occ) in enumerate(onr)
            if occ > 1
                double -= 1
                if double == 0
                    onproduct *= occ*(occ-1)
                    onr[i] = occ-2 # annihilate two particles in onr
                    p = k = i # remember where we make the holes
                    break # should break out of the for loop
                end
            end
        end
    else # need to punch two single holes
        # c_k c_p
        pair, q = fldmod1(chosen, ham.m-2) # floored integer division and modulus in ranges 1:(m-1)
        first, second = fldmod1(pair, singlies-1) # where the holes are to be made
        if second < first # put them in ascending order
            f_hole = second
            s_hole = first
        else
            f_hole = first
            s_hole = second + 1 # as we are counting through all singlies
        end
        counter = 0
        for (i, occ) in enumerate(onr)
            if occ > 0
                counter += 1
                if counter == f_hole
                    onproduct *= occ
                    onr[i] = occ -1 # punch first hole
                    p = i # location of first hole
                elseif counter == s_hole
                    onproduct *= occ
                    onr[i] = occ -1 # punch second hole
                    k = i # location of second hole
                    break
                end
            end
        end
        # we have p<k and 1 < q < ham.m - 2
        if q ≥ k-p
            q += 1 # to avoid putting particles back into the holes
        end
    end # if double > 0 # we're done punching holes

    # now it is time to deal with two creation operators
    # c^†_k-q
    kmq = mod1(k-q, ham.m) # in 1:m # use mod1() to implement periodic boundaries
    occ = onr[kmq]
    onproduct *= occ + 1
    onr[kmq] = occ + 1
    # c^†_p+q
    ppq = mod1(p+q, ham.m) # in 1:m # use mod1() to implement periodic boundaries
    occ = onr[ppq]
    onproduct *= occ + 1
    onr[ppq] = occ + 1

    return ADDRESS(onr), ham.u/(2*ham.m)*sqrt(onproduct)
    # return new address and matrix element
end


"""
    ks(h::BoseHubbardMom1D)
Return a range for `k` values in the interval (-π, π] to be `dot()`ed to an `onr()`
occupation number representation.
"""
function ks(h::BoseHubbardMom1D)
    m = num_modes(h.add)
    step = 2π/m
    if isodd(m)
        start = -π*(1+1/m) + step
    else
        start = -π + step
    end
    return StepRangeLen(start, step, m) # faster than range()
end # fast! - can be completely resolved by compiler


function diagonal_element(h::BoseHubbardMom1D, add)
    onrep = BitStringAddresses.onr(add) # get occupation number representation

    # single particle part of Hubbard momentum space Hamiltonian
    # ke = -2*h.t.*cos.(ks(h))⋅onrep # works but allocates memory due to broadcasting
    # ugly but no allocations:
    ke = 0.0
    for (k,on) in zip(ks(h),onrep)
        ke += -2*h.t * cos(k) * on
    end

    # now compute diagonal interaction energy
    onproduct = 0 # Σ_kp < c^†_p c^†_k c_k c_p >
    # for p in 1:h.m
    #   for k in 1:h.m
    #     if k==p
    #       onproduct += onrep[k]*(onrep[k]-1)
    #     else
    #       onproduct += 2*onrep[k]*onrep[p] # two terms in sum over creation operators
    #     end
    #   end
    # end
    for p = 1:h.m
        # faster triangular loop; 9 μs instead of 33 μs for nearUniform(BoseFS{200,199})
        @inbounds onproduct += onrep[p] * (onrep[p] - 1)
        @inbounds @simd for k = 1:p-1
            onproduct += 4*onrep[k]*onrep[p]
        end
    end
    # @show onproduct
    pe = h.u/(2*h.m)*onproduct
    return ke + pe
end
