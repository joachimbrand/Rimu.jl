"""
    BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + \\frac{V}{M}\\sum_{kpqr} b^†_{r} a^†_{q} b_p a_k δ_{r+q,p+k}
```

# Arguments

* `add`: the starting address.
* `ua`: the `u` parameter for Hamiltonian a.
* `ub`: the `u` parameter for Hamiltonian b.
* `ta`: the `t` parameter for Hamiltonian a.
* `tb`: the `t` parameter for Hamiltonian b.
* `v`: the inter-species interaction parameter V.

# See also

* [`HubbardMom1D`](@ref)
* [`BoseHubbardReal1D2C`](@ref)

"""
struct BoseHubbardMom1D2C{T,HA,HB,V} <: TwoComponentHamiltonian{T}
    ha::HA
    hb::HB
end

function BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0)
    ha = HubbardMom1D(add.bsa;u=ua,t=ta)
    hb = HubbardMom1D(add.bsb;u=ub,t=tb)
    T = promote_type(eltype(ha), eltype(hb))
    return BoseHubbardMom1D2C{T,typeof(ha),typeof(hb),v}(ha, hb)
end

function Base.show(io::IO, h::BoseHubbardMom1D2C)
    addr = starting_address(h)
    ua = h.ha.u
    ub = h.hb.u
    ta = h.ha.t
    tb = h.hb.t
    v = h.v
    print(io, "BoseHubbardMom1D2C($addr; ua=$ua, ub=$ub, ta=$ta, tb=$tb, v=$v)")
end

function starting_address(h::BoseHubbardMom1D2C)
    return BoseFS2C(starting_address(h.ha), starting_address(h.hb))
end

LOStructure(::Type{<:BoseHubbardMom1D2C{<:Real}}) = HermitianLO()

Base.getproperty(h::BoseHubbardMom1D2C, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::BoseHubbardMom1D2C, ::Val{:ha}) = getfield(h, :ha)
Base.getproperty(h::BoseHubbardMom1D2C, ::Val{:hb}) = getfield(h, :hb)
Base.getproperty(h::BoseHubbardMom1D2C{<:Any,<:Any,<:Any,V}, ::Val{:v}) where {V} = V

function numOfHops(ham::BoseHubbardMom1D2C, add::BoseFS2C)
    M = num_modes(add)
    sa = numberoccupiedsites(add.bsa)
    sb = numberoccupiedsites(add.bsb)
    return numOfHops(ham.ha, add.bsa) + numOfHops(ham.hb, add.bsb) + sa*(M-1)*sb
    # number of excitations that can be made
end

function diagME(ham::BoseHubbardMom1D2C, add::BoseFS2C)
    M = num_modes(add)
    onrep_a = onr(add.bsa)
    onrep_b = onr(add.bsb)
    interaction2c = Int32(0)
    for p in 1:M
        iszero(onrep_b[p]) && continue
        for k in 1:M
            interaction2c += onrep_a[k]*onrep_b[p] # b†_p b_p a†_k a_k
        end
    end
    return diagME(ham.ha, add.bsa) + diagME(ham.hb, add.bsb) + ham.v/M*interaction2c
end

"""
    hop_across_two_addresses(add_a, add_b, chosen[, sa, sb])

Perform a hop across two addresses (in momentum space). Optional arguments `sa` and `sb`
should equal the numbers of occupied sites in the respective components.
"""
@inline function hop_across_two_addresses(
    add_a::BoseFS{NA,M}, add_b::BoseFS{NB,M}, chosen, sa, sb
) where {NA,NB,M}
    onrep_a = BitStringAddresses.m_onr(add_a)
    onrep_b = BitStringAddresses.m_onr(add_b)
    # b†_s b_q a†_p a_r
    s = p = q = r = 0
    onproduct_a = 1
    onproduct_b = 1
    hole_a, remainder = fldmod1(chosen, (M - 1) * sb) # hole_a: position for hole_a
    p, hole_b = fldmod1(remainder, sb) # hole_b: position for hole_b
    # annihilate an A boson:
    for (i, occ) in enumerate(onrep_a)
        if occ > 0
            hole_a -= 1 # searching for the position for hole_a
            if hole_a == 0 # found the hole_a here
                onproduct_a *= occ # record the normalisation factor before annihilate
                @inbounds onrep_a[i] = occ - 1 # annihilate an A boson: a_r
                r = i # remember where we make the hole
                break # should break out of the for loop
            end
        end
    end
    if p ≥ r
        p += 1 # to skip the hole_a
    end
    # create an A boson:
    ΔP = p - r # change in momentun
    p = mod1(p, M) # enforce periodic boundary condition
    @inbounds onrep_a[p] += 1 # create an A boson: a†_p
    @inbounds onproduct_a *= onrep_a[p] # record the normalisation factor after creation
    # annihilate a B boson:
    for (i, occ) in enumerate(onrep_b)
        if occ > 0
            hole_b -= 1 # searching for the position for hole_b
            if hole_b == 0 # found the hole_b here
                onproduct_b *= occ # record the normalisation factor before annihilate
                @inbounds onrep_b[i] = occ-1 # annihilate a B boson: b_q
                q = i # remember where we make the holes
                break # should break out of the for loop
            end
        end
    end
    s = mod1(q-ΔP, M) # compute s with periodic boundary condition
    # create a B boson:
    @inbounds onrep_b[s] += 1 # create a B boson: b†_s
    @inbounds onproduct_b *= onrep_b[s] # record the normalisation factor after creation
    # if mod(q+r,M)-mod(s+p,M) != 0 # sanity check for momentum conservation
    #     error("Momentum is not conserved!")
    # end
    return BoseFS{NA,M}(onrep_a), BoseFS{NB,M}(onrep_b), onproduct_a, onproduct_b
end

function hop(ham::BoseHubbardMom1D2C, add::BoseFS2C, chosen)
    M = num_modes(add)
    nhops_a = numOfHops(ham.ha, add.bsa)
    nhops_b = numOfHops(ham.hb, add.bsb)
    if chosen ≤ nhops_a
        naddress_from_bsa, elem = hop(ham.ha, add.bsa, chosen)
        return BoseFS2C(naddress_from_bsa, add.bsb), elem
    elseif nhops_a < chosen ≤ nhops_a + nhops_b
        chosen -= nhops_a
        naddress_from_bsb, elem = hop(ham.hb, add.bsb, chosen)
        return BoseFS2C(add.bsa, naddress_from_bsb), elem
    else
        chosen -= nhops_a + nhops_b
        sa = numberoccupiedsites(add.bsa)
        sb = numberoccupiedsites(add.bsb)
        new_bsa, new_bsb, onproduct_a, onproduct_b = hop_across_two_addresses(
            add.bsa, add.bsb, chosen, sa, sb
        )
        new_add = BoseFS2C(new_bsa, new_bsb)
        # return new_add, elem
        elem = ham.v/M * sqrt(onproduct_a * onproduct_b)
        new_add = BoseFS2C(new_bsa, new_bsb)
        return new_add, elem
    end
    # return new address and matrix element
end

"""
    HopsBoseMom1D2C

Specialized [`AbstractHops`](@ref) that keep track of number of hops and number of occupied
sites in both components of the address.
"""
struct HopsBoseMom1D2C{A<:BoseFS2C,V,T,H<:TwoComponentHamiltonian{T}} <: AbstractHops{A,T}
    hamiltonian::H
    address::A
    length::Int
    num_hops_a::Int
    num_hops_b::Int
    num_occupied_a::Int
    num_occupied_b::Int
end

function hops(h::BoseHubbardMom1D2C{T,<:Any,<:Any,V}, a::BoseFS2C) where {T,V}
    hops_a = numOfHops(h.ha, a.bsa)
    hops_b = numOfHops(h.hb, a.bsb)
    occ_a = numberoccupiedsites(a.bsa)
    occ_b = numberoccupiedsites(a.bsb)
    length = hops_a + hops_b + occ_a * (num_modes(a) - 1) * occ_b

    return HopsBoseMom1D2C{typeof(a),V,T,typeof(h)}(
        h, a, length, hops_a, hops_b, occ_a, occ_b
    )
end

function Base.getindex(s::HopsBoseMom1D2C{A,V}, i) where {A,V}
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    if i ≤ s.num_hops_a
        new_a, matrix_element = hop(s.hamiltonian.ha, s.address.bsa, i)
        new_add = A(new_a, s.address.bsb)
    elseif i ≤ s.num_hops_a + s.num_hops_b
        i -= s.num_hops_a
        new_b, matrix_element = hop(s.hamiltonian.hb, s.address.bsb, i)
        new_add = A(s.address.bsa, new_b)
    else
        i -= s.num_hops_a + s.num_hops_b
        new_a, new_b, prod_a, prod_b = hop_across_two_addresses(
            s.address.bsa, s.address.bsb, i, s.num_occupied_a, s.num_occupied_b
        )
        new_add = A(new_a, new_b)
        matrix_element = V/num_modes(new_add) * sqrt(prod_a * prod_b)
    end
    return new_add, matrix_element
end

Base.size(s::HopsBoseMom1D2C) = (s.length,)
