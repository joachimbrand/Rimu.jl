@with_kw struct BoseHubbardMom1D2C{T, HA, HB, V} <: TwoComponentHamiltonian{T}
    ha:: HA
    hb:: HB
end

@doc """
    ham = BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + \\frac{V}{M}\\sum_{kpqr} b^†_{r} a^†_{q} b_p a_k δ_{r+q,p+k}
```

# Arguments
- `ha::BoseHubbardMom1D` and `hb::BoseHubbardMom1D`: standard Hamiltonian for boson A and B, see [`HubbardMom1D`](@ref)
- `m`: number of modes (needs be the same for `ha` and `hb`!)
- `v=0.0`: the inter-species interaction parameter, default value: 0.0, i.e. non-interacting

""" BoseHubbardMom1D2C

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardMom1D2C{T, HA, HB, V}}) where {T <: Real, HA, HB, V} = HermitianLO()

# function BoseHubbardMom1D2C(ha::HA, hb::HB, v::T) where {M, HA, HB, T} = BoseHubbardMom1D2C{T, HA, HB, M}(ha, hb, v)

function BoseHubbardMom1D2C(add::BoseFS2C{NA,NB,M,AA,AB}; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v::T=1.0) where {NA,NB,M,AA,AB,T}
    ha = HubbardMom1D{NA,M}(add.bsa;u=ua,t=ta)
    hb = HubbardMom1D{NB,M}(add.bsb;u=ub,t=tb)
    return BoseHubbardMom1D2C{T,typeof(ha),typeof(hb),v}(ha, hb)
end

function starting_address(h::BoseHubbardMom1D2C)
    return BoseFS2C(starting_address(h.ha), starting_address(h.hb))
end

function numOfHops(ham::BoseHubbardMom1D2C, add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    sa = numberoccupiedsites(add.bsa)
    sb = numberoccupiedsites(add.bsb)
    return numOfHops(ham.ha, add.bsa) + numOfHops(ham.hb, add.bsb) + sa*(M-1)*sb
    # number of excitations that can be made
end


function hop(ham::BoseHubbardMom1D2C{T,HA,HB,V}, add::BoseFS2C{NA,NB,M,AA,AB}, chosen::Integer) where {T,HA,HB,V,NA,NB,M,AA,AB}
    # ham_a = BoseHubbardMom1D(ham.na, ham.m, ham.ua, ham.ta, add.bsa)
    # ham_b = BoseHubbardMom1D(ham.nb, ham.m, ham.ub, ham.tb, add.bsb)
    nhops_a = numOfHops(ham.ha, add.bsa)
    nhops_b = numOfHops(ham.hb, add.bsb)
    # println("Hops in A: $nhops_a, Hops in B: $nhops_b,")
    # if chosen > numOfHops(ham,add)
    #     error("Hop is out of range!")
    if chosen ≤ nhops_a
        naddress_from_bsa, elem = hop(ham.ha, add.bsa, chosen)
        # println("Hop in A, chosen = $chosen") # debug
        return BoseFS2C{NA,NB,M,AA,AB}(naddress_from_bsa,add.bsb), elem
    elseif nhops_a < chosen ≤ nhops_a+nhops_b
        chosen -= nhops_a
        naddress_from_bsb, elem = hop(ham.hb, add.bsb, chosen)
        # println("Hop in B, chosen = $chosen") # debug
        return BoseFS2C{NA,NB,M,AA,AB}(add.bsa,naddress_from_bsb), elem
    else
        chosen -= (nhops_a+nhops_b)
        sa = numberoccupiedsites(add.bsa)
        sb = numberoccupiedsites(add.bsb)
        # println("Hops across A and B: $(sa*(ham.m-1)*sb)")
        new_bsa, new_bsb, onproduct_a, onproduct_b = hopacross2adds(add.bsa, add.bsb, chosen)
        new_add = BoseFS2C{NA,NB,M,AA,AB}(new_bsa,new_bsb)
        # println("Hop A to B, chosen = $chosen") # debug
        # return new_add, elem
        elem = V/M*sqrt(onproduct_a*onproduct_b)
        new_add = BoseFS2C{NA,NB,M,AA,AB}(new_bsa,new_bsb)
        return new_add, elem
    end
    # return new address and matrix element
end

# hopacross2adds needed for computing hops across two components
@inline function hopacross2adds(add_a::BoseFS{NA,M,AA}, add_b::BoseFS{NB,M,AB}, chosen::Integer) where {NA,NB,M,AA,AB}
    sa = numberoccupiedsites(add_a)
    sb = numberoccupiedsites(add_b)
    onrep_a = onr(add_a)
    onrep_b = onr(add_b)
    # b†_s b_q a†_p a_r
    s = p = q = r = 0
    onproduct_a = 1
    onproduct_b = 1
    hole_a, remainder = fldmod1(chosen, (M-1)*sb) # hole_a: position for hole_a
    p, hole_b = fldmod1(remainder, sb) # hole_b: position for hole_b
    # annihilate an A boson:
    for (i, occ) in enumerate(onrep_a)
        if occ > 0
            hole_a -= 1 # searching for the position for hole_a
            if hole_a == 0 # found the hole_a here
                onproduct_a *= occ # record the normalisation factor before annihilate
                onrep_a = @set onrep_a[i] = occ-1 # annihilate an A boson: a_r
                r = i # remember where we make the hole
                break # should break out of the for loop
            end
        end
    end
    if p ≥ r
        p += 1 # to skip the hole_a
    end
    # create an A boson:
    ΔP = p-r # change in momentun
    p = mod1(p, M) # enforce periodic boundary condition
    onrep_a = @set onrep_a[p] += 1 # create an A boson: a†_p
    onproduct_a *= onrep_a[p] # record the normalisation factor after creation
    # annihilate a B boson:
    for (i, occ) in enumerate(onrep_b)
        if occ > 0
            hole_b -= 1 # searching for the position for hole_b
            if hole_b == 0 # found the hole_b here
                onproduct_b *= occ # record the normalisation factor before annihilate
                onrep_b = @set onrep_b[i] = occ-1 # annihilate a B boson: b_q
                q = i # remember where we make the holes
                break # should break out of the for loop
            end
        end
    end
    s = mod1(q-ΔP, M) # compute s with periodic boundary condition
    # create a B boson:
    onrep_b = @set onrep_b[s] += 1 # create a B boson: b†_s
    onproduct_b *= onrep_b[s] # record the normalisation factor after creation
    # if mod(q+r,M)-mod(s+p,M) != 0 # sanity check for momentum conservation
    #     error("Momentum is not conserved!")
    # end
    return BoseFS{NA,M,AA}(onrep_a), BoseFS{NB,M,AB}(onrep_b), onproduct_a, onproduct_b
end


function diagME(ham::BoseHubbardMom1D2C{T,HA,HB,V}, add::BoseFS2C{NA,NB,M,AA,AB}) where {T,HA,HB,V,NA,NB,M,AA,AB}
    # ham_a = BoseHubbardMom1D(ham.na, ham.m, ham.ua, ham.ta, add.bsa)
    # ham_b = BoseHubbardMom1D(ham.nb, ham.m, ham.ub, ham.tb, add.bsb)
    onrep_a = BitStringAddresses.onr(add.bsa)
    onrep_b = BitStringAddresses.onr(add.bsb)
    interaction2c = Int32(0)
    for p in 1:M
        for k in 1:M
            interaction2c += onrep_a[k]*onrep_b[p] # b†_p b_p a†_k a_k
        end
    end
    return diagME(ham.ha, add.bsa) + diagME(ham.hb, add.bsb) + V/M*interaction2c
end
