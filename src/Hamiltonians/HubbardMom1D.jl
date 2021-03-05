"""
    HubbardMom1D(address; u=1.0, t=1.0)

Implements a one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = -t \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}\\\\
ϵ_k = - 2 t \\cos(k)
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.

# See also

* [`HubbardReal1D`](@ref)
* [`ExtendedHubbardReal1D`](@ref)

"""
struct HubbardMom1D{TT,M,AD<:AbstractFockAddress,U,T} <: AbstractHamiltonian{TT}
    add::AD # default starting address, should have N particles and M modes
    ks::SVector{M,TT} # values for k
    kes::SVector{M,TT} # values for kinetic energy
end

function HubbardMom1D(add::BoseFS{<:Any,M}; u=1.0, t=1.0) where {M}
    U, T = promote(float(u), float(t))
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    kes = SVector{M}(-2*cos.(kr))
    return HubbardMom1D{typeof(U),M,typeof(add),U,T}(add, ks, kes)
end

function Base.show(io::IO, h::HubbardMom1D)
    print(io, "HubbardMom1D($(h.add); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardMom1D)
    return h.add
end

LOStructure(::Type{<:HubbardMom1D{<:Real}}) = HermitianLO()

Base.getproperty(h::HubbardMom1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardMom1D, ::Val{:ks}) = getfield(h, :ks)
Base.getproperty(h::HubbardMom1D, ::Val{:kes}) = getfield(h, :kes)
Base.getproperty(h::HubbardMom1D, ::Val{:add}) = getfield(h, :add)
Base.getproperty(h::HubbardMom1D{<:Any,<:Any,<:Any,U}, ::Val{:u}) where {U} = U
Base.getproperty(h::HubbardMom1D{<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where {T} = T

ks(h::HubbardMom1D) = getfield(h, :ks)

"""
    singlies, doublies = numSandDoccupiedsites(address)
Returns the number of singly and doubly occupied sites for a bosonic bit string address.
"""
function numSandDoccupiedsites(b::BoseFS)
    singlies = 0
    doublies = 0
    for (n, _, _) in occupied_orbitals(b)
        singlies += 1
        doublies += n > 1
    end
    return singlies, doublies
end

function numSandDoccupiedsites(onrep::AbstractArray)
    # this one is faster by about a factor of 2 if you already have the onrep
    # returns number of singly and doubly occupied sites
    singlies = 0
    doublies = 0
    for n in onrep
        if n > 0
            singlies += 1
            if n > 1
                doublies += 1
            end
        end
    end
    return singlies, doublies
end

# standard interface function
function numOfHops(ham::HubbardMom1D, add::BoseFS)
    singlies, doublies = numSandDoccupiedsites(add)
    return numOfHops(ham, add, singlies, doublies)
end

# 3-argument version
@inline function numOfHops(ham::HubbardMom1D, add::BoseFS, singlies, doublies)
    M = num_modes(ham)
    return singlies*(singlies-1)*(M - 2) + doublies*(M - 1)
end

@inline function interaction_energy_diagonal(
    h::HubbardMom1D{<:Any,M,<:BoseFS}, onrep::StaticVector{M,I}
) where {M,I}
    # now compute diagonal interaction energy
    onproduct = zero(I) # Σ_kp < c^†_p c^†_k c_k c_p >
    # Not having @inbounds here is faster?
    for p in 1:M
        iszero(onrep[p]) && continue
        onproduct += onrep[p] * (onrep[p] - one(I))
        for k in 1:p-1
            onproduct += I(4) * onrep[k] * onrep[p]
        end
    end
    return h.u / 2M * onproduct
end

function kinetic_energy(h::HubbardMom1D, add::AbstractFockAddress)
    onrep = BitStringAddresses.m_onr(add) # get occupation number representation
    return kinetic_energy(h, onrep)
end

@inline function kinetic_energy(h::HubbardMom1D, onrep::StaticVector)
    return h.kes⋅onrep # safe as onrep is Real
end

@inline function diagME(h::HubbardMom1D, add)
    onrep = BitStringAddresses.m_onr(add) # get occupation number representation
    return diagME(h, onrep)
end

@inline function diagME(h::HubbardMom1D, onrep::StaticVector)
    return kinetic_energy(h, onrep) + interaction_energy_diagonal(h, onrep)
end

@inline function hop(ham::HubbardMom1D, add, chosen)
    hop(ham, add, chosen, numSandDoccupiedsites(add)...)
end

@inline function hop(
    ham::HubbardMom1D{<:Any,M,A}, add, chosen, singlies, doublies
) where {M,A}
    onrep = BitStringAddresses.m_onr(add)
    # get occupation number representation as a static array
    onproduct = 1
    k = p = q = 0
    double = chosen - singlies*(singlies-1)*(M - 2)
    # start by making holes as the action of two annihilation operators
    if double > 0 # need to choose doubly occupied site for double hole
        # c_p c_p
        double, q = fldmod1(double, M-1)
        # double is location of double
        # q is momentum transfer
        for (i, occ) in enumerate(onrep)
            if occ > 1
                double -= 1
                if double == 0
                    onproduct *= occ*(occ-1)
                    @inbounds onrep[i] = occ-2
                    # annihilate two particles in onrep
                    p = k = i # remember where we make the holes
                    break # should break out of the for loop
                end
            end
        end
    else # need to punch two single holes
        # c_k c_p
        pair, q = fldmod1(chosen, M-2) # floored integer division and modulus in ranges 1:(m-1)
        first, second = fldmod1(pair, singlies-1) # where the holes are to be made
        if second < first # put them in ascending order
            f_hole = second
            s_hole = first
        else
            f_hole = first
            s_hole = second + 1 # as we are counting through all singlies
        end
        counter = 0
        for (i, occ) in enumerate(onrep)
            if occ > 0
                counter += 1
                if counter == f_hole
                    onproduct *= occ
                    @inbounds onrep[i] = occ-1
                    # punch first hole
                    p = i # location of first hole
                elseif counter == s_hole
                    onproduct *= occ
                    @inbounds onrep[i] = occ-1
                    # punch second hole
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
    kmq = mod1(k-q, M) # in 1:m # use mod1() to implement periodic boundaries
    @inbounds occ = onrep[kmq]
    onproduct *= occ + 1
    @inbounds onrep[kmq] = occ + 1
    # c^†_p+q
    ppq = mod1(p+q, M) # in 1:m # use mod1() to implement periodic boundaries
    @inbounds occ = onrep[ppq]
    onproduct *= occ + 1
    @inbounds onrep[ppq] = occ + 1

    return A(SVector(onrep)), ham.u/(2*M)*sqrt(onproduct)
    # return new address and matrix element
end

###
### hops
###
struct HopsBoseMom1D{A<:BoseFS,T,H<:AbstractHamiltonian{T}} <: AbstractHops{A,T}
    hamiltonian::H
    address::A
    length::Int
    singlies::Int
    doublies::Int
end

function hops(h::HubbardMom1D, a::BoseFS)
    singlies, doublies = numSandDoccupiedsites(a)
    num = numOfHops(h, a, singlies, doublies)
    return HopsBoseMom1D(h, a, num, singlies, doublies)
end

function Base.getindex(s::HopsBoseMom1D, i)
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    new_address, matrix_element = hop(s.hamiltonian, s.address, i, s.singlies, s.doublies)
    return (new_address, matrix_element)
end

Base.size(s::HopsBoseMom1D) = (s.length,)

###
### momentum
###
struct MomentumMom1D{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    ham::H
end
LOStructure(::Type{MomentumMom1D{H,T}}) where {H,T <: Real} = HermitianLO()
numOfHops(ham::MomentumMom1D, add) = 0
diagME(mom::MomentumMom1D, add) = mod1(onr(add)⋅ks(mom.ham) + π, 2π) - π # fold into (-π, π]

momentum(ham::HubbardMom1D) = MomentumMom1D(ham)
