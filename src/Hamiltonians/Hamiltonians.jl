"""
This module defines Hamiltonian types and standard methods.
Model Hamiltonians should be subtyped to [`AbstractHamiltonian`](@ref).
Models implemented so far are:

* [`BoseHubbardReal1D`](@ref) Bose-Hubbard chain, real space
* [`ExtendedBHReal1D`](@ref) extended Bose-Hubbard model with on-site and nearest neighbour interactions, real space, one dimension
"""
module Hamiltonians

using Parameters, StaticArrays, LinearAlgebra, SparseArrays
using Setfield

using ..DictVectors
using ..BitStringAddresses
using ..ConsistentRNG

export AbstractHamiltonian, TwoComponentBosonicHamiltonian, Hops, generateRandHop
export diagME, numOfHops, hop, hasIntDimension, dimensionLO, fDimensionLO
export rayleigh_quotient

export BosonicHamiltonian, bit_String_Length
export BoseHubbardReal1D, ExtendedBHReal1D, BoseHubbardReal1D2C
export BoseHubbardMom1D, Momentum, BoseHubbardMom1D2C
export HubbardMom1D
export HubbardReal1D

include("abstract.jl")

include("BoseHubbardReal1D.jl")
include("HubbardMom1D.jl")

include("BoseHubbardReal1D2C.jl")
include("BoseHubbardMom1D2C.jl")

include("BoseHubbardMom1D.jl")
include("ExtendedBHReal1D.jl")
include("Momentum.jl")

include("HubbardReal1D.jl")

const BoseHubbardExtOrNot = Union{ExtendedBHReal1D, BoseHubbardReal1D}
# type alias for convenience

"""
    numOfHops(ham, add)

Compute the number of number of reachable configurations from address `add`.
"""
function numOfHops(ham::BoseHubbardExtOrNot, add)
    return numberlinkedsites(add)
end

"""
    newadd, me = hop(ham, add, chosen)

Compute matrix element of `hamiltonian` and new address of a single hop from
address `add` with integer index `chosen`.
"""
function hop(ham::BoseHubbardExtOrNot, add, chosen::Integer)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - ham.t*sqrt(onproduct)
    # return new address and matrix element
end


################################################
#
# Internals of the Bose Hubbard model:
# private functions (not exported)
"""
    bosehubbardinteraction(address)

Return Σ_i *n_i* (*n_i*-1) for computing the Bose-Hubbard on-site interaction
(without the *U* prefactor.)
"""
function bosehubbardinteraction(b::BoseFS{<:Any,<:Any,A}) where A
    return bosehubbardinteraction(Val(num_chunks(A)), b)
end

@inline function bosehubbardinteraction(_, b::BoseFS)
    result = 0
    for (n, _, _) in occupied_orbitals(b)
        result += n * (n - 1)
    end
    return result
end

@inline function bosehubbardinteraction(::Val{1}, b::BoseFS)
    # currently this ammounts to counting occupation numbers of orbitals
    chunk = chunks(b.bs)[1]
    matrixelementint = 0
    while !iszero(chunk)
        chunk >>>= trailing_zeros(chunk) # proceed to next occupied orbital
        bosonnumber = trailing_ones(chunk) # count how many bosons inside
        # surpsingly it is faster to not check whether this is nonzero and do the
        # following operations anyway
        chunk >>>= bosonnumber # remove the counted orbital
        matrixelementint += bosonnumber * (bosonnumber - 1)
    end
    return matrixelementint
end

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

function numberoccupiedsites(b::BoseFS{<:Any,<:Any,S}) where S
    return numberoccupiedsites(Val(num_chunks(S)), b)
end

@inline function numberoccupiedsites(::Val{1}, b::BoseFS)
    chunk = b.bs.chunks[1]
    result = 0
    while true
        chunk >>= trailing_zeros(chunk)
        chunk >>= trailing_ones(chunk)
        result += 1
        iszero(chunk) && break
    end
    return result
end

@inline function numberoccupiedsites(_, b::BoseFS)
    # This version is faster than using the occupied_orbital iterator
    address = b.bs
    result = 0
    K = num_chunks(address)
    last_mask = UInt64(1) << 63 # = 0b100000...
    prev_top_bit = false
    # This loop compiles away for address<:BSAdd*
    for i in K:-1:1
        chunk = chunks(address)[i]
        # This part handles sites that span across chunk boundaries.
        # If the previous top bit and the current bottom bit are both 1, we have to subtract
        # 1 from the result or the mode will be counted twice.
        result -= (chunk & prev_top_bit) % Int
        prev_top_bit = (chunk & last_mask) > 0
        while !iszero(chunk)
            chunk >>>= trailing_zeros(chunk)
            chunk >>>= trailing_ones(chunk)
            result += 1
        end
    end
    return result
end

function numberlinkedsites(address)
    # return the number of other walker addresses that are linked in the
    # Hamiltonian
    # here implemented for 1D Bose Hubbard
    return 2*numberoccupiedsites(address)
end

"""
    naddress, onproduct = hopnextneighbour(add, chosen)

Compute the new address of a hopping event for the Bose-Hubbard model.
Returns the new address and the product of occupation numbers of the involved
orbitals.
"""
function hopnextneighbour(b::BoseFS{N,M,A}, chosen) where {N,M,A}
    address = b.bs
    site = (chosen + 1) >>> 1
    if isodd(chosen) # Hopping to the right
        next = 0
        curr = 0
        offset = 0
        sc = 0
        reached_end = false
        for (i, (num, sn, bit)) in enumerate(occupied_orbitals(b))
            next = num * (sn == sc + 1) # only set next to > 0 if sites are neighbours
            reached_end = i == site + 1
            reached_end && break
            curr = num
            offset = bit + num
            sc = sn
        end
        if sc == M
            new_address = ((address ⊻ (A(UInt64(1)) << (offset-1))) << 1) | A(UInt64(1))
            prod = curr * (trailing_ones(address) + 1) # mul occupation num of first obital
        else
            next *= reached_end
            new_address = address ⊻ A(UInt64(3)) << (offset - 1)
            prod = curr * (next + 1)
        end
    else # Hopping to the left
        if site == 1 && isodd(address)
            # For leftmost site, we shift the whole address circularly by one bit.
            new_address = (address >>> 1) | A(UInt64(1)) << (N + M - 2)
            prod = trailing_ones(address) * leading_ones(new_address)
        else
            prev = 0
            curr = 0
            offset = 0
            sp = 0
            for (i, (num, sc, bit)) in enumerate(occupied_orbitals(b))
                prev = curr * (sc == sp + 1) # only set prev to > 0 if sites are neighbours
                curr = num
                offset = bit
                i == site && break
                sp = sc
            end
            new_address = address ⊻ A(UInt64(3)) << (offset - 1)
            prod = curr * (prev + 1)
        end
    end
    return BoseFS{N,M,A}(new_address), prod
end

end # module Hamiltonians
