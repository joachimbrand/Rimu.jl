"""
    ham = BoseHubbardReal1D2C(add::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + V\\sum_{i} n_{a_i}n_{b_i}
```

# Arguments
* `add::BoseFS2C`: the two-component address type, see [`BoseFS2C`](@ref)
* `h_a::BoseHubbardReal1D` and `h_b::BoseHubbardReal1D`: standard Hamiltonian for boson A and B, see [`BoseHubbardReal1D`](@ref)
* `v`: the inter-species interaction parameter V

"""
struct BoseHubbardReal1D2C{T, HA, HB, V} <: TwoComponentHamiltonian{T}
    ha::HA
    hb::HB
end

# set the `LOStructure` trait
LOStructure(::Type{<:BoseHubbardReal1D2C{<:Real}}) = HermitianLO()

function BoseHubbardReal1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)
    ha = HubbardReal1D(add.bsa; u=ua, t=ta)
    hb = HubbardReal1D(add.bsb; u=ub, t=tb)
    T = promote_type(eltype(ha), eltype(hb))
    return BoseHubbardReal1D2C{T,typeof(ha),typeof(hb),v}(ha, hb)
end

function Base.show(io::IO, h::BoseHubbardReal1D2C{<:Any,<:Any,<:Any,V}) where V
    addr = starting_address(h)
    ua = h.ha.u
    ub = h.hb.u
    ta = h.ha.t
    tb = h.hb.t
    print(io, "BoseHubbardReal1D2C($addr; ua=$ua, ub=$ub, ta=$ta, tb=$tb, v=$V)")
end

function starting_address(h::BoseHubbardReal1D2C)
    return BoseFS2C(starting_address(h.ha), starting_address(h.hb))
end

# number of excitations that can be made
function numOfHops(ham::BoseHubbardReal1D2C, add)
    return 2*(numberoccupiedsites(add.bsa) + numberoccupiedsites(add.bsb))
end

function bosehubbard2Cinteraction(add::BoseFS2C)
    c1 = onr(add.bsa)
    c2 = onr(add.bsb)
    interaction = zero(eltype(c1))
    for site = 1:length(c1)
        if !iszero(c2[site])
            interaction += c2[site] * c1[site]
        end
    end
    return interaction
end

function diagME(ham::BoseHubbardReal1D2C{T,HA,HB,V}, address::BoseFS2C) where {T,HA,HB,V}
    return (
        diagME(ham.ha, address.bsa) +
        diagME(ham.hb, address.bsb) +
        V * bosehubbard2Cinteraction(address)
    )
end

function hop(ham::BoseHubbardReal1D2C, add, chosen)
    nhops = numOfHops(ham,add)
    nhops_a = 2*numberoccupiedsites(add.bsa)
    if chosen in 1:nhops_a
        naddress_from_bsa, onproduct = hopnextneighbour(add.bsa, chosen)
        elem = - ham.ha.t*sqrt(onproduct)
        return BoseFS2C(naddress_from_bsa,add.bsb), elem
    elseif chosen in nhops_a+1:nhops
        chosen -= nhops_a
        naddress_from_bsb, onproduct = hopnextneighbour(add.bsb, chosen)
        elem = -ham.hb.t * sqrt(onproduct)
        return BoseFS2C(add.bsa,naddress_from_bsb), elem
    else
        error("invalid hop")
    end
    # return new address and matrix element
end

struct HopsBoseReal1D2C{A<:BoseFS2C,T,H<:TwoComponentHamiltonian{T}} <: AbstractHops{A,T}
    hamiltonian::H
    address::A
    length::Int
    num_hops_a::Int
end

function hops(h::BoseHubbardReal1D2C, a::BoseFS2C)
    hops_a = numOfHops(h.ha, a.bsa)
    hops_b = numOfHops(h.hb, a.bsb)
    length = hops_a + hops_b

    return HopsBoseReal1D2C(h, a, length, hops_a)
end

function Base.getindex(s::HopsBoseReal1D2C{A}, i) where {A}
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    if i ≤ s.num_hops_a
        new_a, matrix_element = hop(s.hamiltonian.ha, s.address.bsa, i)
        new_add = A(new_a, s.address.bsb)
    else
        i -= s.num_hops_a
        new_b, matrix_element = hop(s.hamiltonian.hb, s.address.bsb, i)
        new_add = A(s.address.bsa, new_b)
    end
    return new_add, matrix_element
end

Base.size(s::HopsBoseReal1D2C) = (s.length,)
