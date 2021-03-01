@with_kw struct BoseHubbardReal1D2C{T, HA, HB, V} <: TwoComponentBosonicHamiltonian{T}
    ha:: HA
    hb:: HB
end

@doc """
    ham = BoseHubbardReal1D2C(add::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + V\\sum_{i} n_{a_i}n_{b_i}
```

# Arguments
- `add::BoseFS2C`: the two-component address type, see [`BoseFS2C`](@ref)
- `h_a::BoseHubbardReal1D` and `h_b::BoseHubbardReal1D`: standard Hamiltonian for boson A and B, see [`BoseHubbardReal1D`](@ref)
- `v`: the inter-species interaction parameter V

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" BoseHubbardReal1D2C

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardReal1D2C{<:Real}}) = HermitianLO()

function BoseHubbardReal1D2C(add::BoseFS2C; ua::T=1.0,ub::T=1.0,ta::T=1.0,tb::T=1.0,v::T=1.0) where T
    ha = BoseHubbardReal1D(add.bsa;u=ua,t=ta)
    hb = BoseHubbardReal1D(add.bsb;u=ub,t=tb)
    return BoseHubbardReal1D2C{T,BoseHubbardReal1D{T},BoseHubbardReal1D{T},v}(ha,hb)
end

# number of excitations that can be made
function numOfHops(ham::BoseHubbardReal1D2C, add)
    return 2*(numberoccupiedsites(add.bsa)+numberoccupiedsites(add.bsb))
end

function bosehubbard2Cinteraction(add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    c1 = onr(add.bsa)
    c2 = onr(add.bsb)
    interaction = 0::Int
    for site = 1:M
        if !iszero(c2[site])
            interaction += c2[site]*c1[site]
        end
    end
    return interaction
end

function diagME(ham::BoseHubbardReal1D2C{T,HA,HB,V}, address::BoseFS2C) where {T,HA,HB,V}
    return ham.ha.u * bosehubbardinteraction(address.bsa) / 2 + ham.hb.u * bosehubbardinteraction(address.bsb) / 2 + V * bosehubbard2Cinteraction(address)
end

function hop(ham::BoseHubbardReal1D2C, add, chosen::Integer)
    nhops = numOfHops(ham,add)
    nhops_a = 2*numberoccupiedsites(add.bsa)
    if chosen in 1:nhops_a
        naddress_from_bsa, onproduct = hopnextneighbour(add.bsa, chosen)
        elem = - ham.ha.t*sqrt(onproduct)
        return BoseFS2C(naddress_from_bsa,add.bsb), elem
    elseif chosen in nhops_a+1:nhops
        chosen -= nhops_a
        naddress_from_bsb, onproduct = hopnextneighbour(add.bsb, chosen)
        elem = - ham.hb.t*sqrt(onproduct)
        return BoseFS2C(add.bsa,naddress_from_bsb), elem
    end
    # return new address and matrix element
end
