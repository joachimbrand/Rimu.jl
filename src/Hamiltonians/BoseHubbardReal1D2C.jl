"""
    BoseHubbardReal1D2C(address::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + V\\sum_{i} n_{a_i}n_{b_i}
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `ua`: the on-site interaction parameter parameter for Hamiltonian a.
* `ub`: the on-site interaction parameter parameter for Hamiltonian b.
* `ta`: the hopping strength for Hamiltonian a.
* `tb`: the hopping strength for Hamiltonian b.
* `v`: the inter-species interaction parameter V.

# See also

* [`HubbardReal1D`](@ref)
* [`BoseHubbardMom1D2C`](@ref)

"""
struct BoseHubbardReal1D2C{T,HA,HB,V} <: TwoComponentHamiltonian{T}
    ha::HA
    hb::HB
end

function BoseHubbardReal1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)
    ha = HubbardReal1D(add.bsa; u=ua, t=ta)
    hb = HubbardReal1D(add.bsb; u=ub, t=tb)
    T = promote_type(eltype(ha), eltype(hb))
    return BoseHubbardReal1D2C{T,typeof(ha),typeof(hb),v}(ha, hb)
end

function Base.show(io::IO, h::BoseHubbardReal1D2C)
    addr = starting_address(h)
    ua = h.ha.u
    ub = h.hb.u
    ta = h.ha.t
    tb = h.hb.t
    v = h.v
    print(io, "BoseHubbardReal1D2C($addr; ua=$ua, ub=$ub, ta=$ta, tb=$tb, v=$v)")
end

function starting_address(h::BoseHubbardReal1D2C)
    return BoseFS2C(starting_address(h.ha), starting_address(h.hb))
end

LOStructure(::Type{<:BoseHubbardReal1D2C{<:Real}}) = IsHermitian()

Base.getproperty(h::BoseHubbardReal1D2C, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::BoseHubbardReal1D2C, ::Val{:ha}) = getfield(h, :ha)
Base.getproperty(h::BoseHubbardReal1D2C, ::Val{:hb}) = getfield(h, :hb)
Base.getproperty(h::BoseHubbardReal1D2C{<:Any,<:Any,<:Any,V}, ::Val{:v}) where {V} = V

# number of excitations that can be made
function num_offdiagonals(ham::BoseHubbardReal1D2C, add)
    return 2*(num_occupied_modes(add.bsa) + num_occupied_modes(add.bsb))
end

"""
    bose_hubbard_2c_interaction(::BoseFS2C)

Compute the interaction between the two components.
"""
function bose_hubbard_2c_interaction(add::BoseFS2C)
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

function diagonal_element(ham::BoseHubbardReal1D2C, address::BoseFS2C)
    return (
        diagonal_element(ham.ha, address.bsa) +
        diagonal_element(ham.hb, address.bsb) +
        ham.v * bose_hubbard_2c_interaction(address)
    )
end

function get_offdiagonal(ham::BoseHubbardReal1D2C, add, chosen)
    nhops = num_offdiagonals(ham,add)
    nhops_a = 2 * num_occupied_modes(add.bsa)
    if chosen ≤ nhops_a
        naddress_from_bsa, onproduct = hopnextneighbour(add.bsa, chosen)
        elem = - ham.ha.t * onproduct
        return BoseFS2C(naddress_from_bsa,add.bsb), elem
    else
        chosen -= nhops_a
        naddress_from_bsb, onproduct = hopnextneighbour(add.bsb, chosen)
        elem = -ham.hb.t * onproduct
        return BoseFS2C(add.bsa,naddress_from_bsb), elem
    end
    # return new address and matrix element
end

struct OffdiagonalsBoseReal1D2C{
    A<:BoseFS2C,T,H<:TwoComponentHamiltonian{T}
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    num_hops_a::Int
end

function offdiagonals(h::BoseHubbardReal1D2C, a::BoseFS2C)
    hops_a = num_offdiagonals(h.ha, a.bsa)
    hops_b = num_offdiagonals(h.hb, a.bsb)
    length = hops_a + hops_b

    return OffdiagonalsBoseReal1D2C(h, a, length, hops_a)
end

function Base.getindex(s::OffdiagonalsBoseReal1D2C{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    if i ≤ s.num_hops_a
        new_a, matrix_element = get_offdiagonal(s.hamiltonian.ha, s.address.bsa, i)
        new_add = A(new_a, s.address.bsb)
    else
        i -= s.num_hops_a
        new_b, matrix_element = get_offdiagonal(s.hamiltonian.hb, s.address.bsb, i)
        new_add = A(s.address.bsa, new_b)
    end
    return new_add, matrix_element
end

Base.size(s::OffdiagonalsBoseReal1D2C) = (s.length,)
