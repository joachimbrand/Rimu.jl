"""
    BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0, kwargs...)

Implements a one-dimensional Bose Hubbard chain in momentum space with a two-component
Bose gas.

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
Further keyword arguments are passed on to the constructor of [`HubbardMom1D`](@ref).

# See also

* [`BoseFS2C`](@ref)
* [`BoseHubbardReal1D2C`](@ref)

"""
struct BoseHubbardMom1D2C{T,HA,HB,V} <: TwoComponentHamiltonian{T}
    ha::HA
    hb::HB
end

function BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0, ub=1.0, ta=1.0, tb=1.0, v=1.0, args...)
    ha = HubbardMom1D(add.bsa; u=ua, t=ta, args...)
    hb = HubbardMom1D(add.bsb; u=ub, t=tb, args...)
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

dimension(::BoseHubbardMom1D2C, address) = number_conserving_dimension(address)

function starting_address(h::BoseHubbardMom1D2C)
    return BoseFS2C(starting_address(h.ha), starting_address(h.hb))
end

LOStructure(::Type{<:BoseHubbardMom1D2C{<:Real}}) = IsHermitian()

Base.getproperty(h::BoseHubbardMom1D2C, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::BoseHubbardMom1D2C, ::Val{:ha}) = getfield(h, :ha)
Base.getproperty(h::BoseHubbardMom1D2C, ::Val{:hb}) = getfield(h, :hb)
Base.getproperty(h::BoseHubbardMom1D2C{<:Any,<:Any,<:Any,V}, ::Val{:v}) where {V} = V

function num_offdiagonals(ham::BoseHubbardMom1D2C, add::BoseFS2C)
    M = num_modes(add)
    sa = num_occupied_modes(add.bsa)
    sb = num_occupied_modes(add.bsb)
    return num_offdiagonals(ham.ha, add.bsa) + num_offdiagonals(ham.hb, add.bsb) + sa*(M-1)*sb
    # number of excitations that can be made
end

function diagonal_element(ham::BoseHubbardMom1D2C, add::BoseFS2C)
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
    return (
        diagonal_element(ham.ha, add.bsa) +
        diagonal_element(ham.hb, add.bsb) +
        ham.v/M*interaction2c
    )
end

function get_offdiagonal(ham::BoseHubbardMom1D2C, add::BoseFS2C, chosen)
    return offdiagonals(ham, add)[chosen]
end

"""
    OffdiagonalsBoseMom1D2C

Specialized [`AbstractOffdiagonals`](@ref) that keep track of number of off-diagonals and
number of occupied sites in both components of the address.
"""
struct OffdiagonalsBoseMom1D2C{
    A<:BoseFS2C,T,H<:TwoComponentHamiltonian{T},O1,O2,M1,M2
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    offdiags_a::O1
    offdiags_b::O2
    map_a::M1
    map_b::M2
end

function offdiagonals(h::BoseHubbardMom1D2C{T}, a::BoseFS2C) where {T}
    offdiags_a = offdiagonals(h.ha, a.bsa)
    offdiags_b = offdiagonals(h.hb, a.bsb)
    map_a = OccupiedModeMap(a.bsa)
    map_b = OccupiedModeMap(a.bsb)
    occ_a = length(map_a)
    occ_b = length(map_b)
    len = length(offdiags_a) + length(offdiags_b) + occ_a * (num_modes(a) - 1) * occ_b

    return OffdiagonalsBoseMom1D2C(h, a, len, offdiags_a, offdiags_b, map_a, map_b)
end

function Base.getindex(s::OffdiagonalsBoseMom1D2C{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    num_hops_a = length(s.offdiags_a)
    num_hops_b = length(s.offdiags_b)
    if i ≤ num_hops_a
        new_a, matrix_element = s.offdiags_a[i]
        new_add = A(new_a, s.address.bsb)
    elseif i ≤ num_hops_a + num_hops_b
        i -= num_hops_a
        new_b, matrix_element = s.offdiags_b[i]
        new_add = A(s.address.bsa, new_b)
    else
        i -= num_hops_a + num_hops_b
        new_a, new_b, val = momentum_transfer_excitation(
            s.address.bsa, s.address.bsb, i, s.map_a, s.map_b
        )
        new_add = A(new_a, new_b)
        matrix_element = s.hamiltonian.v/num_modes(new_add) * val
    end
    return new_add, matrix_element
end

Base.size(s::OffdiagonalsBoseMom1D2C) = (s.length,)
