"""
    HubbardReal1D(address; u=1.0, t=1.0)

Implements a one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1)
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.

# See also

* [`HubbardMom1D`](@ref)
* [`ExtendedHubbardReal1D`](@ref)

"""
struct HubbardReal1D{TT,A<:AbstractFockAddress,U,T} <: AbstractHamiltonian{TT}
    add::A
end

function HubbardReal1D(addr; u=1.0, t=1.0)
    U, T = promote(float(u), float(t))
    return HubbardReal1D{typeof(U),typeof(addr),U,T}(addr)
end

function Base.show(io::IO, h::HubbardReal1D)
    print(io, "HubbardReal1D($(h.add); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardReal1D)
    return getfield(h, :add)
end

LOStructure(::Type{<:HubbardReal1D{<:Real}}) = Hermitian()

Base.getproperty(h::HubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T
Base.getproperty(h::HubbardReal1D, ::Val{:add}) = getfield(h, :add)

function num_offdiagonals(::HubbardReal1D, address::BoseFS)
    return 2 * numberoccupiedsites(address)
end

"""
    bose_hubbard_interaction(address)

Return Σ_i *n_i* (*n_i*-1) for computing the Bose-Hubbard on-site interaction (without the
*U* prefactor.)

# Example

```jldoctest
julia> Hamiltonians.bose_hubbard_interaction(BoseFS{4,4}((2,1,1,0)))
2
julia> Hamiltonians.bose_hubbard_interaction(BoseFS{4,4}((3,0,1,0)))
6
```
"""
function bose_hubbard_interaction(b::BoseFS{<:Any,<:Any,A}) where A
    return bose_hubbard_interaction(Val(num_chunks(A)), b)
end

@inline function bose_hubbard_interaction(_, b::BoseFS)
    result = 0
    for (n, _, _) in occupied_orbitals(b)
        result += n * (n - 1)
    end
    return result
end

@inline function bose_hubbard_interaction(::Val{1}, b::BoseFS)
    # currently this ammounts to counting occupation numbers of orbitals
    chunk = chunks(b.bs)[1]
    matrixelementint = 0
    while !iszero(chunk)
        chunk >>>= (trailing_zeros(chunk) % UInt) # proceed to next occupied orbital
        bosonnumber = trailing_ones(chunk) # count how many bosons inside
        # surpsingly it is faster to not check whether this is nonzero and do the
        # following operations anyway
        chunk >>>= (bosonnumber % UInt) # remove the counted orbital
        matrixelementint += bosonnumber * (bosonnumber - 1)
    end
    return matrixelementint
end

function diagonal_element(h::HubbardReal1D, address::BoseFS)
    h.u * bose_hubbard_interaction(address) / 2
end

function get_offdiagonal(h::HubbardReal1D, add::BoseFS, chosen)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - h.t * sqrt(onproduct)
end
