"""
    HubbardReal1D(;[u=1.0, t=1.0])

Implements a one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1)
```

# Arguments
- `u`: the interaction parameter
- `t`: the hopping strength

"""
struct HubbardReal1D{TT,U,T,A} <: AbstractHamiltonian{TT}
    add::A
end

# addr for compatibility.
function HubbardReal1D(addr; u=1.0, t=1.0)
    U, T = promote(u, t)
    return HubbardReal1D{typeof(U),U,T,typeof(addr)}(addr)
end

LOStructure(::Type{<:HubbardReal1D{<:Real}}) = HermitianLO()

Base.getproperty(h::HubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardReal1D{<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,T}, ::Val{:t}) where T = T

function diagME(h::HubbardReal1D, address::BoseFS)
    h.u * bosehubbardinteraction(address) / 2
end

function numOfHops(::HubbardReal1D, address::BoseFS)
    return numberlinkedsites(address)
end

function hop(h::HubbardReal1D, add::BoseFS, chosen)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - h.t * sqrt(onproduct)
end
