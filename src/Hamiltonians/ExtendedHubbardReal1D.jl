"""
    ExtendedHubbardReal1D(address; u=1.0, v=1.0, t=1.0)

Implements the extended Hubbard model on a one-dimensional chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1) + v \\sum_{\\langle i,j\\rangle} n_i n_j
```

# Parameters
- `u`: on-site interaction parameter
- `v`: the next-neighbor interaction
- `t`: the hopping strength
"""
struct ExtendedHubbardReal1D{TT,A<:AbstractFockAddress,U,V,T} <: AbstractHamiltonian{TT}
    add::A
end

# addr for compatibility.
function ExtendedHubbardReal1D(addr; u=1.0, v=1.0, t=1.0)
    U, V, T = promote(float(u), float(v), float(t))
    return ExtendedHubbardReal1D{typeof(U),typeof(addr),U,V,T}(addr)
end

function Base.show(io::IO, h::ExtendedHubbardReal1D)
    print(io, "ExtendedHubbardReal1D($(h.add); u=$(h.u), v=$(h.v), t=$(h.t))")
end

function starting_address(h::ExtendedHubbardReal1D)
    return getfield(h, :add)
end

LOStructure(::Type{<:ExtendedHubbardReal1D{<:Real}}) = HermitianLO()

Base.getproperty(h::ExtendedHubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::ExtendedHubbardReal1D, ::Val{:add}) = getfield(h, :add)
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,V}, ::Val{:v}) where V = V
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T

function diagME(h::ExtendedHubbardReal1D, b::BoseFS)
    ebhinteraction, bhinteraction = extended_bose_hubbard_interaction(b)
    return h.u * bhinteraction / 2 + h.v * ebhinteraction
end

function numOfHops(::ExtendedHubbardReal1D, address::BoseFS)
    return numberlinkedsites(address)
end

function hop(h::ExtendedHubbardReal1D, add::BoseFS, chosen)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - h.t * sqrt(onproduct)
end

"""
    extended_bose_hubbard_interaction(address, m)

Compute the on-site product sum_j n_j(n_j-1) and the next neighbour term
sum_j n_j n_{j+1} with periodic boundary conditions.
"""
function extended_bose_hubbard_interaction(b::BoseFS{<:Any,M}) where M
    address = b.bs
    # compute the diagonal matrix element of the Extended Bose Hubbard Hamiltonian
    # currently this ammounts to counting occupation numbers of orbitals
    #println("adress= ", bin(address))
    #if periodicboundericondition
    ## only periodic boundary conditions are implemented so far
    bhmmatrixelementint = 0
    ebhmmatrixelementint = 0
    bosonnumber2=0
    #address >>>= trailing_zeros(address) # proceed to next occupied orbital
    bosonnumber1 = trailing_ones(address) # count how many bosons inside
    # surpsingly it is faster to not check whether this is nonzero and do the
    # following operations anyway
    bhmmatrixelementint+= bosonnumber1 * (bosonnumber1-1)
    firstbosonnumber = bosonnumber1 #keap on memory the boson number of the first
    #to do the calculation with the last boson
    address >>>= bosonnumber1 # remove the countedorbital
    address >>>= 1
    for i=1:M-1
        # proceed to next occupied orbital
        bosonnumber2 = trailing_ones(address) # count how many bosons inside
        # surpsingly it is faster to not check whether this is nonzero and do the
        # following operations anyway
        address >>>= bosonnumber2 # remove the countedorbital
        ebhmmatrixelementint += bosonnumber2 * (bosonnumber1)
        bhmmatrixelementint+= bosonnumber2 * (bosonnumber2-1)
        bosonnumber1=bosonnumber2
        address >>>= 1
    end
    ebhmmatrixelementint+= bosonnumber2 * firstbosonnumber  #periodic bondary condition
    return ebhmmatrixelementint , bhmmatrixelementint
end
