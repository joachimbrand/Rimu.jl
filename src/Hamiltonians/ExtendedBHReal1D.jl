@with_kw struct ExtendedBHReal1D{T} <: BosonicHamiltonian{T}
  n::Int = 6    # number of bosons
  m::Int = 6    # number of lattice sites
  u::T = 1.0    # on-site interaction strength
  v::T = 1.0    # on-site interaction strength
  t::T = 1.0    # hopping strength
  AT::Type = BSAdd64 # address type
end
@doc """
    ham = ExtendedBHReal1D(n=6, m=6, u=1.0, v=1.0, t=1.0, AT=BSAdd64)

Implements the extended Bose Hubbard model on a one-dimensional chain
in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1) + v \\sum_{\\langle i,j\\rangle} n_i n_j
```

# Arguments
- `n::Int`: number of bosons
- `m::Int`: number of lattice sites
- `u::Float64`: on-site interaction parameter
- `v::Float64`: the next-neighbor interaction
- `t::Float64`: the hopping strength
- `AT::Type`: address type for identifying configuration
""" ExtendedBHReal1D

# set the `LOStructure` trait
LOStructure(::Type{ExtendedBHReal1D{T}}) where T <: Real = HermitianLO()

"""
    ExtendedBHReal1D(add::AbstractFockAddress; u=1.0, v=1.0 t=1.0)
Set up the `BoseHubbardReal1D` with the correct particle and mode number and
address type. Parameters `u` and `t` can be passed as keyword arguments.
"""
function ExtendedBHReal1D(add::BSA; u=1.0, v=1.0, t=1.0) where BSA <: AbstractFockAddress
  n = num_particles(add)
  m = num_modes(add)
  return ExtendedBHReal1D(n,m,u,v,t,BSA)
end

# functor definitions need to be done separately for each concrete type
function (h::ExtendedBHReal1D)(s::Symbol)
    if s == :dim # attempt to compute dimension as `Int`
        return hasIntDimension(h) ? dimensionLO(h) : nothing
    elseif s == :fdim
        return fDimensionLO(h) # return dimension as floating point
    end
    return nothing
end

function diagME(h::ExtendedBHReal1D, address)
  ebhinteraction, bhinteraction= ebhm(address, h.m)
  return h.u * bhinteraction / 2 + h.v * ebhinteraction
end

# The off-diagonal matrix elements of the 1D Hubbard chain are the same for
# the extended and original Bose-Hubbard model.

"""
    ebhm(address, m)

Compute the on-site product sum_j n_j(n_j-1) and the next neighbour term
sum_j n_j n_{j+1} with periodic boundary conditions.
"""
function ebhm(address, mModes)
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
  for i=1:mModes-1
    #println("i mModes= ",i)
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
  #end
  return ebhmmatrixelementint , bhmmatrixelementint
end #ebhm

ebhm(b::BoseFS, m) = ebhm(b.bs, m)
ebhm(b::BoseFS{N,M,A})  where {N,M,A} = ebhm(b, M)
