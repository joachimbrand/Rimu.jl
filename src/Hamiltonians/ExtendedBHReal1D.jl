@with_kw struct ExtendedBHReal1D{T} <: AbstractHamiltonian{T}
    n::Int = 6    # number of bosons
    m::Int = 6    # number of lattice sites
    u::T = 1.0    # on-site interaction strength
    v::T = 1.0    # on-site interaction strength
    t::T = 1.0    # hopping strength
    AT::Type = BoseFS{6,6} # address type
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

function starting_address(h::ExtendedBHReal1D)
    return nearUniform(h.AT)
end

function diagME(h::ExtendedBHReal1D, b)
    ebhinteraction, bhinteraction = extended_bose_hubbard_interaction(b)
    return h.u * bhinteraction / 2 + h.v * ebhinteraction
end

function numOfHops(ham::ExtendedBHReal1D, add)
    return 2 * numberoccupiedsites(add)
end

function hop(ham::ExtendedBHReal1D, add, chosen::Integer)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - ham.t*sqrt(onproduct)
    # return new address and matrix element
end
