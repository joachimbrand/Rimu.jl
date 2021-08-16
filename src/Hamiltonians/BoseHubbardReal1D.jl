@with_kw struct BoseHubbardReal1D{T} <: AbstractHamiltonian{T}
    n::Int = 6    # number of bosons
    m::Int = 6    # number of lattice sites
    u::T = 1.0    # interaction strength
    t::T = 1.0    # hopping strength
    AT::Type = BSAdd64 # address type
end

@doc """
    ham = BoseHubbardReal1D(;[n=6, m=6, u=1.0, t=1.0, AT = BSAdd64])

Implements a one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1)
```

!!! warning
    This Hamiltonian is deprecated. Please use [`HubbardReal1D`](@ref) instead.

# Arguments
- `n::Int`: the number of bosons
- `m::Int`: the number of lattice sites
- `u::Float64`: the interaction parameter
- `t::Float64`: the hopping strength
- `AT::Type`: the address type

# Functor use:
    w = ham(v)
    ham(w, v)
Compute the matrix - vector product `w = ham * v`. The two-argument version is
mutating for `w`.

""" BoseHubbardReal1D

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardReal1D{T}}) where T <: Real = Hermitian()

"""
    BoseHubbardReal1D(add::AbstractFockAddress; u=1.0, t=1.0)
Set up the `BoseHubbardReal1D` with the correct particle and mode number and
address type. Parameters `u` and `t` can be passed as keyword arguments.
"""
function BoseHubbardReal1D(add::BSA; u=1.0, t=1.0) where BSA <: AbstractFockAddress
    n = num_particles(add)
    m = num_modes(add)
    return BoseHubbardReal1D(n,m,u,t,BSA)
end

function starting_address(h::BoseHubbardReal1D)
    return near_uniform(h.AT)
end

function diagonal_element(h::BoseHubbardReal1D, address)
    h.u * bose_hubbard_interaction(address) / 2
end

function num_offdiagonals(ham::BoseHubbardReal1D, add)
    return 2 * num_occupied_modes(add)
end

function get_offdiagonal(ham::BoseHubbardReal1D, add, chosen::Integer)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - ham.t*sqrt(onproduct)
    # return new address and matrix element
end
