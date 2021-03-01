@with_kw struct BoseHubbardReal1D{T} <: BosonicHamiltonian{T}
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

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" BoseHubbardReal1D

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardReal1D{T}}) where T <: Real = HermitianLO()

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

# functor definitions need to be done separately for each concrete type
function (h::BoseHubbardReal1D)(s::Symbol)
    if s == :dim # attempt to compute dimension as `Int`
        return hasIntDimension(h) ? dimensionLO(h) : nothing
    elseif s == :fdim
        return fDimensionLO(h) # return dimension as floating point
    end
    return nothing
end

# """
#     setupBoseHubbardReal1D(; n, m, u, t, [AT = BoseFS, genInitialONR = nearUniform])
#     -> ham::BoseHubbardReal1D, address::AT
# Set up the Hamiltonian `ham` and initial address `address` for the Bose Hubbard
# model with the given parameters as keyword arguments, see
# [`BoseHubbardReal1D`](@ref). For `AT` pass an address type (or suitable
# constructor) and for `genInitialONR` a function that takes `n` and `m` as
# arguments and returns an occupation number representation, see
# [`nearUniform()`](@ref):
#
# `onr = genInitialONR(n,m)`
# """
# function setupBoseHubbardReal1D(;n::Int, m::Int, u, t,
#                     AT = BoseFS, genInitialONR = nearUniform)
#     address = AT(genInitialONR(n,m))
#     ham = BoseHubbardReal1D(n = n,
#                             m = m,
#                             u = u,
#                             t = t,
#                             AT = typeof(address)
#     )
#     return ham, address
# end

"""
    diagME(ham, add)

Compute the diagonal matrix element of the linear operator `ham` at
address `add`.
"""
function diagME(h::BoseHubbardReal1D, address)
    h.u * bosehubbardinteraction(address) / 2
end
