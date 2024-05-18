"""
    local_interaction(::AbstractFockAddress, u)
    local_interaction(::AbstractFockAddress, ::AbstractFockAddress, v)

Return the sum of (mode-wise) local interactions ``\\frac{u}{2} \\sum_i n_i(n_i-1)`` of a
single component Fock state, or ``v \\sum_i n_{↑,i} n_{↓,i}`` between two Fock states. For a
multi-component Fock state, return the eigenvalue of

```math
\\frac{1}{2}\\sum_{i, σ, τ} u_{σ,τ} a^†_{σ,i}a^†_{τ,i}a^†_{τ,i}a^†_{σ,i} ,
```

where `u::SMatrix` is a symmetric matrix of interaction constants, `i` is a mode index,
and `σ`, `τ` are component indices.

See also [`BoseFS`](@ref), [`FermiFS`](@ref), [`CompositeFS`](@ref).
"""
local_interaction(b::SingleComponentFockAddress, u) = u * bose_hubbard_interaction(b) / 2
local_interaction(f::FermiFS, _) = 0
function local_interaction(a::SingleComponentFockAddress, b::SingleComponentFockAddress, u)
    return u * dot(occupied_modes(a), occupied_modes(b))
end
function local_interaction(fs::CompositeFS, u)
    return _interactions(fs.components, u)
end

"""
    _interaction_col(a, bs::Tuple, us::Tuple)

Sum the local interactions of the Fock state `a` with all states in `bs` using the
interaction constants in `us`. This is used to compute all interactions in the column
below the diagonal of the interaction matrix.
"""
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...))
    return local_interaction(a, b, u) + _interaction_col(a, bs, us)
end

"""
    _interactions(addresses, interaction_matrix)

Compute all pairwise interactions in a tuple of `addresses`. The `interaction_matrix` sets the
intraction strengths.

The code is equivalent to the following.

```julia
acc = 0.0
for (i, a) in enumerate(addresses)
    acc += local_interaction(a, interaction_matrix[i, i])
    for (j, b) in enumerate(addresses[i+1:end])
        acc += local_interaction(a, b, interaction_matrix[i, j])
    end
end
return acc
```

It is implemented recursively to ensure type stability.
"""
@inline _interactions(::Tuple{}, ::SMatrix{0,0}) = 0.0
@inline function _interactions(
    (a, as...)::NTuple{N,AbstractFockAddress}, m::SMatrix{N,N}
) where {N}
    # Split the matrix into the column we need now, and the rest.
    (u, column...) = Tuple(m[:, 1])
    # Type-stable way to subset SMatrix:
    rest = SMatrix{N-1,N-1}(view(m, 2:N, 2:N))

    # Get the self-interaction first.
    self = local_interaction(a, u)
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, column)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, rest)
end

"""
    external_potential(add::AbstractFockAddress, pot)

Calculate the value of a diagonal single particle operator (e.g. a trap potential) at
the address `add`.
```math
\\sum_{iσ} v_{iσ} n_{iσ}
```
The (precomputed) potential energy per particle at each mode passed as `pot` should be
a length `M` vector for a [`SingleComponentFockAddress`](@ref), or a `M×C` matrix for
a [`CompositeFS `](@ref), where `M` is the number of modes and `C` the number of
components.
"""
Base.@propagate_inbounds function external_potential(add::SingleComponentFockAddress, pot)
    pe = 0.0
    @boundscheck checkbounds(pot, 1:num_modes(add))
    for (n,i) in occupied_modes(add)
        pe += n * pot[i]
    end
    return pe
end

function external_potential(add::CompositeFS, pot::Matrix)
    pe = 0.0
    @boundscheck checkbounds(pot, 1:num_modes(add), 1:num_components(add))
    for (i,c) in enumerate(add.components)
        @inbounds pe += external_potential(c, @view pot[:,i])
    end
    return pe
end

###
### HubbardRealSpace
###
"""
    HubbardRealSpace(address; geometry=PeriodicBoundaries(M,), t=ones(C), u=ones(C, C), v=zeros(C, D))

Hubbard model in real space. Supports single or multi-component Fock state
addresses (with `C` components) and various (rectangular) lattice geometries
in `D` dimensions.

```math
  \\hat{H} = -\\sum_{\\langle i,j\\rangle,σ} t_σ a^†_{iσ} a_{jσ} +
  \\frac{1}{2}\\sum_{i,σ} u_{σσ} n_{iσ} (n_{iσ} - 1) +
  \\sum_{i,σ≠τ}u_{στ} n_{iσ} n_{iτ}
```

If `v` is nonzero then this calculates ``\\hat{H} + \\hat{V}`` by adding the
harmonic trapping potential
```math
    \\hat{V} = \\sum_{i,σ,d} v_{σd} x_{di}^2 n_{iσ}
```
where ``x_{di}`` is the distance of site ``i`` from the centre of the trap
along dimension ``d``.

## Address types

* [`BoseFS`](@ref): Single-component Bose-Hubbard model.
* [`FermiFS`](@ref): Single-component Fermi-Hubbard model.
* [`CompositeFS`](@ref): For multi-component models.

Note that a single component of fermions cannot interact with itself. A warning
is produced if `address`is incompatible with the interaction parameters `u`.

## Geometries

Implemented [`CubicGrid`](@ref)s for keyword `geometry`

* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)

Default is `geometry=PeriodicBoundaries(M,)`, i.e. a one-dimensional lattice with the
number of sites `M` inferred from the number of modes in `address`.

## Other parameters

* `t`: the hopping strengths. Must be a vector of length `C`. The `i`-th element of the
  vector corresponds to the hopping strength of the `i`-th component.
* `u`: the on-site interaction parameters. Must be a symmetric matrix. `u[i, j]`
  corresponds to the interaction between the `i`-th and `j`-th component. `u[i, i]`
  corresponds to the interaction of a component with itself. Note that `u[i,i]` must
  be zero for fermionic components.
* `v`: the trap potential strengths. Must be a matrix of size `C × D`. `v[i,j]` is
  the strength of the trap for component `i` in the `j`th dimension.
"""
struct HubbardRealSpace{
    C, # components
    A<:AbstractFockAddress,
    G<:CubicGrid,
    D, # dimension
    # The following need to be type params.
    T<:SVector{C,Float64},
    U<:Union{SMatrix{C,C,Float64},Nothing},
    V<:Union{SMatrix{C,D,Float64},Nothing},
    P<:Union{Matrix{Float64},Nothing}
} <: AbstractHamiltonian{Float64}
    address::A
    t::T # hopping strengths
    u::U # interactions
    v::V # trap strengths
    potential::P # potential energy of each component at each lattice site
    geometry::G
end

function HubbardRealSpace(
    address::AbstractFockAddress;
    geometry::CubicGrid=PeriodicBoundaries((num_modes(address),)),
    t=ones(num_components(address)),
    u=ones(num_components(address), num_components(address)),
    v=zeros(num_components(address), num_dimensions(geometry))
)
    C = num_components(address)
    D = num_dimensions(geometry)
    S = size(geometry)

    # Sanity checks
    if prod(size(geometry)) ≠ num_modes(address)
        throw(ArgumentError("`geometry` does not have the correct number of sites"))
    elseif length(u) ≠ 1 && !issymmetric(u)
        throw(ArgumentError("`u` must be symmetric"))
    elseif length(u) ≠ C * C
        throw(ArgumentError("`u` must be a $C × $C matrix"))
    elseif size(t) ≠ (C,)
        throw(ArgumentError("`t` must be a vector of length $C"))
    elseif length(v) ≠ C * D
        throw(ArgumentError("`v` must be a $C × $D matrix"))
    elseif address isa BoseFS2C
        throw(ArgumentError(
            "`BoseFS2C` is not supported for this Hamiltonian, use `CompositeFS`"
        ))
    end
    warn_fermi_interaction(address, u)

    t_vec = SVector{C,Float64}(t)
    u_mat = iszero(u) ? nothing : SMatrix{C,C,Float64}(u)

    # Precompute the trap potential terms
    if iszero(v)
        v_mat = nothing
        pot_vec = nothing
    else
        v_mat = SMatrix{C,D,Float64}(v)
        ranges = Tuple(range(-fld(M,2); length=M) for M in S)
        x_sq = map(x -> Tuple(x).^2, CartesianIndices(ranges))
        pot_vec = zeros(prod(S), C) # or undef...
        for c in 1:C
            pot_vec[:,c] .= vec(map(x -> sum(v_mat[c,:] .* x), x_sq))
        end
    end

    return HubbardRealSpace{C,typeof(address),typeof(geometry),D,typeof(t_vec),typeof(u_mat),typeof(v_mat),typeof(pot_vec)}(
        address, t_vec, u_mat, v_mat, pot_vec, geometry,
    )
end

"""
    warn_fermi_interaction(address, u)

Warn if interaction matrix `u` does not make sense for `address`.
"""
function warn_fermi_interaction(address::CompositeFS, u)
    C = num_components(address)
    for c in 1:C
        if address.components[c] isa FermiFS && u ≠ ones(C,C) && u[c,c] ≠ 0
            @warn "component $(c) is fermionic, but was given a self-interaction " *
                "strength of $(u[c,c])" maxlog=1
        end
    end
end
function warn_fermi_interaction(address::FermiFS, u)
    if u ≠ ones(1, 1) && u[1, 1] ≠ 0
        @warn "address is fermionic, but was given a self-interaction " *
            "strength of $(u[1,1])" maxlog=1
    end
end
warn_fermi_interaction(_, _) = nothing

LOStructure(::Type{<:HubbardRealSpace}) = IsHermitian()

function Base.show(io::IO, h::HubbardRealSpace{C}) where C
    io = IOContext(io, :compact => true)
    println(io, "HubbardRealSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  t = ", Float64.(h.t), ",")
    if isnothing(h.u)
        println(io, "  u = ", zeros(C,C), ",")
    else
        println(io, "  u = ", Float64.(h.u), ",")
    end
    !isnothing(h.v) && println(io, "  v = ", Float64.(h.v), ",")
    println(io, ")")
end

# Overload equality due to stored potential energy arrays.
Base.:(==)(H::HubbardRealSpace, G::HubbardRealSpace) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(h::HubbardRealSpace) = h.address

dimension(::HubbardRealSpace, address) = number_conserving_dimension(address)

function diagonal_element(h::HubbardRealSpace, address)
    int = isnothing(h.u) ? 0.0 : local_interaction(address, h.u)
    pot = isnothing(h.v) ? 0.0 : external_potential(address, h.potential)
    return int + pot
end
function diagonal_element(h::HubbardRealSpace{1}, address)
    int = isnothing(h.u) ? 0.0 : local_interaction(address, h.u[1])
    pot = if isnothing(h.v)
            0.0
        else
            @boundscheck checkbounds(h.potential, 1:num_modes(address), 1)
            @inbounds external_potential(address, @view h.potential[:,1])
        end
    return int + pot
end

###
### Offdiagonals
###
# This may be an inefficient implementation, but it is not actually used anywhere in the
# main algorithm.
get_offdiagonal(h::HubbardRealSpace, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::HubbardRealSpace, add) = length(offdiagonals(h, add))

"""
    HubbardRealSpaceCompOffdiagonals{G,A} <: AbstractOffdiagonals{A,Float64}

Offdiagonals for a single address component. Used with [`HubbardRealSpace`](@ref) model
with a single-component address, or a component of a [`CompositeFS`](@ref).
"""
struct HubbardRealSpaceCompOffdiagonals{G,A} <: AbstractOffdiagonals{A,Float64}
    geometry::G
    address::A
    t::Float64
    length::Int
end

function offdiagonals(h::HubbardRealSpace, comp, add)
    neighbours = 2 * num_dimensions(h.geometry)
    return HubbardRealSpaceCompOffdiagonals(
        h.geometry, add, h.t[comp], num_occupied_modes(add) * neighbours
    )
end

Base.size(o::HubbardRealSpaceCompOffdiagonals) = (o.length,)

@inline function Base.getindex(o::HubbardRealSpaceCompOffdiagonals, chosen)
    neighbours = 2 * num_dimensions(o.geometry)
    particle, neigh = fldmod1(chosen, neighbours)
    src_index = find_occupied_mode(o.address, particle)
    neigh = neighbor_site(o.geometry, src_index.mode, neigh)

    if neigh == 0
        return o.address, 0.0
    else
        dst_index = find_mode(o.address, neigh)
        new_add, value = excitation(o.address, (dst_index,), (src_index,))
        return new_add, -o.t * value
    end
end

# For simple models with one component.
offdiagonals(h::HubbardRealSpace{1,A}, add::A) where {A} = offdiagonals(h, 1, add)

# Multi-component part
"""
    HubbardRealSpaceOffdiagonals{A,T<:Tuple} <: AbstractOffdiagonals{A,Float64}

Offdiagonals of a [`HubbardRealSpace`](@ref) model with a [`CompositeFS`](@ref) address.
"""
struct HubbardRealSpaceOffdiagonals{A,T<:Tuple} <: AbstractOffdiagonals{A,Float64}
    address::A
    parts::T
    length::Int
end

"""
    get_comp_offdiags(h::HubbardRealSpace, add)

Get offdiagonals of all components of address in a type-stable manner.
"""
@inline function get_comp_offdiags(h::HubbardRealSpace, address)
    return _get_comp_offdiags(address.components, h, Val(1))
end

# All steps of recursive function (should) get inlined, creating a type-stable tuple of
# offdiagonals.
@inline function _get_comp_offdiags((a,as...), h, ::Val{I}) where {I}
    return (offdiagonals(h, I, a), _get_comp_offdiags(as, h, Val(I+1))...)
end
@inline _get_comp_offdiags(::Tuple{}, h, ::Val) = ()

function offdiagonals(h::HubbardRealSpace{C,A}, address::A) where {C,A<:CompositeFS}
    parts = get_comp_offdiags(h, address)
    return HubbardRealSpaceOffdiagonals(address, parts, sum(length, parts))
end

Base.size(o::HubbardRealSpaceOffdiagonals) = (o.length,)

# Becomes type unstable without inline for lots of components. Recursive function is used
# because the type of the result of `o.parts[i]` can not be inferred.
@inline function Base.getindex(o::HubbardRealSpaceOffdiagonals{A}, chosen) where {A}
    return _getindex(o.parts, o.address, chosen, Val(1))
end
@inline function _getindex((p, ps...), address::A, chosen, comp::Val{I}) where {A,I}
    if chosen ≤ length(p)
        new_add, val = p[chosen]
        return BitStringAddresses.update_component(address, new_add, comp), val
    else
        chosen -= length(p)
        return _getindex(ps, address, chosen, Val(I + 1))
    end
end
