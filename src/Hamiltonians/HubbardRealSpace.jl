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
local_interaction(b::BoseFS, u) = u * bose_hubbard_interaction(b) / 2
local_interaction(f::FermiFS, _) = 0
local_interaction(f::FermiFS, g::FermiFS, v) = v * count_ones(f.bs & g.bs)
local_interaction(f::FermiFS, b::BoseFS, v) = local_interaction(b, f, v)

function local_interaction(a::BoseFS, b::BoseFS, v)
    occ_a = occupied_modes(a)
    occ_b = occupied_modes(b)

    (n_a, i_a, _), st_a = iterate(occ_a)
    (n_b, i_b, _), st_b = iterate(occ_b)

    acc = 0
    while true
        if i_a > i_b
            # b is behind and needs to do a step
            iter = iterate(occ_b, st_b)
            isnothing(iter) && return acc * v
            (n_b, i_b, _), st_b = iter
        elseif i_a < i_b
            # a is behind and needs to do a step
            iter = iterate(occ_a, st_a)
            isnothing(iter) && return acc * v
            (n_a, i_a, _), st_a = iter
        else
            # a and b are at the same position
            acc += n_a * n_b
            # now both need to do a step
            iter = iterate(occ_a, st_a)
            isnothing(iter) && return acc * v
            (n_a, i_a, _), st_a = iter
            iter = iterate(occ_b, st_b)
            isnothing(iter) && return acc * v
            (n_b, i_b, _), st_b = iter
        end
    end
end
function local_interaction(b::BoseFS, f::FermiFS, v)
    acc = 0
    for (n, i) in occupied_modes(b)
        acc += is_occupied(f, i) * n
    end
    return acc * v
end
function local_interaction(fs::CompositeFS, m)
    _interactions(fs.components, m)
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
    _interaction(addresses, interaction_matrix)

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

###
### HubbardRealSpace
###
"""
    HubbardRealSpace(address; u=ones(C, C), t=ones(C), geometry=PeriodicBoundaries(M,))

Hubbard model in real space. Supports single or multi-component Fock state
addresses (with `C` components) and various (rectangular) lattice geometries.

## Address types

* [`BoseFS`](@ref): Single-component Bose-Hubbard model.
* [`FermiFS`](@ref): Single-component Fermi-Hubbard model. This address only provides a
  single species of Fermions. You probably want to use [`CompositeFS`](@ref).
* [`CompositeFS`](@ref): For multi-component models.

## Geometries

* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)

## Other parameters

* `u`: the interaction parameters. Must be a symmetric matrix. `u[i, j]` corresponds to the
  interaction between the `i`-th and `j`-th component. `u[i, i]` corresponds to the
  interaction of a component with itself. Note that `u[i,i]` must be zero for fermionic
  components.
* `t`: the hopping strengths. Must be a vector of length `C`. The `i`-th element of the
  vector corresponds to the hopping strength of the `i`-th component.

"""
struct HubbardRealSpace{
    C,A,G,
    # The following need to be type params.
    T<:SVector{C,Float64},
    U<:SMatrix{C,C,Float64},
} <: AbstractHamiltonian{Float64}
    address::A
    u::U
    t::T
    geometry::G
end

function HubbardRealSpace(
    address;
    u=ones(num_components(address), num_components(address)),
    t=ones(num_components(address)),
    geometry=PeriodicBoundaries((num_modes(address),))
)
    C = num_components(address)

    # Sanity checks
    if prod(size(geometry)) ≠ num_modes(address)
        error("`geometry` does not have the correct number of sites")
    elseif length(u) ≠ 1 && !issymmetric(u)
        error("`u` must be symmetric")
    elseif length(u) ≠ C * C
        error("`u` must be a $C × $C matrix")
    elseif size(t) ≠ (C,)
        error("`t` must be a vector of length $C")
    elseif address isa BoseFS2C
        error("`BoseFS2C` is not supported for this Hamiltonian, use `CompositeFS`")
    end
    warn_fermi_interaction(address, u)

    u_mat = SMatrix{C,C,Float64}(u)
    t_vec = SVector{C,Float64}(t)
    return HubbardRealSpace{C,typeof(address),typeof(geometry),typeof(t_vec),typeof(u_mat)}(
        address, u_mat, t_vec, geometry,
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
            @warn "component $(c) is Fermionic, but was given a self-interaction " *
                "strength of $(u[c,c])" maxlog=1
        end
    end
end
function warn_fermi_interaction(address::FermiFS, u)
    if u ≠ ones(1, 1) && u[1, 1] ≠ 0
        @warn "address is Fermionic, but was given a self-interaction " *
            "strength of $(u[1,1])" maxlog=1
    end
end
warn_fermi_interaction(_, _) = nothing

LOStructure(::Type{<:HubbardRealSpace}) = Hermitian()

function Base.show(io::IO, h::HubbardRealSpace)
    println(io, "HubbardRealSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  u = ", Float64.(h.u), ",")
    println(io, "  t = ", Float64.(h.t), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, ")")
end

starting_address(h::HubbardRealSpace) = h.address
diagonal_element(h::HubbardRealSpace, address) = local_interaction(address, h.u)
diagonal_element(h::HubbardRealSpace{1}, address) = local_interaction(address, h.u[1])

###
### Offdiagonals
###
# This may be an inefficient implementation, but it is not actually used anywhere in the
# main algorithm.
get_offdiagonal(h::HubbardRealSpace, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::HubbardRealSpace, add) = length(offdiagonals(h, add))

# Bosonic part
"""
    HubbardRealSpaceBoseOffdiagonals{G,A} <: AbstractOffdiagonals{A,Float64}

Offdiagonals for a Bosonic part of a [`HubbardRealSpace`](@ref) model.

Used when the model's address is a [`BoseFS`](@ref), or a [`CompositeFS`](@ref) with a
[`BoseFS`](@ref) component.
"""
struct HubbardRealSpaceBoseOffdiagonals{G,A<:BoseFS} <: AbstractOffdiagonals{A,Float64}
    geometry::G
    address::A
    t::Float64
    length::Int
end

function offdiagonals(h::HubbardRealSpace, comp, add::BoseFS)
    neighbours = num_neighbours(h.geometry)
    return HubbardRealSpaceBoseOffdiagonals(
        h.geometry, add, h.t[comp], num_occupied_modes(add) * neighbours,
    )
end

Base.size(o::HubbardRealSpaceBoseOffdiagonals) = (o.length,)

function Base.getindex(o::HubbardRealSpaceBoseOffdiagonals, chosen)
    neighbours = num_neighbours(o.geometry)
    particle, neigh = fldmod1(chosen, neighbours)
    i = find_occupied_mode(o.address, particle)
    target_site = neighbour_site(o.geometry, i.mode, neigh)
    if iszero(target_site)
        # Move is illegal in specified geometry.
        return o.address, 0.0
    else
        j = find_mode(o.address, target_site)
        new_address, onproduct = move_particle(o.address, i, j)
        return new_address, -o.t * √onproduct
    end
end

# Fermi part
"""
    HubbardRealSpaceFermiOffdiagonals <: AbstractOffdiagonals

Offdiagonals for a Fermionic part of a [`HubbardRealSpace`](@ref) model.

Used when the model's address is a [`FermiFS`](@ref), or a [`CompositeFS`](@ref) with a
[`FermiFS`](@ref) component.
"""
struct HubbardRealSpaceFermiOffdiagonals{G,A<:FermiFS} <: AbstractOffdiagonals{A,Float64}
    geometry::G
    address::A
    t::Float64
    length::Int
end

function offdiagonals(h::HubbardRealSpace, comp, add::FermiFS{N}) where {N}
    neighbours = num_neighbours(h.geometry)
    return HubbardRealSpaceFermiOffdiagonals(
        h.geometry, add, h.t[comp], N * neighbours,
    )
end

Base.size(o::HubbardRealSpaceFermiOffdiagonals) = (o.length,)

function Base.getindex(o::HubbardRealSpaceFermiOffdiagonals, chosen)
    @boundscheck 1 ≤ chosen ≤ length(o) || throw(BoundsError(o, chosen))
    neighbours = num_neighbours(o.geometry)
    particle, neigh = fldmod1(chosen, neighbours)
    source_site = find_occupied_mode(o.address, particle)
    target_site = neighbour_site(o.geometry, source_site, neigh)
    if iszero(target_site)
        return o.address, 0.0
    else
        new_address, sign = move_particle(o.address, source_site, target_site)
        return new_address, -o.t * sign
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
