"""
    real_space_interaction(::AbstractFockAddress, u)
    real_space_interaction(::AbstractFockAddress, ::AbstractFockAddress, v)

Return the real space interaction between one or two fock states.
"""
real_space_interaction(b::BoseFS, u) = u * bose_hubbard_interaction(b) / 2
real_space_interaction(f::FermiFS, _) = 0
real_space_interaction(f::FermiFS, g::FermiFS, v) = v * count_ones(f.bs & g.bs)
real_space_interaction(f::FermiFS, b::BoseFS, v) = real_space_interaction(b, f, v)

function real_space_interaction(a::BoseFS, b::BoseFS, v)
    occ_a = occupied_orbitals(a)
    occ_b = occupied_orbitals(b)

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
function real_space_interaction(b::BoseFS, f::FermiFS, v)
    acc = 0
    for (n, i) in occupied_orbitals(b)
        acc += is_occupied(f, i) * n
    end
    return acc * v
end
function real_space_interaction(fs::CompositeFS, m)
    _interactions(fs.adds, m)
end

"""
    _interaction_col(a, bs::Tuple, us::Tuple)

Compute all interacitons in a column (below the diagonal). `a` is the address on the left-hand
side of the interaction, and `bs` and `us` are the right-hand side addresses and interactions.
"""
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...))
    return real_space_interaction(a, b, u) + _interaction_col(a, bs, us)
end

"""
    _interaction(addresses, interaction_matrix)

Compute all pairwise interactions in a tuple of `addresses`. The `interaction_matrix` sets the
intraction strengths.

The code is equivalent to the following.

```julia
acc = 0.0
for (i, a) in enumerate(addresses)
    acc += real_space_interaction(a, interaction_matrix[i, i])
    for (j, b) in enumerate(addresses[i+1:end])
        acc += real_space_interaction(a, b, interaction_matrix[i, j])
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
    self = real_space_interaction(a, u)
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, column)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, rest)
end

###
### HubbardRealSpace
###
"""
    HubbardRealSpace(address; u=ones(C, C), t=ones(C), geom=PeriodicBoundaries(M,))

General Hubbard model in real space. Supports various different kinds of particle sizes and
geometries.

```math
```

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

* `u`
* `t`: the t parameters for

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
    geom::G
end

function HubbardRealSpace(
    address;
    u=ones(num_components(address), num_components(address)),
    t=ones(num_components(address)),
    geom=PeriodicBoundaries((num_modes(address),))
)
    C = num_components(address)

    # Sanity checks
    if prod(size(geom)) ≠ num_modes(address)
        error("`geom` does not have the correct number of sites")
    elseif length(u) ≠ 1 && !issymmetric(u)
        error("`u` must be symmetric")
    elseif length(u) ≠ C * C
        error("`u` must be a $C × $C matrix")
    elseif size(t) ≠ (C,)
        error("`t` must be a vector of length $C")
    elseif address isa BoseFS2C
        error("`BoseFS2C` is not supported for this Hamiltonian, use `CompositeFS`")
    end

    u_mat = SMatrix{C,C,Float64}(u)
    t_vec = SVector{C,Float64}(t)
    return HubbardRealSpace{C,typeof(address),typeof(geom),typeof(t_vec),typeof(u_mat)}(
        address, u_mat, t_vec, geom,
    )
end

LOStructure(::Type{<:HubbardRealSpace}) = Hermitian()

function Base.show(io::IO, h::HubbardRealSpace)
    println(io, "HubbardRealSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  u = ", Float64.(h.u), ",")
    println(io, "  t = ", Float64.(h.t), ",")
    println(io, "  geom = ", h.geom, ",")
    println(io, ")")
end

starting_address(h::HubbardRealSpace) = h.address
diagonal_element(h::HubbardRealSpace, address) = real_space_interaction(address, h.u)
diagonal_element(h::HubbardRealSpace{1}, address) = real_space_interaction(address, h.u[1])

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

Offdiagonals for the bosonic part of a [`HubbardRealSpace`](@ref) model.

Used when the model's address is a [`BoseFS`](@ref), or a [`CompositeFS`](@ref) with a
[`BoseFS`](@ref) component.
"""
struct HubbardRealSpaceBoseOffdiagonals{G,A<:BoseFS} <: AbstractOffdiagonals{A,Float64}
    geom::G
    address::A
    t::Float64
    length::Int
end

function offdiagonals(h::HubbardRealSpace, comp, add::BoseFS)
    neighbours = num_neighbours(h.geom)
    return HubbardRealSpaceBoseOffdiagonals(
        h.geom, add, h.t[comp], numberoccupiedsites(add) * neighbours,
    )
end

Base.size(o::HubbardRealSpaceBoseOffdiagonals) = (o.length,)

function Base.getindex(o::HubbardRealSpaceBoseOffdiagonals, chosen)
    neighbours = num_neighbours(o.geom)
    particle, neigh = fldmod1(chosen, neighbours)
    i = find_particle(o.address, particle)
    target_site = neighbour_site(o.geom, i.site, neigh)
    if iszero(target_site)
        # Move is illegal in specified geometry.
        return o.address, 0.0
    else
        j = find_site(o.address, target_site)
        new_address, onproduct = move_particle(o.address, i, j)
        return new_address, -o.t * √onproduct
    end
end

# Fermi part
struct HubbardRealSpaceFermiOffdiagonals{G,A<:FermiFS} <: AbstractOffdiagonals{A,Float64}
    geom::G
    address::A
    t::Float64
    length::Int
end

function offdiagonals(h::HubbardRealSpace, comp, add::FermiFS{N}) where {N}
    neighbours = num_neighbours(h.geom)
    return HubbardRealSpaceFermiOffdiagonals(
        h.geom, add, h.t[comp], N * neighbours,
    )
end

Base.size(o::HubbardRealSpaceFermiOffdiagonals) = (o.length,)

function Base.getindex(o::HubbardRealSpaceFermiOffdiagonals, chosen)
    @boundscheck 1 ≤ chosen ≤ length(o) || throw(BoundsError(o, chosen))
    neighbours = num_neighbours(o.geom)
    particle, neigh = fldmod1(chosen, neighbours)
    source_site = find_particle(o.address, particle)
    target_site = neighbour_site(o.geom, source_site, neigh)
    if iszero(target_site)
        return o.address, 0.0
    else
        new_address, sign = move_particle(o.address, source_site, target_site)
        return new_address, -o.t * sign
    end
end

# TODO: do this without intermediate vector
# TODO: make this work for multi-component models
function Base.iterate(o::HubbardRealSpaceFermiOffdiagonals)
    add = o.address
    neighbours = num_neighbours(o.geom)
    # While we do not know the exact number of offdiagonals, we know an upper bound.
    # We only fill the result up to the point we need it, and return a view into it.
    result = MVector{num_particles(add) * neighbours,Tuple{typeof(add),Float64}}(undef)
    idx = 0
    for site in occupied_orbitals(add)
        for i in 1:neighbours
            idx += 1
            neigh = neighbour_site(o.geom, site, i)
            if iszero(neigh)
                @inbounds result[idx] = (add, 0.0)
            else
                new_add, sign = move_particle(add, site, neigh)
                @inbounds result[idx] = (new_add, -o.t * sign)
            end
        end
    end
    return iterate(o, (SVector(result), 1))
end
function Base.iterate(o::HubbardRealSpaceFermiOffdiagonals, (vals, i))
    if i > length(vals)
        return nothing
    else
        return vals[i], (vals, i + 1)
    end
end

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
get_comp_offdiags(h::HubbardRealSpace, address) = _get_comp_offdiags(address.adds, h, Val(1))

@inline function _get_comp_offdiags((a,as...), h, ::Val{I}) where {I}
    return (offdiagonals(h, I, a), _get_comp_offdiags(as, h, Val(I+1))...)
end
@inline _get_comp_offdiags(::Tuple{}, h, ::Val) = ()

function offdiagonals(h::HubbardRealSpace{C,A}, address::A) where {C,A<:CompositeFS}
    parts = get_comp_offdiags(h, address)
    return HubbardRealSpaceOffdiagonals(address, parts, sum(length, parts))
end

Base.size(o::HubbardRealSpaceOffdiagonals) = (o.length,)

function Base.getindex(o::HubbardRealSpaceOffdiagonals{A}, chosen) where {A}
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

###
### Single-component models
###
offdiagonals(h::HubbardRealSpace{1,A}, add::A) where {A} = offdiagonals(h, 1, add)
