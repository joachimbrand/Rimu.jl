"""
    _mom_space_energies_and_ks(ks_vec_of_vecs, geometry, t, dispersion::Function = hubbard_dispersion)

Return a tuple `(kes_mat, ks_mat)` with the kinetic energies and momentum vectors for each mode given
allowed momenta `ks_vec_of_vecs`, lattice `geometry`, hopping strengths `t`, and a single-particle
`dispersion` (default `hubbard_dispersion`).
"""
function _mom_space_energies_and_ks(ks_vec_of_vecs::Vector, geometry::CubicGrid{D, S}, t::SMatrix, 
        dispersion::Function) where {D, S}
    # Calculate the dispersion relation for a given set of k values and hopping strength t.
    C = size(t, 1)
    M = prod(S)
    kes_mat = zeros(Float64,C,M)
    ks_mat = zeros(Float64,D,M)
    for i in 1:M
        mom_val = _grid_column_of_momenta(M-i+1, ks_vec_of_vecs, geometry)
        ks_mat[:,i] = mom_val
        for j in 1:C
            kes_mat[j,i] = convert(Float64,sum(dispersion.(t[j,:], mom_val)))
        end
    end
    return SMatrix{C,M,Float64}(kes_mat), SMatrix{D,M,Float64}(ks_mat)
end
function _grid_column_of_momenta(linear_mode_index::Int, ks::Vector, geometry::CubicGrid)
    mode_coordinates = geometry[linear_mode_index]
    return [ks[dim][index_per_dim] for (dim, index_per_dim) in enumerate(mode_coordinates)]
end

"""
    _mom_hopping(kes, address)

    Calculate the hopping term for a single-component or multi-component Fock state address in momentum space.
"""
@inline function _mom_hopping(kes::SMatrix{C, <:Any, T}, address::CompositeFS) where {C, T}
    # Direct the calculation to a type-stable unrolled accumulator
    return _mom_hopping_unrolled(kes, address.components, Val(C))
end

# Base case: When the static counter reaches 0, stop and return the accumulated product
@inline _mom_hopping_unrolled(::SMatrix, ::Tuple, ::Val{0}) = 0.0

# Recursive case: Processes component 'I' at compile time, then moves to 'I-1'
@inline function _mom_hopping_unrolled(
    kes::SMatrix{<:Any, <:Any, T}, comps::Tuple, ::Val{I}
) where {T, I}
    occ = occupied_mode_map(comps[I]) # This is a ModeMap for the I-th component
    onproduct = zero(T)
    for x in occ
        onproduct += kes[I, x.mode] * x.occnum
    end
    
    return onproduct + _mom_hopping_unrolled(kes, comps, Val(I - 1))
end
@inline function _mom_hopping(kes::SMatrix{1}, address::SingleComponentFockAddress) 
    occ = occupied_mode_map(address)
    onproduct = zero(eltype(kes))
    @inbounds for x in occ
        onproduct += kes[1, x.mode] * x.occnum
    end
    return onproduct
end

"""
    mom_transfer_offdiagonal(add, chosen, map, g; fold=true)
    mom_transfer_offdiagonal(add1, add2, chosen, map1, map2, g; fold=true)

This function does the excitation operation on the given `add` or between `add1` and `add2` 
in momentum space for the same or two different components of a multi-component Fock state 
address, respectively, which contributes to the off-diagonal part of the Hamiltonian. 
The excitation is carried out to get a response similar to the nearest neighbour 
interaction and the on-site interaction operation in real space. The excitation is 
determined by the integer `chosen`. `map`, `map1`, and `map2` are the occupied 
mode maps for the relevant components of the multi-component Fock state.`g` is 
the geometry of the lattice. If `fold` is true, momentum transfer that goes
outside the first Brillouin zone is folded back into it.

See also [`mom_transfer_diagonal`](@ref). 
"""
@inline function mom_transfer_offdiagonal(
    add::SingleComponentFockAddress{<:Any, M}, chosen::Int, map::ModeMap,
    g::CubicGrid{D,S}) where {M, D, S}
    # Get the momentum transfer for a given excitation.
    singlies = length(map) # number of at least singly occupied modes
    double = chosen - singlies * (singlies - 1) * (M - 2)
    if double > 0
        # Both moves from the same mode.
        double, mom_change = fldmod1(double, M - 1)
        idx = first(map) # placeholder
        for i in map
            double -= i.occnum ≥ 2
            if double == 0
                idx = i
                break
            end
        end
        src_indices = (idx, idx)
    else
        # Moves from different modes.
        pair, mom_change = fldmod1(chosen, M - 2)
        fst, snd = fldmod1(pair, singlies - 1) # where the holes are to be made
        if snd < fst # put them in ascending order
            f_hole = snd
            s_hole = fst
        else
            f_hole = fst
            s_hole = snd + 1 # as we are counting through all singlies
        end
        src_indices = (map[f_hole], map[s_hole])
    end
    src_modes = (src_indices[1].mode, src_indices[2].mode)
    src_loc = (g[src_modes[1]], g[src_modes[2]])
    Q = g[mom_change+1] - g[1]
    dst_loc = (src_loc[2]-Q, src_loc[1]+Q)
    dst_loc = (mod1.(dst_loc[1], S) , mod1.(dst_loc[2], S))
    if dst_loc == src_loc || reverse(dst_loc) == src_loc
        # If the momentum transfer is out of bounds, we return the original address.
        Q = g[M] - g[1]
        dst_loc = (src_loc[2]-Q, src_loc[1]+Q)
        dst_loc = (mod1.(dst_loc[1], S) , mod1.(dst_loc[2], S))
    end
    dst_indices = find_mode(add, (g[dst_loc[1]], g[dst_loc[2]]))
    return excitation(add, dst_indices, src_indices)..., src_modes..., -Q
end

@inline function mom_transfer_offdiagonal(
    add1::SingleComponentFockAddress{<:Any, M}, add2::SingleComponentFockAddress{<:Any, M}, 
    chosen::Int, map1::ModeMap, map2::ModeMap, g::CubicGrid{D,S}) where {M, D, S}
    # Get the momentum transfer for a given excitation.
    singlies = length(map2)
    pair, mom_change = fldmod1(chosen, M - 1)
    f_hole, s_hole = fldmod1(pair, singlies) # where the holes are to be made
    src_indices = (map1[f_hole], map2[s_hole])
    src_modes = (src_indices[1].mode, src_indices[2].mode)
    src_loc = (g[src_modes[1]], g[src_modes[2]])
    Q = g[mom_change+1] - g[1]
    dst_loc = (src_loc[1]+Q, src_loc[2]-Q)
    dst_loc = (mod1.(dst_loc[1], S) , mod1.(dst_loc[2], S))
    return excitation(add1, find_mode(add1, (g[dst_loc[1]],)), (src_indices[1],))..., 
        excitation(add2, find_mode(add2, (g[dst_loc[2]],)), (src_indices[2],))..., src_modes..., -Q
end

"""
    _mom_transfer_diagonal(map, g, u, w)
    _mom_transfer_diagonal(map1, map2, g, u, w)

This function does the excitation operation on the given `map` or between `map1` and `map2` 
which are the occupied mode maps for the relevant components  of the multi-component
Fock state in momentum space. The operation is carried out for the same or two different 
components of a multi-component Fock state address, respectively, which contributes to 
the diagonal part of the Hamiltonian. The excitation is carried out to get the 
response similar to the nearest neighbour interaction and on-site interaction 
operation in real space. `g` is the geometry of the lattice. `u` and `w` are 
the on-site and nearest neighbour interaction strengths, respectively. If 
either `u` or `w` is `nothing`, the corresponding interaction term is ignored. 

"""
@inline function _mom_transfer_diagonal(map::BoseOccupiedModeMap, g::CubicGrid{D,S}, u, w) where {D, S}
    onproduct = 0.0
    u_scaled = u / 2.0 + w * D
    for i in 1:length(map)
        occ_i = Float64(map[i].occnum)
        onproduct += occ_i * (occ_i - 1.0) * u_scaled
        g_i = g[map[i].mode]
        
        for j in 1:i-1
            occ_j = Float64(map[j].occnum)
            q = g_i - g[map[j].mode]
            onproduct += 2.0 * occ_i * occ_j * (u + w * (D + _cosin_sum(q, S)))
        end
    end
    return onproduct
end

@inline function _mom_transfer_diagonal(map::BoseOccupiedModeMap, g::CubicGrid{D,S}, ::Nothing, w) where {D, S}
    onproduct = 0.0
    for i in 1:length(map)
        occ_i = Float64(map[i].occnum)
        onproduct += occ_i * (occ_i - 1.0) * D
        g_i = g[map[i].mode]
        
        for j in 1:i-1
            occ_j = Float64(map[j].occnum)
            q = g_i - g[map[j].mode]
            onproduct += 2.0 * occ_i * occ_j * (D + _cosin_sum(q, S))
        end
    end
    return onproduct * w
end

@inline function _mom_transfer_diagonal(map::BoseOccupiedModeMap, ::CubicGrid, u, ::Nothing)
    onproduct = 0.0
    for i in 1:length(map)
        occ_i = Float64(map[i].occnum)
        onproduct += occ_i * (occ_i - 1.0) / 2.0
        for j in 1:i-1
            occ_j = Float64(map[j].occnum)
            onproduct += 2.0 * occ_i * occ_j
        end
    end
    return onproduct * u
end

@inline function _mom_transfer_diagonal(map::FermiOccupiedModeMap, g::CubicGrid{D,S}, _, w) where {D, S}
    onproduct = 0.0
    for i in 1:length(map)
        mode_i = map[i].mode
        occ_i  = Float64(map[i].occnum)
        onproduct += occ_i * (occ_i - 1.0)
        g_i = g[mode_i] 
        for j in 1:i-1
            occ_j = Float64(map[j].occnum)
            q = g_i - g[map[j].mode] 
            onproduct += 2.0 * occ_i * occ_j * (D - _cosin_sum(q, S))
        end
    end
    return onproduct * w
end

@inline _mom_transfer_diagonal(::FermiOccupiedModeMap, ::CubicGrid, _, ::Nothing) = 0

@inline function _mom_transfer_diagonal(map1::ModeMap, map2::ModeMap, ::CubicGrid{D}, u, w) where D
    onproduct = 0.0
    for i in map1
        occ_i = Float64(i.occnum)
        for j in map2
            occ_j = Float64(j.occnum)
            onproduct += occ_i * occ_j
        end
    end
    return onproduct * _interaction_parameter_diag(u, w, D)
end

@inline function _mom_transfer_diagonal(map1::FermiOccupiedModeMap, map2::FermiOccupiedModeMap, 
    ::CubicGrid{D}, u, w) where D
    return length(map1) * length(map2) * _interaction_parameter_diag(u, w, D)
end

@inline function _cosin_sum(q::SVector{D}, S::NTuple{D}) where {D}
    onproduct = 0.0
    for i in 1:D
        onproduct += cos(2π*q[i] / S[i])
    end
    return onproduct
end

"""
    mom_transfer_diagonal(components, g)
    
This function returns a diagonal element of the Hamiltonian coresponding to the given address 
stored in `components`. Here, `components` is a tuple of [`HubbardMomSpaceComponentData`](@ref) 
for each pair combination of component of the multi-component Fock state address.

'''math
    Ĥ_\\text{int} = ½\\sum_{p,q,σ,σ'} V_{σσ'} b̂^†_{pσ} b̂^†_{qσ'} b̂_{qσ'} b̂_{pσ}
'''

where `V_{σσ}' is the interaction coefficient that depends on interaction parameters that are 
stored in `components` and `g` is the geometry of the lattice.

"""
@inline _mom_transfer_diagonal(components::Tuple{}, g::CubicGrid) = 0.0

@inline function _mom_transfer_diagonal((data, rest...)::Tuple, g::CubicGrid)
    if isnothing(data.u) && isnothing(data.w)
        current_product = 0.0
    else
        idx1, idx2 = component_index(data)
        
        current_product = if idx1 == idx2
            _mom_transfer_diagonal(data.occmap1, g, data.u, data.w)
        else
            _mom_transfer_diagonal(data.occmap1, data.occmap2, g, data.u, data.w)
        end
    end

    # Sum up the current step with the rest of the unrolled tuple
    return current_product + _mom_transfer_diagonal(rest, g)
end

@inline _interaction_parameter_diag(u::Float64, w::Float64, D::Int) = u + 2 * w * D
@inline _interaction_parameter_diag(::Nothing, w::Float64, D::Int) = 2 * w * D
@inline _interaction_parameter_diag(u::Float64, ::Nothing, _) = u

"""
    HubbardMomSpace(address; geometry=PeriodicBoundaries(M,), t=ones(C, D), u=ones(C, C), 
        w=zeros(C, C), dispersion=hubbard_dispersion) <: AbstractHamiltonian{Float64}

Hubbard model in momentum space. Supports single or multi-component Fock state
addresses (with `C` components) and various (rectangular) lattice geometries
in `D` dimensions and with a total of `M` momentum modes.

```math
  \\hat{H} = -\\sum_{k,σ} ϵ_{kσ} n_{kσ} +
  \\sum_{p,q,k,σ,σ'} V_{σσ'} a^†_{p+k,σ} a^†_{q-k,σ'} a_{q,σ'} a_{p,σ}
```
where ``ϵ_{kσ} = Σ_{d=1}^{D} ε(k_d)`` and ``ε(k)`` is a one-dimensional single particle `dispersion` 
(with default [`hubbard_dispersion`](@ref))  and
``V_{σσ'} = (u_{σσ'}(1- \\frac{δ_{σσ'}}{2}) + w_{σσ'} \\sum_{d=1}^{D} \\cos(q_d))/M``
the coefficients of a two-body interaction with onsite (``u_{σσ'}``) and nearest-neighbour 
(``w_{σσ'}``) interaction terms.

## Address types

* [`BoseFS`](@ref): Single-component Bose-Hubbard model.
* [`FermiFS`](@ref): Single-component Fermi-Hubbard model.
* [`CompositeFS`](@ref): For multi-component models (must have equal number of modes).

Note that a single component of fermions cannot interact with itself. A warning
is produced if `address`is incompatible with the interaction parameters `u`.

## Geometries

Implemented [`CubicGrid`](@ref)s for keyword `geometry`

* [`PeriodicBoundaries`](@ref)

Default is `geometry=PeriodicBoundaries(M,)`, i.e., a one-dimensional lattice with the
number of sites `M` inferred from the number of modes in `address`.

## Other parameters
* `t`: the hopping strengths. Must be a matrix of size `C × D`. The `i`-th and `j`-th element of the
  matrix corresponds to the hopping strength of the `i`-th component and `j`-th direction.
* `u`: the on-site interaction parameters. Must be a symmetric matrix of size `C × C`. `u[i, j].`
  corresponds to the interaction between the `i`-th and `j`-th component. `u[i, i]`
  corresponds to the interaction of a component with itself.
* `w`: the nearest neighbour interaction parameters. Must be a symmetric matrix of size `C × C`.
  `w[i, j]` corresponds to the interaction between the `i`-th and `j`-th component.
  
* `dispersion`: the function used to calculate the dispersion relation. Default is 
    [`hubbard_dispersion`](@ref) which corresponds to the standard tight binding model.  

See also [`HubbardRealSpace`](@ref), [`HubbardMom1D`](@ref), [`ExtendedHubbardReal1D`](@ref).
"""
struct HubbardMomSpace{
    TT,
    C, # components    
    D, # dimension
    A<:AbstractFockAddress,
    G<:CubicGrid,
    # The following need to be type params.
    KS<:SMatrix{D,<:Any, Float64}, # k values
    KES<:SMatrix{C,<:Any,Float64},
    T<:SMatrix{C,D,<:Any}, # hopping strengths
    U<:Union{SMatrix{C,C,Float64},Nothing},
    W<:Union{SMatrix{C,C,Float64},Nothing},
} <: AbstractHamiltonian{TT}
    address::A
    ks_mat::KS # k values
    kes_mat::KES # kinetic energy values
    t::T
    u::U # interactions
    w::W # nearest neighbour interactions
    geometry::G
end

function HubbardMomSpace(
    address::AbstractFockAddress;
    geometry::CubicGrid=PeriodicBoundaries((num_modes_check_equal(address),)),
    t=ones(num_components(address), num_dimensions(geometry)),
    u=ones(num_components(address), num_components(address)),
    w=zeros(num_components(address), num_components(address)),
    dispersion::Function = hubbard_dispersion,
)
    C = num_components(address)
    D = num_dimensions(geometry)
    S = size(geometry)

    # Sanity checks
    if prod(size(geometry)) ≠ num_modes_check_equal(address)
        throw(ArgumentError("`geometry` does not have the correct number of sites"))
    elseif !(address isa SingleComponentFockAddress || address isa CompositeFS)
        throw(ArgumentError(
            "unsupported address type; use `CompositeFS` or a subtype of `SingleComponentFockAddress`"
        ))
    end

    t_mat = _t_or_v_to_matrix(:t, t, C, D; zero_is_nothing=false)
    u_mat = _u_or_w_to_matrix(:u, u, C)
    w_mat = _u_or_w_to_matrix(:w, w, C)

    warn_fermi_interaction(address, u_mat)
    ks_vec_of_vecs = Array{Float64}[]
    for i in eachindex(S)
        step = 2π/S[i]
        if isodd(S[i])
            start = -π*(1+1/S[i]) + step
        else
            start = -π + step
        end
        kr = range(start; step = step, length = S[i])
        if isodd(S[i])
            push!(ks_vec_of_vecs,[j for j in kr])
        else
            push!(ks_vec_of_vecs,reverse([j for j in kr]))
        end
    end
    kes_mat, ks_mat = _mom_space_energies_and_ks(ks_vec_of_vecs, geometry, t_mat, dispersion)

    return HubbardMomSpace{eltype(kes_mat),C,D,typeof(address),typeof(geometry),typeof(ks_mat),typeof(kes_mat),
    typeof(t_mat),typeof(u_mat),typeof(w_mat)}(
        address, ks_mat, kes_mat, t_mat, u_mat, w_mat, geometry,
    )
end

LOStructure(::Type{<:HubbardMomSpace}) = IsHermitian()

function Base.show(io::IO, h::HubbardMomSpace{<:Any,C}) where {C}
    io = IOContext(io, :compact => true)
    println(io, "HubbardMomSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  t = ", (h.t), ",")
    if isnothing(h.u)
        println(io, "  u = ", zeros(C,C), ",")
    else
        println(io, "  u = ", Float64.(h.u), ",")
    end
    if isnothing(h.w)
        println(io, "  w = ", zeros(C,C), ",")
    else
        println(io, "  w = ", Float64.(h.w), ",")
    end
    print(io, ")")
end

# Overload equality due to stored potential energy arrays.
Base.:(==)(H::HubbardMomSpace, G::HubbardMomSpace) = 
    all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(h::HubbardMomSpace) = h.address

dimension(::HubbardMomSpace, address) = number_conserving_dimension(address)

# offdiagonals =========================================================================== #
"""
    HubbardMomSpaceComponentData(
        geometry, parent::A, address1, address2, u, w
    ) <: AbstractVector{Pair{A,TT}}

This holds the off-diagonals for a single- and multi-component two-body on-site and 
nearest-neighbour interaction terms. It is structured where the index of this vector
determines `p`, `q`, `σ`, `σ'` and `k` in the two particle operator given by 
```math a^†_{p+k,σ} a^†_{q-k,σ'} a_{q,σ'} a_{p,σ}```. The operator is operated on a 
`parent` which is a single-component (where `address1`=`address2`=`parents`) or 
multi-component Fock address where `address1` and `address2` are the single-component 
Fock addresses representing  `σ` and `σ'` component respectively. The operation 
creates a new address and the corresponding coefficient which is stored as a pair. 
`u` and `w` represents the interaction coefficient coresponding to on-site and 
nearest-neighbour interactions, respectively, and are used to calculate the 
coefficient of the respective new address.
"""
struct HubbardMomSpaceComponentData{
    TT,C,I1,I2,D,G,A,A1,A2,U<:Union{Float64,Nothing},W<:Union{Float64,Nothing},O1,O2
} <: AbstractVector{Pair{A,TT}}
    geometry::G
    parent_address::A
    address1::A1
    address2::A2
    u::U # interaction strength
    w::W # nearest neighbour interaction strength
    occmap1::O1
    occmap2::O2
    function HubbardMomSpaceComponentData{TT,C,I1,I2,D}(
        geometry::G,
        parent::A,
        address1::A1,
        address2::A2,
        u::U,
        w::W,
        occmap1::O1=occupied_mode_map(address1),
        occmap2::O2=occupied_mode_map(address2),
    ) where {TT,C,I1,I2,D,G,A,A1,A2,U,W,O1,O2}
        return new{TT,C,I1,I2,D,G,A,A1,A2,U,W,O1,O2}(
            geometry, parent, address1, address2, u, w, occmap1, occmap2
        )
    end
end


function Base.size(data::HubbardMomSpaceComponentData{<:Any,<:Any,I,I}) where {I}
    if isnothing(data.u) && isnothing(data.w)
        return (0,)
    else
        M = num_modes_check_equal(data.address1)
        s1, d1 = num_singly_doubly_occupied_sites(data.address1)
        return  (s1 * (s1 - 1) * (M - 2) + d1 * (M - 1),)
    end
end

function Base.size(data::HubbardMomSpaceComponentData)
    if isnothing(data.u) && isnothing(data.w)
        return (0,)
    else
        M = num_modes_check_equal(data.address1)
        s1 = length(data.occmap1)
        s2 = length(data.occmap2)
        return (s1 * s2 * (M - 1),)
    end
end

component_index(::HubbardMomSpaceComponentData{<:Any,<:Any,I1,I2}) where {I1,I2} = (I1, I2)

function Base.getindex(data::HubbardMomSpaceComponentData{TT,C,I,I,D}, chosen::Int) where {TT,C,I,D}
    geometry = data.geometry
    S = size(geometry)
    M = prod(S)
    map1 = data.occmap1
    new_add, onproduct,_,_,q = mom_transfer_offdiagonal(data.address1, chosen, map1, geometry)
    if data.parent_address isa CompositeFS
        new_parent = BitStringAddresses.update_component(
            data.parent_address, new_add, Val(I)
        )
    else
        new_parent = new_add
    end
    return new_parent => convert(TT, _interaction_parameter(data.u, data.w, q, S) * onproduct / M)
end

function Base.getindex(data::HubbardMomSpaceComponentData{TT,C,I1,I2,D}, chosen::Int) where {TT,C,I1,I2,D}
    geometry = data.geometry
    S = size(geometry)
    M = prod(S)
    new_add1, onproduct1,new_add2, onproduct2,_,_,q = 
        mom_transfer_offdiagonal(data.address1,data.address2,chosen,data.occmap1,data.occmap2,data.geometry)
    new_parent = BitStringAddresses.update_component(
        data.parent_address, new_add1, Val(I1)
    )
    new_parent = BitStringAddresses.update_component(
        new_parent, new_add2, Val(I2)
    )
    return new_parent => convert(TT, 2 * _interaction_parameter(data.u, data.w, q, S) * onproduct1 * onproduct2 / M)
end

@inline _interaction_parameter(u::Float64, w::Float64, q::SVector, S::NTuple) = u/2 + w * _cosin_sum(q, S)
@inline _interaction_parameter(::Nothing, w::Float64, q::SVector, S::NTuple) = w * _cosin_sum(q, S)
@inline _interaction_parameter(u::Float64, ::Nothing, ::SVector, ::NTuple) = u/2

# column ================================================================================= #
struct HubbardMomSpaceColumn{TT,H,G,A,C<:Tuple} <: AbstractOperatorColumn{A,TT,H}
    hamiltonian::H
    geometry::G
    address::A
    components::C
    num_offdiagonals::Int
end

parent_operator(column::HubbardMomSpaceColumn) = column.hamiltonian
starting_address(column::HubbardMomSpaceColumn) = column.address

function diagonal_element(col::HubbardMomSpaceColumn{TT}) where {TT}
    ke = _mom_hopping(col.hamiltonian.kes_mat, col.address)
    diag = _mom_transfer_diagonal(col.components, col.geometry)/num_modes_check_equal(col.address)
    return convert(TT, ke + diag)
end

function operator_column(h::HubbardMomSpace{TT,<:Any,<:Any,A,G}, address) where {TT,A,G}
    components = _column_components(h, address)
    return HubbardMomSpaceColumn{TT,typeof(h),G,A,typeof(components)}(
        h, h.geometry, address, components, sum(length, components)
    )
end

# Collect HubbardMomSpaceComponentData for each component of the address.
@inline function _column_components(h::HubbardMomSpace{TT,1,D}, 
        address::SingleComponentFockAddress) where {TT,D}
    u = isnothing(h.u) ? nothing : h.u[1]
    w = isnothing(h.w) ? nothing : h.w[1]
    return (
        HubbardMomSpaceComponentData{TT,1,1,1,D}(
            h.geometry, address, address, address, u, w
        ),
    )
end

@inline function _column_components(h::HubbardMomSpace{TT,1,D}, address::FermiFS) where {TT,D}
    w = isnothing(h.w) ? nothing : h.w[1, 1]
    return (
        HubbardMomSpaceComponentData{TT,1,1,1,D}(
            h.geometry, address, address, address, nothing, w
        ),
    )
end

@inline function _column_components(h::HubbardMomSpace{TT,<:Any,D}, address::CompositeFS) where {TT,D}
    return _column_components(h, address, address.components, h.u, h.w, Val(1))
end

@inline function _column_components(::HubbardMomSpace{TT,<:Any,D}, _, ::Tuple{}, ::Union{SMatrix{0,0},Nothing}, 
    ::Union{SMatrix{0,0},Nothing}, ::Val) where {TT,D}
    return ()
end
@inline function _column_components(
    h::HubbardMomSpace{TT,C,D}, address, (a, as...),
    m::Union{SMatrix{N,N},Nothing}, σ::Union{SMatrix{N,N},Nothing},
    ::Val{I1}) where {TT,C,D,N,I1}
    # Split the matrix into the column we need now, and the rest.
    (u, u_column...) = isnothing(m) ? (nothing, nothing) : Tuple(m[:, 1])
    (w, w_column...) = isnothing(σ) ? (nothing, nothing) : Tuple(σ[:, 1])
    # Type-stable way to subset SMatrix:
    m_rest = isnothing(m) ? nothing : SMatrix{N-1,N-1}(view(m, 2:N, 2:N))
    σ_rest = isnothing(σ) ? nothing : SMatrix{N-1,N-1}(view(σ, 2:N, 2:N))
    
    return (HubbardMomSpaceComponentData{TT,C,I1,I1,D}(h.geometry, address, a, a, u, w), 
        _mom_interactions_col(h,address,a,as,u_column,w_column,Val(I1),Val(I1+1))...,
        _column_components(h, address, as, m_rest, σ_rest, Val(I1+1) )...,)
end

@inline _mom_interactions_col(
    ::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress,::Tuple{}, 
    ::Tuple{}, ::Tuple{}, ::Val, ::Val
) = ()

@inline function _mom_interactions_col(
    h::HubbardMomSpace{TT,C,D}, address::AbstractFockAddress, a::SingleComponentFockAddress, 
    (b,as...)::NTuple{N}, (u, us...)::NTuple{N}, (w, ws...)::NTuple{N}, ::Val{I1}, ::Val{I2}
) where {TT,C,D,N,I1,I2}

    return (HubbardMomSpaceComponentData{TT,C,I1,I2,D}(h.geometry, address, a, b, u, w), 
            _mom_interactions_col(h, address, a, as, us, ws, Val(I1), Val(I2+1))...)
end

@inline _mom_interactions_col(
    ::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress, ::Tuple{},
    ::Tuple{Nothing}, ::Tuple{}, ::Val, ::Val
) = ()

@inline function _mom_interactions_col(
    h::HubbardMomSpace{TT,C,D}, address::AbstractFockAddress, a::SingleComponentFockAddress, 
    (b,as...)::NTuple{N}, m::Tuple{Nothing}, σ::NTuple{N}, ::Val{I1}, ::Val{I2}
) where {TT,C,D,N,I1,I2}

    return (HubbardMomSpaceComponentData{TT,C,I1,I2,D}(h.geometry, address, a, b, m[1], σ[1]), 
            _mom_interactions_col(h, address, a, as, m, σ[2:N], Val(I1), Val(I2+1))...)
end

@inline _mom_interactions_col(
    ::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress, ::Tuple{},
    ::Tuple{}, ::Tuple{Nothing}, ::Val, ::Val
) = ()

@inline function _mom_interactions_col(
    h::HubbardMomSpace{TT,C,D}, address::AbstractFockAddress, a::SingleComponentFockAddress, 
    (b,as...)::NTuple{N}, m::Tuple{N}, σ::Tuple{Nothing}, ::Val{I1}, ::Val{I2}
) where {TT,C,D,N,I1,I2}

    return (HubbardMomSpaceComponentData{TT,C,I1,I2,D}(h.geometry, address, a, b, m[1], σ[1]), 
            _mom_interactions_col(h, address, a, as, m[2:N], σ, Val(I1), Val(I2+1))...)
end

@inline _mom_interactions_col(
    ::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress, ::Tuple{},
    ::Tuple{Nothing}, ::Tuple{Nothing}, ::Val, ::Val
) = ()

@inline function _mom_interactions_col(
    h::HubbardMomSpace{TT,C,D}, address::AbstractFockAddress, a::SingleComponentFockAddress, 
    (b,as...)::NTuple{N}, m::Tuple{Nothing}, σ::Tuple{Nothing}, ::Val{I1}, ::Val{I2}
) where {TT,C,D,N,I1,I2}

    return (HubbardMomSpaceComponentData{TT,C,I1,I2,D}(h.geometry, address, a, b, m[1], σ[1]), 
            _mom_interactions_col(h, address, a, as, m, σ, Val(I1), Val(I2+1))...)
end

# Split one-dimensional array `index` that indexes over many components simultaneously into
# a two-dimensional one. The dimension of the new index picks the component, while the
# second picks the offdiagonal within the component.

function random_offdiagonal(column::HubbardMomSpaceColumn)
    random_number = rand(1:column.num_offdiagonals)
    component, remainder = _split_component_from_index(column, random_number)

    addr, val = index_apply(getindex, column.components, component, remainder)
    return addr, 1/column.num_offdiagonals, val
end

struct HubbardMomSpaceColumnOffdiagonals{TT,A,G,C<:Tuple} <: AbstractVector{Pair{A,TT}}
    address::A
    geometry::G
    components::C
    num_offdiagonals::Int
end

function offdiagonals(column::HubbardMomSpaceColumn{TT,<:Any,G,A,C}) where {TT,G,A,C}
    return HubbardMomSpaceColumnOffdiagonals{TT,A,G,C}(
        column.address,
        column.hamiltonian.geometry,
        column.components,
        column.num_offdiagonals,
    )
end

@inline function Base.iterate(ods::HubbardMomSpaceColumnOffdiagonals, state=(1,1))
    component_index, chosen = state
    component_index = _test_interaction_coeff(ods, component_index)
    if chosen > index_apply(size, ods.components, component_index)[1]
        chosen = 1
        component_index += 1
    end
    if component_index > length(ods.components)
        return nothing
    else
        result = index_apply(
            getindex, ods.components, component_index, chosen
        )
        return result, (component_index, chosen + 1)
    end
end

@inline function _test_interaction_coeff(ods::HubbardMomSpaceColumnOffdiagonals, component_index::Int)
    if index_apply(size, ods.components, component_index)[1] == 0 && component_index < length(ods.components)
        return _test_interaction_coeff(ods, component_index + 1)
    else
        return component_index
    end
end

Base.size(ods::HubbardMomSpaceColumnOffdiagonals) = (ods.num_offdiagonals,)
Base.eltype(::HubbardMomSpaceColumnOffdiagonals{TT,A}) where {TT,A} = Pair{A,TT}

function Base.getindex(column::HubbardMomSpaceColumnOffdiagonals, index)
    component_index, inner_index = _split_component_from_index(column, index)
    return index_apply(getindex, column.components, component_index, inner_index)
end

########################################################################################################
# Momentum operator in momentum space
########################################################################################################

struct MomentumMomSpace{T,C,D,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{SVector{D,T}}
    ham::H
end
LOStructure(::Type{MomentumMomSpace}) = IsDiagonal()
num_offdiagonals(::MomentumMomSpace, _) = 0
function diagonal_element(mom::MomentumMomSpace{T,1,D}, address::SingleComponentFockAddress) where {T,D}
    return SVector{D,T}(dot(mom.ham.ks_mat[i, :], occupied_mode_map(address)) for i in 1:D)
end
function diagonal_element(mom::MomentumMomSpace{T,C,D}, address::CompositeFS) where {T,C,D}
    return SVector{D,T}(sum(dot(mom.ham.ks_mat[i, :], occupied_mode_map(c)) for c in address.components) for i in 1:D)
end
# fold into (-π, π]
starting_address(mom::MomentumMomSpace) = starting_address(mom.ham)
dimension(mom::MomentumMomSpace, _) = 1

momentum(ham::HubbardMomSpace{T,C,D}) where {T,C,D} = MomentumMomSpace{T,C,D,typeof(ham)}(ham)
