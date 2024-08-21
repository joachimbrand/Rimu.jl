"""
    ExtendedHubbardReal1D(address; u=1.0, v=1.0, t=1.0, boundary_condition=:periodic)

Implements the extended Hubbard model on a one-dimensional chain in real space. This
Hamiltonian can be either real or complex, depending on the choice of `boundary_condition`.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1) +
v \\sum_{\\langle i,j\\rangle} n_i n_j
```

# Arguments

* `address`: the starting address.
* `u`: on-site interaction parameter
* `v`: the next-neighbor interaction
* `t`: the hopping strength
* `boundary_condition` The following values are supported:
  * `:periodic`: usual period boundary condition realising a ring geometry.
  * `:hard_wall`: hopping over the boundary is not allowed.
  * `:twisted`: like `:periodic` but hopping over the boundary incurs an additional factor
    of `-1`.
  * `θ <: Number`: like `:periodic` and `:twisted` but hopping over the boundary incurs a
    factor ``\\exp(iθ)`` for a hop to the right and ``\\exp(−iθ)`` for a hop to the left.
    With this choice the Hamiltonian will have a complex `eltype` whereas otherwise the
    `eltype` is determined by the type of the parameters `t`, `u`, and `v`.

See also [`HubbardRealSpace`](@ref).
"""
struct ExtendedHubbardReal1D{TT,A<:SingleComponentFockAddress,U,V,T,BOUNDARY_CONDITION} <: AbstractHamiltonian{TT}
    address::A
end

# addr for compatibility.
function ExtendedHubbardReal1D(addr; u=1.0, v=1.0, t=1.0, boundary_condition = :periodic)
    if boundary_condition == :periodic || boundary_condition == :twisted || boundary_condition == :hard_wall
        U, V, T = promote(float(u), float(v), float(t))
        return ExtendedHubbardReal1D{typeof(U),typeof(addr),U,V,T,boundary_condition}(addr)
    elseif boundary_condition isa Number
        U, V, T = promote(float(u), float(v), float(t))
        return ExtendedHubbardReal1D{typeof(complex(U)),typeof(addr),U,V,T,boundary_condition}(addr)
    else
        throw(ArgumentError("invalid boundary condition"))
    end
end

function Base.show(io::IO, h::ExtendedHubbardReal1D)
    compact_addr = repr(h.address, context=:compact => true) # compact print address
    print(io, "ExtendedHubbardReal1D($(compact_addr); u=$(h.u), v=$(h.v), t=$(h.t), ")
    print(io, "boundary_condition=$(repr(h.boundary_condition)))")
end

function starting_address(h::ExtendedHubbardReal1D)
    return getfield(h, :address)
end

dimension(::ExtendedHubbardReal1D, address) = number_conserving_dimension(address)

function LOStructure(::Type{<:ExtendedHubbardReal1D{<:Real,<:Any,<:Any,<:Any,T}}) where T
    if iszero(T)
        return IsDiagonal()
    else
        return IsHermitian()
    end
end
function LOStructure(::Type{<:ExtendedHubbardReal1D{<:Complex,<:Any,U,V,T}}) where {U,V,T}
    if iszero(T)
        return IsDiagonal() # TODO: implement adjoint
    elseif iszero(imag(U)) && iszero(imag(V))
        return IsHermitian() # still Hermitian with complex t and twisted boundaries
    else
        return AdjointUnknown() # diagonal elements are complex; TODO: implement adjoint
    end
end

Base.getproperty(h::ExtendedHubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::ExtendedHubbardReal1D, ::Val{:address}) = getfield(h, :address)
Base.getproperty(::ExtendedHubbardReal1D{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,V}, ::Val{:v}) where V = V
Base.getproperty(::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T
function Base.getproperty(
    ::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,<:Any,BOUNDARY_CONDITION},
    ::Val{:boundary_condition}
) where BOUNDARY_CONDITION
    BOUNDARY_CONDITION
end

function num_offdiagonals(::ExtendedHubbardReal1D, address::SingleComponentFockAddress)
    return 2 * num_occupied_modes(address)
end

"""
    extended_hubbard_interaction(h::ExtendedHubbardReal1D, address)

Compute and return both the nearest neighbor occupation number product
``\\sum_j n_j n_{j+1}`` (according to the boundary conditions of `h`) as well as the on-site
product ``\\sum_j n_j (n_j - 1)`` treating the `address` as a one-dimensional chain.

See [`ExtendedHubbardReal1D`](@ref) and [`hopnextneighbour`](@ref).
"""
function extended_hubbard_interaction(h::ExtendedHubbardReal1D, b::SingleComponentFockAddress)
    omm = OccupiedModeMap(b)

    prev = zero(eltype(omm))
    ext_result = 0
    reg_result = 0
    for curr in omm
        ext_result += ifelse(prev.mode == curr.mode - 1, prev.occnum * curr.occnum, 0)
        reg_result += curr.occnum * (curr.occnum - 1)
        prev = curr
    end

    if h.boundary_condition != :hard_wall
        # Handle periodic boundaries
        last = ifelse(omm[end].mode == num_modes(b), omm[end], zero(eltype(omm)))
        first = ifelse(omm[1].mode == 1, omm[1], zero(eltype(omm)))
        ext_result += last.occnum * first.occnum
    end

    return ext_result, reg_result
end

function diagonal_element(h::ExtendedHubbardReal1D, b::SingleComponentFockAddress)
    ebhinteraction, bhinteraction = extended_hubbard_interaction(h, b)
    return convert(eltype(h), h.u * bhinteraction / 2 + h.v * ebhinteraction)
end

function get_offdiagonal(h::ExtendedHubbardReal1D, address::SingleComponentFockAddress, chosen)
    naddress, onproduct = hopnextneighbour(address, chosen, h.boundary_condition)
    if h.t isa Complex && chosen%2 == 0
        return naddress, convert(eltype(h), - conj(h.t) * onproduct)
    else
        return naddress, convert(eltype(h), - h.t * onproduct)
    end
end
