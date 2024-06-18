"""
    ExtendedHubbardReal1D(address; u=1.0, v=1.0, t=1.0, boundary_condition=:periodic)

Implements the extended Hubbard model on a one-dimensional chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1) + v \\sum_{\\langle i,j\\rangle} n_i n_j
```

# Arguments

* `address`: the starting address.
* `u`: on-site interaction parameter
* `v`: the next-neighbor interaction
* `t`: the hopping strength
* `boundary_condition` : the boundary condition to apply. Can be one of `:periodic`, `:twisted`, or `:hard_wall`

"""
struct ExtendedHubbardReal1D{TT,A<:SingleComponentFockAddress,U,V,T,BOUNDARY_CONDITION} <: AbstractHamiltonian{TT}
    add::A
end

# addr for compatibility.
function ExtendedHubbardReal1D(addr; u=1.0, v=1.0, t=1.0, boundary_condition = :periodic)
    if h.boundary_condition == :periodic || h.boundary_condition == :twisted || h.boundary_condition == :hard_wall
        U, V, T = promote(float(u), float(v), float(t))
        return ExtendedHubbardReal1D{typeof(U),typeof(addr),U,V,T,boundary_condition}(addr)
    else
        throw(ArgumentError("invalid boundary condition"))
    end
end

function Base.show(io::IO, h::ExtendedHubbardReal1D)
    print(io, "ExtendedHubbardReal1D($(h.add); u=$(h.u), v=$(h.v), t=$(h.t), boundary_condition=:$(h.boundary_condition))")
end

function starting_address(h::ExtendedHubbardReal1D)
    return getfield(h, :add)
end

dimension(::ExtendedHubbardReal1D, address) = number_conserving_dimension(address)

LOStructure(::Type{<:ExtendedHubbardReal1D{<:Real}}) = IsHermitian()

Base.getproperty(h::ExtendedHubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::ExtendedHubbardReal1D, ::Val{:add}) = getfield(h, :add)
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,V}, ::Val{:v}) where V = V
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,<:Any,BOUNDARY_CONDITION}, ::Val{:boundary_condition}) where BOUNDARY_CONDITION = BOUNDARY_CONDITION

function num_offdiagonals(::ExtendedHubbardReal1D, address::SingleComponentFockAddress)
    return 2 * num_occupied_modes(address)
end

"""
    extended_hubbard_interaction(address, m)

Compute the on-site product sum_j n_j(n_j-1) and the next neighbour term
sum_j n_j n_{j+1} with respective boundary conditions.
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
    ebhinteraction, bhinteraction = extended_hubbard_interaction(h,b)
    return h.u * bhinteraction / 2 + h.v * ebhinteraction
end

function get_offdiagonal(h::ExtendedHubbardReal1D, add::SingleComponentFockAddress, chosen)
    naddress, onproduct = hopnextneighbour(add, chosen, Val(h.boundary_condition))

    return naddress, - h.t * onproduct
end
