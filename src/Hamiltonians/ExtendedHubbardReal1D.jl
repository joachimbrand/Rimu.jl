"""
    ExtendedHubbardReal1D(address; u=1.0, v=1.0, t=1.0, boundary_condition=:periodic, power=nothing)

Implements the extended Hubbard model on a one-dimensional chain in real space. This
Hamiltonian can be either real or complex, depending on the choice of `boundary_condition`.

```math
\\hat{H} = - \\sum_i \\left(t a_i^† a_{i+1} + t^* a_{i+1}^† a_i \\right) + 
\\frac{u}{2}\\sum_i n_i (n_i-1) + v \\sum_{i,j>i} f_{j-i} n_i n_j
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
* `power`: the interaction type. The following values are supported:
  * `nothing`: nearest neighbour interaction (default), i.e. ``f_{j-i} = δ_{j-i,1}``.
  * `p<:Number`: inverse distance interaction, i.e. ``f_{j-i} = (j-i)^{-p}``.

See also [`HubbardRealSpace`](@ref).
"""
struct ExtendedHubbardReal1D{TT,A<:SingleComponentFockAddress,U,V,T,BOUNDARY_CONDITION,POWER} <: AbstractHamiltonian{TT}
    address::A
end

# addr for compatibility.
function ExtendedHubbardReal1D(addr; u=1.0, v=1.0, t=1.0, boundary_condition=:periodic, power = nothing)
    if power isa Number || power === nothing
        if boundary_condition == :periodic || boundary_condition == :twisted || boundary_condition == :hard_wall
            U, V, T = promote(float(u), float(v), float(t))
            return ExtendedHubbardReal1D{typeof(U),typeof(addr),U,V,T,boundary_condition,power}(addr)
        elseif boundary_condition isa Number
            U, V, T = complex.(promote(float(u), float(v), float(t)))
            return ExtendedHubbardReal1D{typeof(U),typeof(addr),U,V,T,boundary_condition,power}(addr)
        else
            throw(ArgumentError("invalid boundary condition"))
        end
    else
         throw(ArgumentError("invalid interaction"))
    end
end

function Base.show(io::IO, h::ExtendedHubbardReal1D)
    compact_addr = repr(h.address, context=:compact => true) # compact print address
    print(io, "ExtendedHubbardReal1D($(compact_addr); u=$(h.u), v=$(h.v), t=$(h.t), ")
    print(io, "boundary_condition=$(repr(h.boundary_condition)), power=$(repr(h.power)))")
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
function LOStructure(::Type{<:ExtendedHubbardReal1D{TT,<:Any,U,V,T}}) where {TT<:Complex,U,V,T}
    if iszero(T)
        return IsDiagonal()
    elseif iszero(imag(U)) && iszero(imag(V))
        return IsHermitian()
    else
        return AdjointKnown()
    end
end

function LinearAlgebra.adjoint(h::ExtendedHubbardReal1D{TT,A,U,V,T,B,I}) where {TT<:Complex,A,U,V,T,B,I}
    return ExtendedHubbardReal1D{TT,A,conj(U)+0im,conj(V)+0im,T,B,I}(h.address)
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
function Base.getproperty(
    ::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,POWER},
    ::Val{:power}
) where POWER
    POWER
end

function num_offdiagonals(::ExtendedHubbardReal1D, address::SingleComponentFockAddress)
    return 2 * num_occupied_modes(address)
end

"""
    extended_hubbard_interaction(h::ExtendedHubbardReal1D, address, power)

Compute and return both the extended range occupation number product
``\\sum_{i,j>i} f_{j-i} n_i n_{j}`` (according to the boundary conditions of `h`) as well as the on-site
product ``\\sum_j n_j (n_j - 1)`` treating the `address` as a one-dimensional chain.

where ``f_{j-i} = 1`` for nearest neighbors (power = nothing) and 
``f_{j-i} = (|j-i|)^{-p}`` for inverse distance interaction (power = p<:Number).

See [`ExtendedHubbardReal1D`](@ref) and [`hopnextneighbour`](@ref).
"""
function extended_hubbard_interaction(h::ExtendedHubbardReal1D, b::SingleComponentFockAddress, ::Nothing)
    omm = occupied_mode_map(b)

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

function extended_hubbard_interaction(h::ExtendedHubbardReal1D, b::SingleComponentFockAddress, power::Number)
    omm = occupied_mode_map(b)
    M = num_modes(b)
    ext_result = 0
    reg_result = 0
    for i in 1:length(omm)
        occ_i = omm[i].occnum
        reg_result += occ_i * (occ_i - 1)
        for j in 1:i-1
            occ_j = omm[j].occnum
            m_ij = omm[i].mode - omm[j].mode
            if (m_ij > M/2 && h.boundary_condition != :hard_wall)
                ext_result += occ_i * occ_j/ ((M - m_ij)^(power))
            else
                ext_result += occ_i * occ_j/ (m_ij^(power))
            end
        end
    end
    return ext_result, reg_result
end

function diagonal_element(h::ExtendedHubbardReal1D, b::SingleComponentFockAddress)
    ebhinteraction, bhinteraction = extended_hubbard_interaction(h, b, h.power)
    return convert(eltype(h), h.u * bhinteraction / 2 + h.v * ebhinteraction)
end

function get_offdiagonal(h::ExtendedHubbardReal1D, address::SingleComponentFockAddress, chosen)
    return _get_offdiagonal_hubbard_real_1D(h, address, chosen)
end
