"""
    ExtendedHubbardReal1D(address; u=1.0, v=1.0, t=1.0)

Implements the extended Hubbard model on a one-dimensional chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1) + v \\sum_{\\langle i,j\\rangle} n_i n_j
```

# Arguments

* `address`: the starting address.
* `u`: on-site interaction parameter
* `v`: the next-neighbor interaction
* `t`: the hopping strength

"""
struct ExtendedHubbardReal1D{TT,A<:SingleComponentFockAddress,U,V,T,PITWISTED,HARDWALL} <: AbstractHamiltonian{TT}
    add::A
end

# addr for compatibility.
function ExtendedHubbardReal1D(addr; u=1.0, v=1.0, t=1.0, pitwisted::Bool = false, hardwall::Bool = false)
    if (pitwisted,hardwall)==(true,true)
        throw(ArgumentError("Hardwall and pitwisted boundaries cannot be called together"))
    else
        U, V, T = promote(float(u), float(v), float(t))
        return ExtendedHubbardReal1D{typeof(U),typeof(addr),U,V,T,pitwisted,hardwall}(addr)
    end
end

function Base.show(io::IO, h::ExtendedHubbardReal1D)
    
    if h.pitwisted
        print(io, "ExtendedHubbardReal1D($(h.add); u=$(h.u), v=$(h.v), t=$(h.t), pitwisted=$(h.pitwisted))")
    elseif h.hardwall
        print(io, "ExtendedHubbardReal1D($(h.add); u=$(h.u), v=$(h.v), t=$(h.t), hardwall=$(h.hardwall))")
    else
        print(io, "ExtendedHubbardReal1D($(h.add); u=$(h.u), v=$(h.v), t=$(h.t))")
    end

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
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,<:Any,PITWISTED}, ::Val{:pitwisted}) where PITWISTED=PITWISTED
Base.getproperty(h::ExtendedHubbardReal1D{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,HARDWALL}, ::Val{:hardwall}) where HARDWALL=HARDWALL

function num_offdiagonals(::ExtendedHubbardReal1D, address::SingleComponentFockAddress)
    return 2 * num_occupied_modes(address)
end

"""
    extended_hubbard_interaction(address, m)

Compute the on-site product sum_j n_j(n_j-1) and the next neighbour term
sum_j n_j n_{j+1} with periodic boundary conditions.
"""
function extended_hubbard_interaction(h::ExtendedHubbardReal1D,b::SingleComponentFockAddress)
    omm = OccupiedModeMap(b)

    prev = zero(eltype(omm))
    ext_result = 0
    reg_result = 0
    for curr in omm
        ext_result += ifelse(prev.mode == curr.mode - 1, prev.occnum * curr.occnum, 0)
        reg_result += curr.occnum * (curr.occnum - 1)
        prev = curr
    end
    if h.hardwall == false
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
    naddress, onproduct = hopnextneighbour(add, chosen; pitwisted = h.pitwisted, hardwall = h.hardwall)
    return naddress, - h.t * onproduct
end
