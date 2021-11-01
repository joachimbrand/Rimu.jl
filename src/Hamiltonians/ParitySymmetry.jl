"""
    left_weight(address)

Compute how left-heavy the `address` is. A `left_weight` of 0 indicates that the address
is symmetric. `left_weight(reverse(address)) = -left_weight(address)`.
"""
function left_weight(f::Union{BoseFS,FermiFS})
    M = num_modes(f)
    midpoint = cld(M, 2)
    weight = 0
    for index in occupied_modes(f)
        weight += (index.mode - midpoint) * index.occnum
    end
    return weight
end
function left_weight(f::CompositeFS)
    return _left_weight(f.components)
end
@inline _left_weight(::Tuple{}) = 0
@inline function _left_weight((c, cs...))
    weight = left_weight(c)
    if weight == 0
        return _left_weight(cs)
    else
        return weight
    end
end

"""
    ParitySymmetry(ham::AbstractHamiltonian{T}; even=true) <: AbstractHamiltonian{T}

Changes the [`offdiagonals`](@ref) of a Hamiltonian in a way that produces even or odd
eigenfunctions. For some Hamiltonians, this reduces the size of the Hilbert space by half.

Whether the even or odd part is to be used is controlled by the keyword argument `even`.

# Notes

* This modifier only works on addresses with an even number of modes.
* For the odd parity symmetry, the starting address of the underlying Hamiltonian can not be
  symmetric.

```jldoctest
julia> ham = HubbardReal1D(BoseFS((0,2,1)))
HubbardReal1D(BoseFS{3,3}((0, 2, 1)); u=1.0, t=1.0)

julia> size(Matrix(ham))
(10, 10)

julia> size(Matrix(ParitySymmetry(ham)))
(6, 6)

julia> size(Matrix(ParitySymmetry(ham; odd=true)))
(4, 4)

julia> eigvals(Matrix(ham))[1] ≈ eigvals(Matrix(ParitySymmetry(ham)))[1]
true
```
"""
struct ParitySymmetry{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    hamiltonian::H
    even::Bool
    direction::Int8
end

function ParitySymmetry(hamiltonian; odd=false, even=!odd)
    address = starting_address(hamiltonian)

    if !isodd(num_modes(address))
        throw(ArgumentError("Starting address must have an odd number of modes"))
    end
    weight = left_weight(address)
    direction = weight ≥ 0 ? -1 : 1
    if !even && weight == 0
        throw(ArgumentError("Even starting address can't be used with odd `ParitySymmetry`"))
    end
    return ParitySymmetry(hamiltonian, even, Int8(direction))
end

function Base.show(io::IO, h::ParitySymmetry)
    print(io, "ParitySymmetry(", h.hamiltonian, ", even=", h.even, ")")
end

for op in (:LOStructure, :starting_address)
    @eval $op(h::ParitySymmetry) = $op(h.hamiltonian)
end

function Base.adjoint(h::ParitySymmetry)
    return ParitySymmetry(adjoint(h.hamiltonian); even=h.even)
end

get_offdiagonal(h::ParitySymmetry, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::ParitySymmetry, add) = num_offdiagonals(h.hamiltonian, add)

struct ParitySymmetryOffdiagonals{
    A,T,O<:AbstractVector{Tuple{A,T}}
} <: AbstractOffdiagonals{A,T}
    od::O
    even::Bool
    direction::Int8
end
Base.size(o::ParitySymmetryOffdiagonals) = size(o.od)

function offdiagonals(h::ParitySymmetry, add)
    return ParitySymmetryOffdiagonals(offdiagonals(h.hamiltonian, add), h.even, h.direction)
end

function Base.getindex(o::ParitySymmetryOffdiagonals, i)
    add, val = o.od[i]
    weight = o.direction * left_weight(add)
    if weight > 0
        add = reverse(add)
        val *= ifelse(o.even, 1, -1)
    elseif weight == 0 && !o.even
        # The value of even addresses for odd paritysymmetry should always be zero.
        val = zero(val)
    end
    @assert left_weight(add) * o.direction ≤ 0
    return add, val
end
function diagonal_element(h::ParitySymmetry, add)
    return diagonal_element(h.hamiltonian, add)
end
