"""
    ParitySymmetry(ham::AbstractHamiltonian{T}; even=true) <: AbstractHamiltonian{T}

Impose even or odd parity on all states and the Hamiltonian `ham` as controlled by the
keyword argument `even`. For some Hamiltonians, this reduces the size of the Hilbert space
by half.

`ParitySymmetry` changes the basis set to states with defined parity. Effectively, a
non-even address ``|α⟩`` is replaced by ``\\frac{1}{√2}(|α⟩ ± |ᾱ⟩)`` for even and odd
parity, respectively, where `ᾱ == reverse(α)`.

# Notes

* This modifier currently only works on [`starting_address`](@ref)s with an odd number of
  modes.
* For odd parity, the [`starting_address`](@ref) of the underlying Hamiltonian cannot be
  symmetric.
* If parity is not a symmetry of the Hamiltonian `ham` then the result is undefined.
* `ParitySymmetry` works by modifying the [`offdiagonals`](@ref) iterator.

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
end

function ParitySymmetry(hamiltonian; odd=false, even=!odd)
    address = starting_address(hamiltonian)
    if !isodd(num_modes(address))
        throw(ArgumentError("Starting address must have an odd number of modes"))
    end
    if !even && address == reverse(address)
        throw(ArgumentError("Even starting address can't be used with odd `ParitySymmetry`"))
    end
    return ParitySymmetry(hamiltonian, even)
end

function Base.show(io::IO, h::ParitySymmetry)
    print(io, "ParitySymmetry(", h.hamiltonian, ", even=", h.even, ")")
end

LOStructure(h::ParitySymmetry) = LOStructure(h.hamiltonian)
function starting_address(h::ParitySymmetry)
    add = starting_address(h.hamiltonian)
    return min(add, reverse(add))
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
end
Base.size(o::ParitySymmetryOffdiagonals) = size(o.od)

function offdiagonals(h::ParitySymmetry, add)
    return ParitySymmetryOffdiagonals(offdiagonals(h.hamiltonian, add), h.even)
end

function Base.getindex(o::ParitySymmetryOffdiagonals, i)
    add, val = o.od[i]
    rev_add = reverse(add)
    left_add = min(rev_add, add)
    if !o.even && left_add ≠ add
        val *= -1
    elseif !o.even && rev_add == add
        val = zero(val)
    end
    return left_add, val
end
function diagonal_element(h::ParitySymmetry, add)
    return diagonal_element(h.hamiltonian, add)
end
