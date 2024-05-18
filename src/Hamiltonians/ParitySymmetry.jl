"""
    ParitySymmetry(ham::AbstractHamiltonian{T}; even=true) <: AbstractHamiltonian{T}

Impose even or odd parity on all states and the Hamiltonian `ham` as controlled by the
keyword argument `even`. Parity symmetry of the Hamiltonian is assumed.
For some Hamiltonians, `ParitySymmetry` reduces the size of the Hilbert space
by half.

`ParitySymmetry` performs a unitary
transformation, leaving the eigenvalues unchanged and preserving the [`LOStructure`](@ref).
This is achieved by changing the basis set to states with defined parity. Effectively, a
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
julia> ham = HubbardReal1D(BoseFS(0,2,1))
HubbardReal1D(fs"|0 2 1⟩"; u=1.0, t=1.0)

julia> size(Matrix(ham))
(10, 10)

julia> size(Matrix(ParitySymmetry(ham)))
(6, 6)

julia> size(Matrix(ParitySymmetry(ham; odd=true)))
(4, 4)

julia> eigvals(Matrix(ham))[1] ≈ eigvals(Matrix(ParitySymmetry(ham)))[1]
true
```
See also [`TimeReversalSymmetry`](@ref).
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
Base.adjoint(h::ParitySymmetry) = ParitySymmetry(h.hamiltonian', even=h.even)

function starting_address(h::ParitySymmetry)
    add = starting_address(h.hamiltonian)
    return min(add, reverse(add))
end

dimension(h::ParitySymmetry, addr) = dimension(h.hamiltonian, addr) # upper bound

get_offdiagonal(h::ParitySymmetry, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::ParitySymmetry, add) = num_offdiagonals(h.hamiltonian, add)

struct ParitySymmetryOffdiagonals{
    A,T,O<:AbstractVector{Tuple{A,T}}
} <: AbstractOffdiagonals{A,T}
    add::A
    add_even::Bool
    od::O
    even::Bool
end
Base.size(o::ParitySymmetryOffdiagonals) = size(o.od)

function offdiagonals(h::ParitySymmetry, add)
    add_even = add == reverse(add)
    return ParitySymmetryOffdiagonals(add, add_even, offdiagonals(h.hamiltonian, add), h.even)
end

function Base.getindex(o::ParitySymmetryOffdiagonals, i)
    in = o.add
    out, val = o.od[i]

    rev_out = reverse(out)
    in_even = o.add_even
    out_even = out == rev_out

    if in_even && !out_even
        new_val = 1/√2 * val
    elseif out_even && !in_even
        new_val = √2 * val
    else
        new_val = float(val)
    end

    left_out = min(rev_out, out)
    if !o.even && left_out ≠ out
        new_val = -new_val
    elseif !o.even && out_even
        new_val = zero(new_val)
    end
    return left_out, new_val
end

function diagonal_element(h::ParitySymmetry, add)
    return diagonal_element(h.hamiltonian, add)
end
