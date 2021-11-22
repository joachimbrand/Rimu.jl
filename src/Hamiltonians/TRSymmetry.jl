"""
    TRSymmetry(ham::AbstractHamiltonian{T}; even=true) <: AbstractHamiltonian{T}

Impose even or odd time reversal on all states and the Hamiltonian `ham` as controlled by
the keyword argument `even`.

# Notes

* This modifier only works two component [`starting_address`](@ref)s.
* For odd time reversal symmetry, the [`starting_address`](@ref) of the underlying
  Hamiltonian must not be symmetric.
* If time reversal symmetry is not a symmetry of the Hamiltonian `ham` then the result is
  undefined.
* `TRSymmetry` works by modifying the [`offdiagonals`](@ref) iterator.

```jldoctest
julia> ham = HubbardMom1D(FermiFS2C((1,0,1),(0,1,1)));

julia> size(Matrix(ham))
(3, 3)

julia> size(Matrix(TRSymmetry(ham)))
(2, 2)

julia> size(Matrix(TRSymmetry(ham, even=false)))
(1, 1)

julia> eigvals(Matrix(TRSymmetry(ham)))[1] ≈ eigvals(Matrix(ham))[1]
true

julia> ham = BoseHubbardMom1D2C(BoseFS2C((1,0,1),(0,1,1)));

julia> size(Matrix(ham))
(12, 12)

julia> size(Matrix(TRSymmetry(ham)))
(7, 7)

julia> size(Matrix(TRSymmetry(ham, even=false)))
(5, 5)

julia> eigvals(Matrix(TRSymmetry(ham, even=false)))[1] ≈ eigvals(Matrix(ham))[1]
true
```
"""
struct TRSymmetry{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    hamiltonian::H
    even::Bool
end

function TRSymmetry(hamiltonian; odd=false, even=!odd)
    address = starting_address(hamiltonian)
    check_tr_address(address)
    if !even && address == time_reverse(address)
        throw(ArgumentError("Even starting address can't be used with odd `TRSymmetry`"))
    end
    return TRSymmetry(hamiltonian, even)
end

function check_tr_address(addr::CompositeFS)
    if num_components(addr) ≠ 2 || !isequal(num_particles.(addr.components)...)
        throw(ArgumentError("Two component address with equal particle numbers required for `TRSymmetry`."))
    end
end
function check_tr_address(addr::BoseFS2C{NA,NB,M,SA,SB,N}) where {NA,NB,M,SA,SB,N}
    if NA ≠ NB || SA ≠ SB
        throw(ArgumentError("Two component address with equal particle numbers required for `TRSymmetry`."))
    end
end


function Base.show(io::IO, h::TRSymmetry)
    print(io, "TRSymmetry(", h.hamiltonian, ", even=", h.even, ")")
end

LOStructure(h::TRSymmetry) = LOStructure(h.hamiltonian)
function starting_address(h::TRSymmetry)
    add = starting_address(h.hamiltonian)
    return min(add, time_reverse(add))
end

function Base.adjoint(h::TRSymmetry)
    return TRSymmetry(adjoint(h.hamiltonian); even=h.even)
end

get_offdiagonal(h::TRSymmetry, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::TRSymmetry, add) = num_offdiagonals(h.hamiltonian, add)

struct TRSymmetryOffdiagonals{
    A,T,O<:AbstractVector{Tuple{A,T}}
} <: AbstractOffdiagonals{A,T}
    od::O
    even::Bool
end
Base.size(o::TRSymmetryOffdiagonals) = size(o.od)

function offdiagonals(h::TRSymmetry, add)
    return TRSymmetryOffdiagonals(offdiagonals(h.hamiltonian, add), h.even)
end

function Base.getindex(o::TRSymmetryOffdiagonals, i)
    add, val = o.od[i]
    rev_add = time_reverse(add)
    left_add = min(rev_add, add)
    if !o.even && left_add ≠ add
        val *= -1
    elseif !o.even && rev_add == add
        val = zero(val)
    end
    return left_add, val
end
function diagonal_element(h::TRSymmetry, add)
    return diagonal_element(h.hamiltonian, add)
end
