"""
    Stoquastic(ham <: AbstractHamiltonian) <: AbstractHamiltonian
A wrapper for an [`AbstractHamiltonian`](@ref) that replaces all off-diagonal matrix
elements `v` by `-abs(v)`, thus making the new Hamiltonian *stoquastic*.

A stoquastic Hamiltonian does not have a Monte Carlo sign problem. For a hermitian `ham`
the smallest eigenvalue of `Stoquastic(ham)` is ≤ the smallest eigenvalue of `ham`.
"""
struct Stoquastic{T,H} <: AbstractHamiltonian{T}
    hamiltonian::H
end

Stoquastic(h) = Stoquastic{eltype(h),typeof(h)}(h)

starting_address(h::Stoquastic) = starting_address(h.hamiltonian)

LOStructure(::Type{<:Stoquastic{<:Any,H}}) where {H} = LOStructure(H)
Base.adjoint(h::Stoquastic) = Stoquastic(h.hamiltonian')

dimension(h::Stoquastic, address) = dimension(h.hamiltonian, address)
num_offdiagonals(h::Stoquastic, address) = num_offdiagonals(h.hamiltonian, address)
diagonal_element(h::Stoquastic, address) = diagonal_element(h.hamiltonian, address)

function get_offdiagonal(h::Stoquastic, address, chosen)
    naddress, matrix_element = get_offdiagonal(h.hamiltonian, address, chosen)
    return naddress, -abs(matrix_element)
end
