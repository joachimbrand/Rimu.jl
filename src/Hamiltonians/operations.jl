###
### This file contains operations defined on Hamiltonians
###
function Base.getindex(ham::AbstractHamiltonian{T}, address1, address2) where T
    # calculate the matrix element when only two bitstring addresses are given
    # this is NOT used for the QMC algorithm and is currenlty not used either
    # for building the matrix for conventional diagonalisation.
    # Only used for verifying matrix.
    # This will be slow and inefficient. Avoid using for larger Hamiltonians!
    address1 == address2 && return diagonal_element(ham, address1) # diagonal
    for (add,val) in offdiagonals(ham, address2) # off-diag column as iterator
        add == address1 && return val # found address1
    end
    return zero(T) # address1 not found
end

LinearAlgebra.adjoint(op::AbstractHamiltonian) = adjoint(LOStructure(op), op)

"""
    adjoint(::LOStructure, op::AbstractHamiltonian)

Represent the adjoint of an [`AbstractHamiltonian`](@ref). Extend this method to define custom
adjoints.
"""
function LinearAlgebra.adjoint(::S, op) where {S<:LOStructure}
    error(
        "`adjoint()` not defined for `AbstractHamiltonian`s with `LOStructure` `$(S)`. ",
        " Is your Hamiltonian hermitian?"
    )
end

LinearAlgebra.adjoint(::IsHermitian, op) = op # adjoint is known
LinearAlgebra.adjoint(::IsDiagonal, op) = op
