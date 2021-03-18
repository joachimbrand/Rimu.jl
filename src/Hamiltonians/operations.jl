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

function Base.:*(h::AbstractHamiltonian{E}, v::AbstractDVec{K,V}) where {E, K, V}
    T = promote_type(E,V) # allow for type promotion
    w = empty(v, T) # allocate new vector; non-mutating version
    for (key,val) in pairs(v)
        w[key] += diagonal_element(h, key)*val
        for (add,elem) in offdiagonals(h, key)
            w[add] += elem*val
        end
    end
    return w
end

# three argument version
function LinearAlgebra.mul!(w::AbstractDVec, h::AbstractHamiltonian, v::AbstractDVec)
    empty!(w)
    for (key,val) in pairs(v)
        w[key] += diagonal_element(h, key)*val
        for (add,elem) in offdiagonals(h, key)
            w[add] += elem*val
        end
    end
    return w
end

"""
    dot(x, H::AbstractHamiltonian, v)

Evaluate `x⋅H(v)` minimizing memory allocations.
"""
function LinearAlgebra.dot(x::AbstractDVec, LO::AbstractHamiltonian, v::AbstractDVec)
    return dot(LOStructure(LO), x, LO, v)
end
# specialised method for UniformProjector
function LinearAlgebra.dot(::UniformProjector, LO::AbstractHamiltonian{T}, v::AbstractDVec{K,T2}) where {K, T, T2}
    result = zero(promote_type(T,T2))
    for (key,val) in pairs(v)
        result += diagonal_element(LO, key) * val
        for (add,elem) in offdiagonals(LO, key)
            result += elem * val
        end
    end
    return result
end

LinearAlgebra.dot(::LOStructure, x, LO::AbstractHamiltonian, v) = dot_from_right(x,LO,v)
# default for LOs without special structure: keep order

function LinearAlgebra.dot(::HermitianLO, x, LO::AbstractHamiltonian, v)
    if length(x) < length(v)
        return conj(dot_from_right(v,LO,x)) # turn args around to execute faster
    else
        return dot_from_right(x,LO,v) # original order
    end
end

"""
    Hamiltonians.dot_from_right(x, LO, v)
Internal function evaluates the 3-argument `dot()` function in order from right
to left.
"""
function dot_from_right(x::AbstractDVec{K,T1}, LO::AbstractHamiltonian{T}, v::AbstractDVec{K,T2}) where {K, T,T1, T2}
    result = zero(promote_type(T1,promote_type(T,T2)))
    for (key,val) in pairs(v)
        result += conj(x[key]) * diagonal_element(LO, key) * val
        for (add,elem) in offdiagonals(LO, key)
            result += conj(x[add]) * elem * val
        end
    end
    return result
end

LinearAlgebra.adjoint(op::AbstractHamiltonian) = adjoint(LOStructure(op), op)

"""
    adjoint(::LOStructure, op::AbstractHamiltonian)

Represent the adjoint of an `AbstractHamiltonian`. Extend this method to define custom
adjoints.
"""
function LinearAlgebra.adjoint(::S, op) where {S<:LOStructure}
    error(
        "`adjoint()` not defined for `AbstractHamiltonian`s with `LOStructure` `$(S)`. ",
        " Is your Hamiltonian hermitian?"
    )
end

LinearAlgebra.adjoint(::HermitianLO, op) = op # adjoint is known

"""
    sm, basis = build_sparse_matrix_from_LO(ham::AbstractHamiltonian, add; nnzs = 0)

Create a sparse matrix `sm` of all reachable matrix elements of a linear operator `ham`
starting from the address `add`. The vector `basis` contains the addresses of basis
configurations.
Providing the number `nnzs` of expected calculated matrix elements may improve performance.
"""
function build_sparse_matrix_from_LO(
    ham::AbstractHamiltonian, fs=starting_address(ham); nnzs = 0
)
    adds = [fs] # list of addresses of length linear dimension of matrix
    I = Int[]         # row indices, length nnz
    J = Int[]         # column indices, length nnz
    V = eltype(ham)[] # values, length nnz
    if nnzs > 0
        sizehint!(I, nnzs)
        sizehint!(J, nnzs)
        sizehint!(V, nnzs)
    end

    i = 0 # 1:dim, column of matrix
    while true # loop over columns of the matrix
        i += 1 # next column
        i > length(adds) && break
        add = adds[i] # new address from list
        # compute and push diagonal matrix element
        melem = diagonal_element(ham, add)
        push!(I, i)
        push!(J, i)
        push!(V, melem)
        for (nadd, melem) in offdiagonals(ham, add) # loop over rows
            j = findnext(a -> a == nadd, adds, 1) # find index of `nadd` in `adds`
            if isnothing(j)
                # new address: increase dimension of matrix by adding a row
                push!(adds, nadd)
                j = length(adds) # row index points to the new element in `adds`
            end
            # new nonzero matrix element
            push!(I, i)
            push!(J, j)
            push!(V, melem)
        end
    end
    # when the index `(i,j)` occurs mutiple times in `I` and `J` the elements are added.
    return sparse(I, J, V), adds
end
