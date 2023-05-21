"""
    MatrixHamiltonian(
        mat::AbstractMatrix{T};
        starting_address::Int = starting_address(mat)
    ) <: AbstractHamiltonian{T}
Wrap an abstract matrix `mat` as an [`AbstractHamiltonian`](@ref) object for use with
regular `Vector`s indexed by integers. Works with stochastic methods of
[`lomc!()`](@ref). Optionally, a [`starting_address`](@ref) can be provided.

Specialised methods are implemented for sparse matrices of type `AbstractSparseMatrixCSC`.
"""
struct MatrixHamiltonian{T,AM,H} <: AbstractHamiltonian{T}
    m::AM
    starting_index::Int
end

function MatrixHamiltonian(
    m::AM;
    starting_address=starting_address(m)
) where AM <:AbstractMatrix
    s = length.(axes(m))
    @assert s[1]==s[2] "Matrix needs to be square, got $s."
    i = Int(starting_address)
    @assert minimum(axes(m, 2)) ≤ i ≤ maximum(axes(m, 2)) "Invalid index $starting_address."
    MatrixHamiltonian{eltype(m), AM, ishermitian(m)}(m, i)
end

LOStructure(::Type{<:MatrixHamiltonian{<:Any,<:Any,true}}) = IsHermitian()
LOStructure(::Type{<:MatrixHamiltonian}) = AdjointKnown()
LinearAlgebra.adjoint(mh::MatrixHamiltonian) = MatrixHamiltonian(mh.m')
LinearAlgebra.adjoint(mh::MatrixHamiltonian{<:Any,<:Any,true}) = mh


starting_address(mh::MatrixHamiltonian) = mh.starting_index

dimension(mh::MatrixHamiltonian, _) = size(mh.m,2)

num_offdiagonals(mh::MatrixHamiltonian, _) = dimension(mh) - 1

function get_offdiagonal(mh::MatrixHamiltonian, add, chosen)
    newadd = ifelse(chosen < add, chosen, chosen+1)
    return newadd, mh.m[newadd, add]
end
diagonal_element(mh::MatrixHamiltonian, i) = mh.m[i,i]

# specialised methods for sparse matrices - avoid spawning via zero value matrix elements
function num_offdiagonals(
    mh::MatrixHamiltonian{<:Any, <:SparseArrays.AbstractSparseMatrixCSC},
    j
)
    nnz(mh.m[:, j]) - 1
end

function offdiagonals(
    mh::MatrixHamiltonian{T, <:SparseArrays.AbstractSparseMatrixCSC},
    col::Integer
) where T
    rows = rowvals(mh.m)[nzrange(mh.m, col)]
    vals = nonzeros(mh.m)[nzrange(mh.m, col)]
    drow = findprev(x->x==col, rows, min(col,length(rows))) # rows[drow]==col should be true
    return SparseMatrixOffdiagonals{Int,T,typeof(rows),typeof(vals)}(rows, vals, drow, col)
end

struct SparseMatrixOffdiagonals{A,T,R,V} <: AbstractOffdiagonals{A,T}
    rows::R # indices of rows with nonzero values
    vals::V # nonzero values
    drow::Int # index of col in rows; rows[drow] == col should be true
    col::Int # colum of the matrix
end
function Base.getindex(smo::SparseMatrixOffdiagonals, chosen)
    ind = ifelse(chosen < smo.drow, chosen, chosen + 1)
    return smo.rows[ind], smo.vals[ind]
end
Base.size(smo::SparseMatrixOffdiagonals) = (length(smo.rows) - 1,)

function get_offdiagonal(
    mh::MatrixHamiltonian{<:Any, <:SparseArrays.AbstractSparseMatrixCSC},
    add,
    chosen
)
    return offdiagonals(mh, add)[chosen]
end
