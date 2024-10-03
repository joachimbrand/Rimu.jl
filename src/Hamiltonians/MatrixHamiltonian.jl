"""
    MatrixHamiltonian(
        mat::AbstractMatrix{T};
        starting_address::Int = starting_address(mat)
    ) <: AbstractHamiltonian{T}
Wrap an abstract matrix `mat` as an [`AbstractHamiltonian`](@ref) object.
Works with stochastic methods of [`ProjectorMonteCarloProblem()`](@ref) and [`DVec`](@ref).
Optionally, a valid index can be provided as the [`starting_address`](@ref).

Specialised methods are implemented for sparse matrices of type `AbstractSparseMatrixCSC`.
One based indexing is required for the matrix `mat`.
"""
struct MatrixHamiltonian{T,AM,H} <: AbstractHamiltonian{T}
    m::AM
    starting_index::Int
end

function MatrixHamiltonian(
    m::AM;
    starting_address=starting_address(m)
) where AM <:AbstractMatrix
    Base.require_one_based_indexing(m)
    s = size(m)
    s[1] == s[2] || throw(ArgumentError("Matrix needs to be square, got $s."))
    i = Int(starting_address)
    1 ≤ i ≤ s[1] || throw(ArgumentError("Invalid `starting_address` $starting_address."))
    MatrixHamiltonian{eltype(m), AM, ishermitian(m)}(m, i)
end

LOStructure(::Type{<:MatrixHamiltonian{<:Any,<:Any,true}}) = IsHermitian()
LOStructure(::Type{<:MatrixHamiltonian}) = AdjointKnown()
LinearAlgebra.adjoint(mh::MatrixHamiltonian) = MatrixHamiltonian(mh.m')
LinearAlgebra.adjoint(mh::MatrixHamiltonian{<:Any,<:Any,true}) = mh
function Base.:(==)(a::MatrixHamiltonian, b::MatrixHamiltonian)
    return a.m == b.m && a.starting_index == b.starting_index
end

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
    drow = isnothing(drow) ? 0 : drow
    return SparseMatrixOffdiagonals{Int,T,typeof(rows),typeof(vals)}(rows, vals, drow, col)
end

struct SparseMatrixOffdiagonals{A,T,R,V} <: AbstractOffdiagonals{A,T}
    rows::R # indices of rows with nonzero values
    vals::V # nonzero values
    drow::Int # index of col in rows; rows[drow] == col should be true
    col::Int # colum of the matrix
end
function Base.getindex(smo::SparseMatrixOffdiagonals, chosen)
    ind = ifelse(chosen < smo.drow, chosen, chosen + !iszero(smo.drow))
    return smo.rows[ind], smo.vals[ind]
end
Base.size(smo::SparseMatrixOffdiagonals) = (length(smo.rows) - !iszero(smo.drow),)

function get_offdiagonal(
    mh::MatrixHamiltonian{<:Any, <:SparseArrays.AbstractSparseMatrixCSC},
    add,
    chosen
)
    return offdiagonals(mh, add)[chosen]
end
