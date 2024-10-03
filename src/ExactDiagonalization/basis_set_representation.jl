"""
    BasisSetRepresentation(
        hamiltonian::AbstractHamiltonian, addr=starting_address(hamiltonian);
        sizelim=10^8, cutoff, filter, max_depth, stop_after, sort=false, kwargs...
    )
    BasisSetRepresentation(hamiltonian::AbstractHamiltonian, addresses::AbstractVector; kwargs...)

Eagerly construct the basis set representation of the operator `hamiltonian` with all
addresses reachable from `addr`. Instead of a single address, a vector of `addresses` can be
passed.

An `ArgumentError` is thrown if `dimension(hamiltonian) > sizelim` in order to prevent
memory overflow. Set `sizelim = Inf` in order to disable this behaviour.

Providing the number `nnzs` of expected calculated matrix elements and `col_hint` for the
estimated number of nonzero off-diagonal matrix elements in each matrix column may improve
performance.

Providing an energy cutoff will skip the columns and rows with diagonal elements greater
than `cutoff`. Alternatively, an arbitrary `filter` function can be used instead.
Addresses passed as arguments are not filtered. To generate the matrix truncated to the
subspace spanned by the `addresses`, use `filter = Returns(false)`.

Setting `sort` to `true` will sort the matrix rows and columns. This is useful when the
order of the columns matters, e.g. when comparing matrices. Any additional keyword arguments
are passed on to `Base.sortperm`.

!!! warning
        The order of the returned basis and matrix rows and columns is arbitrary and
        non-deterministic. Use `sort=true` if the ordering matters.

## Fields
* `sparse_matrix`: sparse matrix representing `hamiltonian` in the basis `basis`
* `basis`: vector of addresses
* `hamiltonian`: the Hamiltonian `hamiltonian`

## Example
```jldoctest; filter = r"(\\d*)\\.(\\d{4})\\d+" => s"\\1.\\2***"
julia> hamiltonian = HubbardReal1D(BoseFS(1,0,0));

julia> bsr = BasisSetRepresentation(hamiltonian)
BasisSetRepresentation(HubbardReal1D(fs"|1 0 0⟩"; u=1.0, t=1.0)) with dimension 3 and 6 stored entries:3×3 SparseArrays.SparseMatrixCSC{Float64, Int32} with 6 stored entries:
   ⋅   -1.0  -1.0
 -1.0    ⋅   -1.0
 -1.0  -1.0    ⋅

julia> BasisSetRepresentation(hamiltonian, bsr.basis[1:2]; filter = Returns(false)) # passing addresses and truncating
BasisSetRepresentation(HubbardReal1D(fs"|1 0 0⟩"; u=1.0, t=1.0)) with dimension 2 and 2 stored entries:2×2 SparseArrays.SparseMatrixCSC{Float64, Int32} with 2 stored entries:
   ⋅   -1.0
 -1.0    ⋅

julia> using LinearAlgebra; round.(eigvals(Matrix(bsr)); digits = 4) # eigenvalues
3-element Vector{Float64}:
 -2.0
  1.0
  1.0

julia> ev = eigvecs(Matrix(bsr))[:,1]; ev = ev .* sign(ev[1]) # ground state eigenvector
3-element Vector{Float64}:
 0.5773502691896257
 0.5773502691896255
 0.5773502691896257

julia> dv = DVec(zip(bsr.basis, ev)) # ground state as DVec
DVec{BoseFS{1, 3, BitString{3, 1, UInt8}},Float64} with 3 entries, style = IsDeterministic{Float64}()
  fs"|0 0 1⟩" => 0.57735
  fs"|0 1 0⟩" => 0.57735
  fs"|1 0 0⟩" => 0.57735
```
Has methods for [`dimension`](@ref), [`sparse`](@ref), [`Matrix`](@ref),
[`starting_address`](@ref).

Part of the [`AbstractHamiltonian`](@ref) interface. See also [`build_basis`](@ref).
"""
struct BasisSetRepresentation{A,SM,H}
    sparse_matrix::SM
    basis::Vector{A}
    hamiltonian::H
end

function BasisSetRepresentation(
    hamiltonian::AbstractHamiltonian, addr_or_vec=starting_address(hamiltonian);
    kwargs...
)
    # In the default case we pass `AdjointUnknown()` in order to skip the
    # symmetrisation of the sparse matrix
    return _bsr_ensure_symmetry(AdjointUnknown(), hamiltonian, addr_or_vec; kwargs...)
end
# special cases are needed for symmetry wrappers

function BasisSetRepresentation(
    hamiltonian::ParitySymmetry, addr=starting_address(hamiltonian);
    kwargs...
)
    # For symmetry wrappers it is necessary to explicity symmetrise the matrix to
    # avoid the loss of matrix symmetry due to floating point rounding errors
    return _bsr_ensure_symmetry(LOStructure(hamiltonian), hamiltonian, addr; kwargs...)
end

function BasisSetRepresentation(
    hamiltonian::TimeReversalSymmetry, addr=starting_address(hamiltonian);
    kwargs...
)
    # For symmetry wrappers it is necessary to explicity symmetrise the matrix to
    # avoid the loss of matrix symmetry due to floating point rounding errors
    return _bsr_ensure_symmetry(LOStructure(hamiltonian), hamiltonian, addr; kwargs...)
end

# default, does not enforce symmetries
function _bsr_ensure_symmetry(
    ::LOStructure, hamiltonian::AbstractHamiltonian, addr_or_vec;
    test_approx_symmetry=true, kwargs...
)
    single_addr = addr_or_vec isa Union{AbstractArray,Tuple} ? addr_or_vec[1] : addr_or_vec
    check_address_type(hamiltonian, addr_or_vec)
    sparse_matrix, basis = build_sparse_matrix_from_LO(hamiltonian, addr_or_vec; kwargs...)
    return BasisSetRepresentation(sparse_matrix, basis, hamiltonian)
end

# build the BasisSetRepresentation while enforcing hermitian symmetry
function _bsr_ensure_symmetry(
    ::IsHermitian, hamiltonian::AbstractHamiltonian, addr_or_vec;
    test_approx_symmetry=true, kwargs...
)
    single_addr = addr_or_vec isa Union{AbstractArray,Tuple} ? addr_or_vec[1] : addr_or_vec
    check_address_type(hamiltonian, addr_or_vec)
    sparse_matrix, basis = build_sparse_matrix_from_LO(hamiltonian, addr_or_vec; kwargs...)
    fix_approx_hermitian!(sparse_matrix; test_approx_symmetry) # enforce hermitian symmetry after building
    return BasisSetRepresentation(sparse_matrix, basis, hamiltonian)
end

"""
    fix_approx_hermitian!(A; test_approx_symmetry=true, kwargs...)

Replaces the matrix `A` by `½(A + A')` in place. This will be successful and the result
is guaranteed to pass the `ishermitian` test only if the matrix is square and already
approximately hermitian.

By default logs an error message if the matrix `A` is found to not be approximately
hermitian. Set `test_approx_symmetry=false` to bypass testing.
Other keyword arguments are passed on to `isapprox`.
"""
function fix_approx_hermitian!(A; test_approx_symmetry=true, kwargs...)
    # Generic and inefficient version. Make sure to replace by efficient specialised code.
    if test_approx_symmetry
        passed = isapprox(A, A'; kwargs...)
        if !passed
            throw(ArgumentError("Matrix is not approximately hermitian."))
        end
    end
    @. A = 1 / 2 * (A + A')
    return A
end

# specialised code for sparse matrices
using SparseArrays: AbstractSparseMatrixCSC, getcolptr, rowvals, nonzeros

# special case for sparse matrices; avoids most allocations, testing is free
function fix_approx_hermitian!(A::AbstractSparseMatrixCSC; test_approx_symmetry=false, kwargs...)
    passed = isapprox_enforce_hermitian!(A; kwargs...)
    if test_approx_symmetry && !passed
        throw(ArgumentError("Matrix is not approximately hermitian."))
    end
    return A
end

"""
    isapprox_enforce_hermitian!(A::AbstractSparseMatrixCSC; kwargs...) -> Bool

Checks whether the matrix `A` is approximately hermitian by checking each pair of transposed
matrix elements with `isapprox`. Keyword arguments are passed on to `isapprox`.
Returns boolean `true` is the test is passed and `false` if not.

Furthermore, the matrix `A` is modified to become exactly equal to `½(A + A')` if the test
is passed.
"""
function isapprox_enforce_hermitian!(A::AbstractSparseMatrixCSC; kwargs...)
    # based on `ishermsym()` from `SparseArrays`; relies on `SparseArrays` internals
    # https://github.com/JuliaSparse/SparseArrays.jl/blob/1bae96dc8f9a8ca8b7879eef4cf71e186598e982/src/sparsematrix.jl#L3793
    m, n = size(A)
    if m != n
        return false
    end

    colptr = getcolptr(A)
    rowval = rowvals(A)
    nzval = nonzeros(A)
    tracker = copy(getcolptr(A))
    for col = 1:size(A, 2)
        # `tracker` is updated such that, for symmetric matrices,
        # the loop below starts from an element at or below the
        # diagonal element of column `col`"
        for p = tracker[col]:colptr[col+1]-1
            val = nzval[p]
            row = rowval[p]

            # Ignore stored zeros
            if iszero(val)
                continue
            end

            # If the matrix was symmetric we should have updated
            # the tracker to start at the diagonal or below. Here
            # we are above the diagonal so the matrix can't be symmetric.
            if row < col
                return false
            end

            # Diagonal element
            if row == col
                if isapprox(val, conj(val); kwargs...)
                    nzval[p] = real(val)
                else
                    return false
                end
                # if val != conj(val)
                #     return false
                # end
            else
                offset = tracker[row]

                # If the matrix is unsymmetric, there might not exist
                # a rowval[offset]
                if offset > length(rowval)
                    return false
                end

                row2 = rowval[offset]

                # row2 can be less than col if the tracker didn't
                # get updated due to stored zeros in previous elements.
                # We therefore "catch up" here while making sure that
                # the elements are actually zero.
                while row2 < col
                    if !iszero(nzval[offset])
                        return false
                    end
                    offset += 1
                    row2 = rowval[offset]
                    tracker[row] += 1
                end

                # Non zero A[i,j] exists but A[j,i] does not exist
                if row2 > col
                    return false
                end

                # A[i,j] and A[j,i] exists
                if row2 == col
                    if isapprox(val, conj(nzval[offset]); kwargs...)
                        val = 1 / 2 * (val + conj(nzval[offset]))
                        nzval[p] = val
                        nzval[offset] = conj(val)
                    else
                        return false
                    end
                    # if val != conj(nzval[offset])
                    #     return false
                    # end
                    tracker[row] += 1
                end
            end
        end
    end
    return true
end

function Base.show(io::IO, b::BasisSetRepresentation)
    print(io, "BasisSetRepresentation($(b.hamiltonian)) with dimension $(dimension(b))" *
        " and $(nnz(b.sparse_matrix)) stored entries:")
    show(io, MIME"text/plain"(), b.sparse_matrix)
end

Interfaces.starting_address(bsr::BasisSetRepresentation) = bsr.basis[1]

Hamiltonians.dimension(bsr::BasisSetRepresentation) = length(bsr.basis)

"""
    sparse(hamiltonian::AbstractHamiltonian, addr=starting_address(hamiltonian); kwargs...)
    sparse(bsr::BasisSetRepresentation)

Return a sparse matrix representation of `hamiltonian` or `bsr`. `kwargs` are passed to
[`BasisSetRepresentation`](@ref).

See [`BasisSetRepresentation`](@ref).
"""
function SparseArrays.sparse(hamiltonian::AbstractHamiltonian, args...; kwargs...)
    return sparse(BasisSetRepresentation(hamiltonian, args...; kwargs...))
end
SparseArrays.sparse(bsr::BasisSetRepresentation) = bsr.sparse_matrix

"""
    Matrix(
        hamiltonian::AbstractHamiltonian, addr=starting_address(hamiltonian);
        sizelim=10^4, kwargs...
    )
    Matrix(bsr::BasisSetRepresentation)

Return a dense matrix representation of `hamiltonian` or `bsr`. `kwargs` are passed to
[`BasisSetRepresentation`](@ref).

See [`BasisSetRepresentation`](@ref).
"""
function Base.Matrix(hamiltonian::AbstractHamiltonian, args...; sizelim=1e4, kwargs...)
    return Matrix(BasisSetRepresentation(hamiltonian, args...; sizelim, kwargs...))
end
Base.Matrix(bsr::BasisSetRepresentation) = Matrix(bsr.sparse_matrix)
