"""
    check_address_type(h::AbstractHamiltonian, addr_or_type)
Throw an `ArgumentError` if `addr_or_type` is not compatible with `h`. Acceptable arguments
are either an address or an address type, or a tuple or array thereof.

See also [`allowed_address_type`](@ref).
"""
@inline function check_address_type(h, ::Type{A}) where {A}
    B = allowed_address_type(h)
    A <: B || throw(ArgumentError("address type mismatch: found $A, expected <: $B"))
end
@inline check_address_type(h, addr) = check_address_type(h, typeof(addr))
@inline function check_address_type(h::AbstractHamiltonian, v::Union{AbstractArray,Tuple})
    all(check_address_type(h, a) for a in v)
end

(h::AbstractHamiltonian)(v) = h * v
(h::AbstractHamiltonian)(w, v) = mul!(w, h, v)

BitStringAddresses.num_modes(h::AbstractHamiltonian) = num_modes(starting_address(h))

"""
    dimension(h::AbstractHamiltonian, addr=starting_address(h))
    dimension(addr::AbstractFockAddress)
    dimension(::Type{<:AbstractFockAddress})

Return the estimated dimension of Hilbert space. May return a `BigInt` number.

When called on an address or address type, the dimension of the linear space spanned by the
address type is returned. When called on an `AbstractHamiltonian`, an upper bound on the
dimension of the matrix representing the Hamiltonian is returned.

# Examples

```jldoctest
julia> dimension(OccupationNumberFS(1,2,3))
16777216

julia> dimension(HubbardReal1D(OccupationNumberFS(1,2,3)))
28

julia> dimension(BoseFS{200,100})
1386083821086188248261127842108801860093488668581216236221011219101585442774669540

julia> Float64(ans)
1.3860838210861882e81
```

Part of the [`AbstractHamiltonian`](@ref) interface. See also [`BasisSetRep`](@ref).
# Extended Help

The default fallback for `dimension` called on an [`AbstractHamiltonian`](@ref) is to return
the dimension of the address space, which provides an upper bound. For new Hamiltonians a
tighter bound can be provided by defining a custom method.

When extending [`AbstractHamiltonian`](@ref), define a method for the two-argument form
`dimension(h::MyNewHamiltonian, addr)`. For number-conserving Hamiltonians, the function
[`Hamiltonians.number_conserving_dimension`](@ref) may be useful.

When extending [`AbstractFockAddress`](@ref), define a method for
`dimension(::Type{MyNewFockAddress})`.
"""
dimension(h::AbstractHamiltonian) = dimension(h, starting_address(h))
dimension(::AbstractHamiltonian, addr) = dimension(addr)
dimension(addr::AbstractFockAddress) = dimension(typeof(addr))
dimension(::T) where {T<:Number} = typemax(T) # e.g. integer addresses

function dimension(::Type{<:BoseFS{N,M}}) where {N,M}
    return number_conserving_bose_dimension(N,M)
end
function dimension(::Type{<:OccupationNumberFS{M,T}}) where {M,T}
    n = typemax(T)
    return BigInt(n + 1)^BigInt(M)
end
function dimension(::Type{<:FermiFS{N,M}}) where {N,M}
    return number_conserving_fermi_dimension(N, M)
end
function dimension(::Type{<:BoseFS2C{NA,NB,M}}) where {NA,NB,M}
    return dimension(BoseFS{NA,M}) * dimension(BoseFS{NB,M})
end
function dimension(::Type{<:CompositeFS{<:Any,<:Any,<:Any,T}}) where {T}
    return prod(dimension, T.parameters)
    # This relies on an implementation detail of the Tuple type and may break in future
    # julia versions.
end

# for backward compatibility
function dimension(::Type{T}, h, addr=starting_address(h)) where {T}
    return T(dimension(h, addr))
end
dimension(::Type{T}, addr::AbstractFockAddress) where {T} = T(dimension(addr))

Base.isreal(h::AbstractHamiltonian) = eltype(h) <: Real
LinearAlgebra.isdiag(h::AbstractHamiltonian) = LOStructure(h) ≡ IsDiagonal()
LinearAlgebra.ishermitian(h::AbstractHamiltonian) = LOStructure(h) ≡ IsHermitian()
LinearAlgebra.issymmetric(h::AbstractHamiltonian) = ishermitian(h) && isreal(h)

BitStringAddresses.near_uniform(h::AbstractHamiltonian) = near_uniform(typeof(starting_address(h)))

"""
    number_conserving_bose_dimension(n, m)

Return the dimension of the number-conserving Fock space for `n` bosons in `m` modes:
`binomial(n + m - 1, n)`.

See also [`number_conserving_fermi_dimension`](@ref), [`number_conserving_dimension`](@ref).
"""
number_conserving_bose_dimension(n, m) = binomial(BigInt(n + m - 1), BigInt(n))
"""
    number_conserving_fermi_dimension(n, m)

Return the dimension of the number-conserving Fock space for `n` fermions in `m` modes:
`binomial(m, n)`.

See also [`number_conserving_bose_dimension`](@ref), [`number_conserving_dimension`](@ref).
"""
number_conserving_fermi_dimension(n, m) = binomial(BigInt(m), BigInt(n))

"""
    number_conserving_dimension(address <: AbstractFockAddress)

Return the dimension of the Fock space spanned by the address type assuming particle number
conservation.

See also [`number_conserving_bose_dimension`](@ref),
[`number_conserving_fermi_dimension`](@ref), [`dimension`](@ref).
"""
function number_conserving_dimension(address::Union{BoseFS,OccupationNumberFS})
    m = num_modes(address)
    n = num_particles(address)
    return number_conserving_bose_dimension(n, m)
end
function number_conserving_dimension(address::FermiFS)
    m = num_modes(address)
    n = num_particles(address)
    return number_conserving_fermi_dimension(n, m)
end
function number_conserving_dimension(addr::BoseFS2C)
    return number_conserving_dimension(addr.bsa) * number_conserving_dimension(addr.bsb)
end
function number_conserving_dimension(address::CompositeFS)
    return prod(number_conserving_dimension, address.components)
end

"""
    rayleigh_quotient(H, v)

Return the Rayleigh quotient of the linear operator `H` and the vector `v`:

```math
\\frac{⟨ v | H | v ⟩}{⟨ v|v ⟩}
```
"""
rayleigh_quotient(lo, v) = dot(v, lo, v) / norm(v)^2

"""
    TwoComponentHamiltonian{T} <: AbstractHamiltonian{T}

Abstract type for representing interacting two-component Hamiltonians in a Fock space with
two different species. At least the following fields should be present:

* `ha` Hamiltonian for species A
* `hb` Hamiltonian for species B

See [`AbstractHamiltonian`](@ref) for a list of methods that need to be defined.

Provides an implementation of [`dimension`](@ref).
"""
abstract type TwoComponentHamiltonian{T} <: AbstractHamiltonian{T} end

function dimension(h::TwoComponentHamiltonian)
    return dimension(h.ha) * dimension(h.hb)
end

"""
    momentum(ham::AbstractHamiltonian)

Momentum as a linear operator in Fock space. Pass a Hamiltonian `ham` in order to convey
information about the Fock basis. Returns an [`AbstractHamiltonian`](@ref) that represents
the momentum operator.

Note: `momentum` is currently only defined on [`HubbardMom1D`](@ref).

# Example

```jldoctest
julia> add = BoseFS((1, 0, 2, 1, 2, 1, 1, 3));


julia> ham = HubbardMom1D(add; u = 2.0, t = 1.0);


julia> mom = momentum(ham);


julia> diagonal_element(mom, add) # calculate the momentum of a single configuration
-1.5707963267948966

julia> v = DVec(add => 10; capacity=1000);


julia> rayleigh_quotient(mom, v) # momentum expectation value for state vector `v`
-1.5707963267948966
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
momentum

"""
    build_sparse_matrix_from_LO(
        ham, address=starting_address(ham);
        cutoff, filter=nothing, nnzs, col_hint, sort=false, kwargs...
    ) -> sm, basis
    build_sparse_matrix_from_LO(ham, addresses::AbstractVector; kwargs...)

Create a sparse matrix `sm` of all reachable matrix elements of a linear operator `ham`
starting from `address`. Instead of a single address, a vector of `addresses` can be passed.
The vector `basis` contains the addresses of basis configurations.

Providing the number `nnzs` of expected calculated matrix elements and `col_hint` for the
estimated number of nonzero off-diagonal matrix elements in each matrix column may improve
performance.

Providing an energy cutoff will skip the columns and rows with diagonal elements greater
than `cutoff`. Alternatively, an arbitrary `filter` function can be used instead. These are
not enabled by default. To generate the matrix truncated to the subspace spanned by the
`addresses`, use `filter = Returns(false)`.

Setting `sort` to `true` will sort the `basis` and order the matrix rows and columns
accordingly. This is useful when the order of the columns matters, e.g. when comparing
matrices. Any additional keyword arguments are passed on to `Base.sortperm`.

See [`BasisSetRep`](@ref).
"""
function build_sparse_matrix_from_LO(
    ham, addr_or_vec=starting_address(ham);
    cutoff=nothing,
    filter=isnothing(cutoff) ? nothing : (a -> diagonal_element(ham, a) ≤ cutoff),
    nnzs=0, col_hint=0, # sizehints are opt-in
    sort=false, kwargs...
)
    # Set up `adds` as queue of addresses. Also returned as the basis.
    adds = addr_or_vec isa Union{AbstractArray,Tuple} ? [addr_or_vec...] : [addr_or_vec]

    T = eltype(ham)
    dict = Dict(add => i for (i, add) in enumerate(adds)) # Map from addresses to indices
    col = Dict{Int,T}()       # Temporary column storage
    sizehint!(col, col_hint)

    is = Int[] # row indices
    js = Int[] # column indice
    vs = T[]   # non-zero values

    sizehint!(is, nnzs)
    sizehint!(js, nnzs)
    sizehint!(vs, nnzs)

    i = 0
    while i < length(adds)
        i += 1
        add = adds[i]
        push!(is, i)
        push!(js, i)
        push!(vs, diagonal_element(ham, add))

        empty!(col)
        for (off, v) in offdiagonals(ham, add)
            iszero(v) && continue
            j = get(dict, off, nothing)
            if isnothing(j)
                # Energy cutoff: remember skipped addresses, but avoid adding them to `adds`
                if !isnothing(filter) && !filter(off)
                    dict[off] = 0
                    j = 0
                else
                    push!(adds, off)
                    j = length(adds)
                    dict[off] = j
                end
            end
            if !iszero(j)
                col[j] = get(col, j, zero(T)) + v
            end
        end
        # Copy the column into the sparse matrix vectors.
        for (j, v) in col
            iszero(v) && continue
            push!(is, i)
            push!(js, j)
            push!(vs, v)
        end
    end

    matrix = sparse(js, is, vs, length(adds), length(adds))
    if sort
        perm = sortperm(adds; kwargs...)
        return permute!(matrix, perm, perm), permute!(adds, perm)
    else
        return matrix, adds
    end
end

"""
    build_basis(
        ham, address=starting_address(ham);
        cutoff, filter, sizelim, sort=false, kwargs...
    ) -> basis
    build_basis(ham, addresses::AbstractVector; kwargs...)

Get all basis element of a linear operator `ham` that are reachable (via
non-zero matrix elements) from the address `address`, returned as a vector.
Instead of a single address, a vector of `addresses` can be passed.
Does not return the matrix, for that purpose use [`BasisSetRep`](@ref).

Providing an energy cutoff will skip addresses with diagonal elements greater
than `cutoff`. Alternatively, an arbitrary `filter` function can be used instead.
Addresses passed as arguments are not filtered.
A maximum basis size `sizelim` can be set which will throw an error if the expected
dimension of `ham` is larger than `sizelim`. This may be useful when memory may be a
concern. These options are disabled by default.

Setting `sort` to `true` will sort the basis. Any additional keyword arguments
are passed on to `Base.sort!`.
"""
function build_basis(
    ham, addr_or_vec=starting_address(ham);
    cutoff=nothing,
    filter=isnothing(cutoff) ? nothing : (a -> diagonal_element(ham, a) ≤ cutoff),
    sort=false,
    max_size=Inf, # retained for backwards compatibility; use sizelim instead
    sizelim=max_size,
    kwargs...
)
    check_address_type(ham, addr_or_vec)
    single_addr = addr_or_vec isa Union{AbstractArray,Tuple} ? addr_or_vec[1] : addr_or_vec
    if dimension(ham, single_addr) > sizelim
        throw(ArgumentError("dimension larger than sizelim"))
    end
    # Set up `adds` as queue of addresses. Also returned as the basis.
    adds = addr_or_vec isa Union{AbstractArray,Tuple} ? [addr_or_vec...] : [addr_or_vec]
    known_basis = Set(adds)     # known addresses

    i = 0
    while i < length(adds)
        i += 1
        add = adds[i]

        for (off, v) in offdiagonals(ham, add)
            (iszero(v) || off in known_basis) && continue   # check if valid
            push!(known_basis, off)
            !isnothing(filter) && !filter(off) && continue  # check filter
            push!(adds, off)
        end
    end

    if sort
        return sort!(adds, kwargs...)
    else
        return adds
    end
end

"""
    BasisSetRep(
        h::AbstractHamiltonian, addr=starting_address(h);
        sizelim=10^6, nnzs, cutoff, filter, sort=false, kwargs...
    )
    BasisSetRep(h::AbstractHamiltonian, addresses::AbstractVector; kwargs...)

Eagerly construct the basis set representation of the operator `h` with all addresses
reachable from `addr`. Instead of a single address, a vector of `addresses` can be passed.

An `ArgumentError` is thrown if `dimension(h) > sizelim` in order to prevent memory
overflow. Set `sizelim = Inf` in order to disable this behaviour.

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

## Fields
* `sm`: sparse matrix representing `h` in the basis `basis`
* `basis`: vector of addresses
* `h`: the Hamiltonian

## Example
```jldoctest
julia> h = HubbardReal1D(BoseFS(1,0,0));

julia> bsr = BasisSetRep(h)
BasisSetRep(HubbardReal1D(BoseFS{1,3}(1, 0, 0); u=1.0, t=1.0)) with dimension 3 and 9 stored entries:3×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:
  0.0  -1.0  -1.0
 -1.0   0.0  -1.0
 -1.0  -1.0   0.0

julia> BasisSetRep(h, bsr.basis[1:2]; filter = Returns(false)) # passing addresses and truncating
BasisSetRep(HubbardReal1D(BoseFS{1,3}(1, 0, 0); u=1.0, t=1.0)) with dimension 2 and 4 stored entries:2×2 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  0.0  -1.0
 -1.0   0.0
```
```julia-repl
julia> using LinearAlgebra; eigvals(Matrix(bsr)) # eigenvalues
3-element Vector{Float64}:
 -1.9999999999999996
  0.9999999999999997
  1.0000000000000002

julia> ev = eigvecs(Matrix(bsr))[:,1] # ground state eigenvector
3-element Vector{Float64}:
 -0.5773502691896257
 -0.5773502691896255
 -0.5773502691896257

julia> DVec(zip(bsr.basis,ev)) # ground state as DVec
DVec{BoseFS{1, 3, BitString{3, 1, UInt8}},Float64} with 3 entries, style = IsDeterministic{Float64}()
  fs"|0 0 1⟩" => -0.57735
  fs"|0 1 0⟩" => -0.57735
  fs"|1 0 0⟩" => -0.57735
```
Has methods for [`dimension`](@ref), [`sparse`](@ref), [`Matrix`](@ref),
[`starting_address`](@ref).

Part of the [`AbstractHamiltonian`](@ref) interface. See also [`build_basis`](@ref).
"""
struct BasisSetRep{A,SM,H}
    sm::SM
    basis::Vector{A}
    h::H
end

function BasisSetRep(h::AbstractHamiltonian, addr_or_vec=starting_address(h); kwargs...)
    # In the default case we pass `AdjointUnknown()` in order to skip the
    # symmetrisation of the sparse matrix
    return _bsr_ensure_symmetry(AdjointUnknown(), h, addr_or_vec; kwargs...)
end
# special cases are needed for symmetry wrappers and are defined there

# default, does not enforce symmetries
function _bsr_ensure_symmetry(
    ::LOStructure, h::AbstractHamiltonian, addr_or_vec;
    sizelim=10^6, test_approx_symmetry=true, kwargs...
)
    single_addr = addr_or_vec isa Union{AbstractArray,Tuple} ? addr_or_vec[1] : addr_or_vec
    if dimension(h, single_addr) > sizelim
        throw(ArgumentError("dimension larger than sizelim"))
    end
    check_address_type(h, addr_or_vec)
    sm, basis = build_sparse_matrix_from_LO(h, addr_or_vec; kwargs...)
    return BasisSetRep(sm, basis, h)
end

# build the BasisSetRep while enforcing hermitian symmetry
function _bsr_ensure_symmetry(
    ::IsHermitian, h::AbstractHamiltonian, addr_or_vec;
    sizelim=10^6, test_approx_symmetry=true, kwargs...
)
    single_addr = addr_or_vec isa Union{AbstractArray,Tuple} ? addr_or_vec[1] : addr_or_vec
    if dimension(h, single_addr) > sizelim
        throw(ArgumentError("dimension larger than sizelim"))
    end
    check_address_type(h, addr_or_vec)
    sm, basis = build_sparse_matrix_from_LO(h, addr_or_vec; kwargs...)
    fix_approx_hermitian!(sm; test_approx_symmetry) # enforce hermitian symmetry after building
    return BasisSetRep(sm, basis, h)
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
            return A
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

function Base.show(io::IO, b::BasisSetRep)
    print(io, "BasisSetRep($(b.h)) with dimension $(dimension(b)) and $(nnz(b.sm)) stored entries:")
    show(io, MIME"text/plain"(), b.sm)
end

starting_address(bsr::BasisSetRep) = bsr.basis[1]

dimension(bsr::BasisSetRep) = length(bsr.basis)

"""
    sparse(h::AbstractHamiltonian, addr=starting_address(h); kwargs...)
    sparse(bsr::BasisSetRep)

Return a sparse matrix representation of `h` or `bsr`. `kwargs` are passed to
[`BasisSetRep`](@ref).

See [`BasisSetRep`](@ref).
"""
function SparseArrays.sparse(h::AbstractHamiltonian, args...; kwargs...)
    return sparse(BasisSetRep(h, args...; kwargs...))
end
SparseArrays.sparse(bsr::BasisSetRep) = bsr.sm

"""
    Matrix(h::AbstractHamiltonian, addr=starting_address(h); sizelim=10^4, kwargs...)
    Matrix(bsr::BasisSetRep)

Return a dense matrix representation of `h` or `bsr`. `kwargs` are passed to
[`BasisSetRep`](@ref).

See [`BasisSetRep`](@ref).
"""
function Base.Matrix(h::AbstractHamiltonian, args...; sizelim=1e4, kwargs...)
    return Matrix(BasisSetRep(h, args...; sizelim, kwargs...))
end
Base.Matrix(bsr::BasisSetRep) = Matrix(bsr.sm)

function Base.getindex(ham::AbstractHamiltonian{T}, address1, address2) where {T}
    # calculate the matrix element when only two bitstring addresses are given
    # this is NOT used for the QMC algorithm and is currently not used either
    # for building the matrix for conventional diagonalisation.
    # Only used for verifying matrix.
    # This will be slow and inefficient. Avoid using for larger Hamiltonians!
    address1 == address2 && return diagonal_element(ham, address1) # diagonal
    res = zero(T)
    for (add, val) in offdiagonals(ham, address2) # off-diag column as iterator
        if add == address1
            res += val # found address1
        end
    end
    return res
end

LinearAlgebra.adjoint(op::AbstractHamiltonian) = adjoint(LOStructure(op), op)

"""
    adjoint(::LOStructure, op::AbstractHamiltonian)

Represent the adjoint of an [`AbstractHamiltonian`](@ref). Extend this method to define
custom adjoints.
"""
function LinearAlgebra.adjoint(::S, op) where {S<:LOStructure}
    error(
        "`adjoint()` not defined for `AbstractHamiltonian`s with `LOStructure` `$(S)`. ",
        " Is your Hamiltonian hermitian?"
    )
end

LinearAlgebra.adjoint(::IsHermitian, op) = op # adjoint is known
LinearAlgebra.adjoint(::IsDiagonal, op) = op

"""
    TransformUndoer{
        T,K<:AbstractHamiltonian,O<:Union{AbstractHamiltonian,Nothing}
    } <: AbstractHamiltonian{T}

Type for creating a new operator for the purpose of calculating overlaps of transformed
vectors, which are defined by some transformation `transform`. The new operator should
represent the effect of undoing the transformation before calculating overlaps, including
with an optional operator `op`.

Not exported; transformations should define all necessary methods and properties,
see [`AbstractHamiltonian`](@ref). An `ArgumentError` is thrown if used with an
unsupported transformation.

# Example

A similarity transform ``\\hat{G} = f \\hat{H} f^{-1}`` has eigenvector
``d = f \\cdot c`` where ``c`` is an eigenvector of ``\\hat{H}``. Then the
overlap ``c' \\cdot c = d' \\cdot f^{-2} \\cdot d`` can be computed by defining all
necessary methods for `TransformUndoer(G)` to represent the operator ``f^{-2}`` and
calculating `dot(d, TransformUndoer(G), d)`.

Observables in the transformed basis can be computed by defining `TransformUndoer(G, A)`
to represent ``f^{-1} A f^{-1}``.

# Supported transformations

* [`GutzwillerSampling`](@ref)
* [`GuidingVectorSampling`](@ref)
"""
struct TransformUndoer{
    T,K<:AbstractHamiltonian,O<:Union{AbstractHamiltonian,Nothing}
} <: AbstractHamiltonian{T}
    transform::K
    op::O
end

function TransformUndoer(k::AbstractHamiltonian, op)
    # default check
    throw(ArgumentError("Unsupported transformation: $k"))
end
TransformUndoer(k::AbstractHamiltonian) = TransformUndoer(k::AbstractHamiltonian, nothing)

# common methods
starting_address(s::TransformUndoer) = starting_address(s.transform)
dimension(s::TransformUndoer, addr) = dimension(s.transform, addr)
function Base.:(==)(a::TransformUndoer, b::TransformUndoer)
    return a.transform == b.transform && a.op == b.op
end
