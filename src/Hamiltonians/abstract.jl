"""
    check_address_type(h::AbstractHamiltonian, addr)
Throw an `ArgumentError` if the type of `addr` is not compatible with `h`.
"""
function check_address_type(h::AbstractHamiltonian, addr::A) where A
    typeof(starting_address(h)) == A || throw(ArgumentError("address type mismatch"))
end

(h::AbstractHamiltonian)(v) = h * v
(h::AbstractHamiltonian)(w, v) = mul!(w, h, v)

BitStringAddresses.num_modes(h::AbstractHamiltonian) = num_modes(starting_address(h))

"""
    logbinomialapprox(n, k)

Approximate formula for log of binomial coefficient. [Source](https://en.wikipedia.org/wiki/Binomial_coefficient#Bounds_and_asymptotic_formulas)
"""
logbinomialapprox(n,k) = (n+0.5)*log((n+0.5)/(n-k+0.5))+k*log((n-k+0.5)/k) - 0.5*log(2π*k)

"""
    dimension(::Type{T}, h)

Return the dimension of Hilbert space as `T`. If the result does not fit into `T`, return
`nothing`. If `T<:AbstractFloat`, an approximate value computed with the improved
Stirling formula may be returned instead.

# Examples

```jldoctest
julia> dimension(HubbardMom1D(BoseFS((1,2,3))))
28
julia> dimension(HubbardMom1D(near_uniform(BoseFS{200,100})))


julia> dimension(Float64, HubbardMom1D(near_uniform(BoseFS{200,100})))
1.3862737677578234e81
julia> dimension(BigInt, HubbardMom1D(near_uniform(BoseFS{200,100})))
1386083821086188248261127842108801860093488668581216236221011219101585442774669540
```
"""
function dimension(::Type{T}, ::BoseFS{N,M}) where {N,M,T<:Integer}
    return try_binomial(T(N + M - 1), T(N))
end
function dimension(::Type{T}, ::BoseFS{N,M}) where {N,M,T<:AbstractFloat}
    return approximate_binomial(T(N + M - 1), T(N))
end
function dimension(::Type{T}, f::FermiFS{N,M}) where {N,M,T<:Integer}
    return try_binomial(T(M), T(N))
end
function dimension(::Type{T}, f::FermiFS{N,M}) where {N,M,T<:AbstractFloat}
    return approximate_binomial(T(M), T(N))
end
function dimension(::Type{T}, b::BoseFS2C) where {T}
    return dimension(T, b.bsa) * dimension(T, b.bsb)
end
function dimension(::Type{T}, c::CompositeFS) where {T}
    return prod(x -> dimension(T, x), c.components)
end

function try_binomial(n::T, k::T) where {T}
    try
        return T(binomial(n, k))
    catch
        return nothing
    end
end
function approximate_binomial(n::T, k::T) where {T}
    try
        T(binomial(Int128(n), Int128(k)))
    catch
        T(exp(logbinomialapprox(n, k)))
    end
end

dimension(h::AbstractHamiltonian) = dimension(Int, h)
dimension(::Type{T}, h::AbstractHamiltonian) where {T} = dimension(T, starting_address(h))

BitStringAddresses.near_uniform(h::AbstractHamiltonian) = near_uniform(typeof(starting_address(h)))

"""
    rayleigh_quotient(H, v)

```math
\\frac{⟨ v | H | v ⟩}{⟨ v|v ⟩}
```
"""
rayleigh_quotient(lo, v) = dot(v, lo, v)/norm(v)^2

"""
    TwoComponentHamiltonian{T} <: AbstractHamiltonian{T}

Abstract type for representing interacting two-component Hamiltonians in a Fock space with
two different species. At least the following fields should be present:

* `ha` Hamiltonian for species A
* `hb` Hamiltonian for species B

See [`AbstractHamiltonian`](@ref) for a list of methods that need to be defined.

Provides and implementation of [`dimension`](@ref).
"""
abstract type TwoComponentHamiltonian{T} <: AbstractHamiltonian{T} end

function dimension(::Type{T}, h::TwoComponentHamiltonian) where {T}
    return dimension(T, h.ha) * dimension(T, h.hb)
end

"""
    momentum(ham::AbstractHamiltonian)

Momentum as a linear operator in Fock space. Pass a Hamiltonian `ham` in order to convey
information about the Fock basis.

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
"""
momentum

"""
    sm, basis = build_sparse_matrix_from_LO(ham::AbstractHamiltonian, add; nnzs = 0)

Create a sparse matrix `sm` of all reachable matrix elements of a linear operator `ham`
starting from the address `add`. The vector `basis` contains the addresses of basis
configurations.
Providing the number `nnzs` of expected calculated matrix elements may improve performance.

See [`BasisSetRep`](@ref).
"""
function build_sparse_matrix_from_LO(
    ham::AbstractHamiltonian, fs=starting_address(ham); nnzs = 0
)
    adds = [fs] # list of addresses of length linear dimension of matrix
    adds_dict = Dict(fs=>1) # dictionary for index lookup
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
            iszero(melem) && continue
            j = get(adds_dict, nadd, nothing) # look up index; much faster than `findnext`
            if isnothing(j)
                # new address: increase dimension of matrix by adding a row
                push!(adds, nadd)
                j = length(adds) # row index points to the new element in `adds`
                adds_dict[nadd] = j
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

"""
    BasisSetRep(h::AbstractHamiltonian, addr=starting_address(h); sizelim=10^4, nnzs = 0)
Eagerly construct the basis set representation of the operator `h` with all addresses
reachable from `addr`. An `ArgumentError` is thrown if `dimension(h) > sizelim` in order
to prevent memory overflow. Set `sizelim = Inf` in order to disable this behaviour.
Providing the number `nnzs` of expected calculated matrix elements may improve performance.

## Fields
* `sm`: sparse matrix representing `h` in the basis `basis`
* `basis`: vector of addresses
* `h`: the Hamiltonian

## Example
```jldoctest
julia> h = HubbardReal1D(BoseFS((1,0,0)));

julia> bsr = BasisSetRep(h)
BasisSetRep(HubbardReal1D(BoseFS{1,3}((1, 0, 0)); u=1.0, t=1.0)) with dimension 3 and 9 stored entries:3×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:
  0.0  -1.0  -1.0
 -1.0   0.0  -1.0
 -1.0  -1.0   0.0
```
```julia-repl
 julia> using LinearAlgebra; eigvals(Matrix(bsr))
 3-element Vector{Float64}:
  -2.0
   1.0
   1.0

julia> ev = eigvecs(Matrix(bsr))[:,1] # ground state eigenvector
3-element Vector{Float64}:
 -0.5773502691896
 -0.5773502691896
 -0.5773502691896

julia> DVec(zip(bsr.basis,ev)) # ground state as DVec
DVec{BoseFS{1, 3, BitString{3, 1, UInt8}},Float64} with 3 entries, style = IsDeterministic{Float64}()
  BoseFS{1,3}((0, 0, 1)) => -0.5773502691896
  BoseFS{1,3}((0, 1, 0)) => -0.5773502691896
  BoseFS{1,3}((1, 0, 0)) => -0.5773502691896
```
Has methods for [`dimension`](@ref), [`sparse`](@ref), [`Matrix`](@ref),
[`starting_address`](@ref).
"""
struct BasisSetRep{A,SM,H}
    sm::SM
    basis::Vector{A}
    h::H
end

function BasisSetRep(
    h::AbstractHamiltonian, addr=starting_address(h);
    sizelim=10^4, nnzs = 0
)
    dimension(Float64, h) < sizelim || throw(ArgumentError("dimension larger than sizelim"))
    check_address_type(h, addr)
    sm, basis = build_sparse_matrix_from_LO(h, addr; nnzs)
    return BasisSetRep(sm, basis, h)
end

function Base.show(io::IO, b::BasisSetRep)
    print(io, "BasisSetRep($(b.h)) with dimension $(dimension(b)) and $(nnz(b.sm)) stored entries:")
    show(io, MIME"text/plain"(), b.sm)
end

starting_address(bsr::BasisSetRep) = bsr.basis[1]

dimension(bsr::BasisSetRep) = dimension(Int, bsr)
dimension(::Type{T}, bsr::BasisSetRep) where {T} = T(length(bsr.basis))


"""
    sparse(h::AbstractHamiltonian, addr=starting_address(h); sizelim=10^4)
    sparse(bsr::BasisSetRep)
Return a sparse matrix representation of `h` or `bsr`. An `ArgumentError` is thrown if
`dimension(h) > sizelim` in order to prevent memory overflow. Set `sizelim = Inf` in order
to disable this behaviour.

See [`BasisSetRep`](@ref).
"""
function SparseArrays.sparse(h::AbstractHamiltonian, args...; kwargs...)
    return sparse(BasisSetRep(h, args...; kwargs...))
end
SparseArrays.sparse(bsr::BasisSetRep) = bsr.sm

"""
    Matrix(h::AbstractHamiltonian, addr=starting_address(h); sizelim=10^4)
    Matrix(bsr::BasisSetRep)
Return a dense matrix representation of `h` or `bsr`. An `ArgumentError` is thrown if
`dimension(h) > sizelim` in order to prevent memory overflow. Set `sizelim = Inf` in order
to disable this behaviour.

See [`BasisSetRep`](@ref).
"""
function Base.Matrix(h::AbstractHamiltonian, args...; kwargs...)
    return Matrix(BasisSetRep(h, args...; kwargs...))
end
Base.Matrix(bsr::BasisSetRep) = Matrix(bsr.sm)
