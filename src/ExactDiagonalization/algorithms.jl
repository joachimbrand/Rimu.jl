"""
    KrylovKitDirect(; kwargs...)

Algorithm for solving a large [`ExactDiagonalizationProblem`](@ref) to find a few
eigenvalues and vectors directly using the KrylovKit.jl package. The problem is solved
without instantiating a sparse matrix. This is slower than [`KrylovKitMatrix()`](@ref),
but it requires less memory and thus can be useful for large matrices that would not fit
into memory. Will parallelise using threading and MPI if available.

The Lanczos method is used for hermitian matrices, and the Arnoldi method is used for
non-hermitian matrices.
The `kwargs` are passed on to the function
[`KrylovKit.eigsolve()`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the KrylovKit.jl package to be loaded with `using KrylovKit`.
"""
struct KrylovKitDirect{K<:NamedTuple}
    kw_nt::K # NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function KrylovKitDirect(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitDirect requires that KrylovKit is loaded, i.e. `using KrylovKit`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::KrylovKitDirect)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "KrylovKitDirect()")
    else
        print(io, "KrylovKitDirect")
        show(io, s.kw_nt)
    end
end

"""
    KrylovKitMatrix(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) using the KrylovKit.jl
package after instantiating a sparse matrix. This is faster than
[`KrylovKitDirect()`](@ref), but it requires more memory and will only be useful if the
matrix fits into memory.

The Lanczos method is used for hermitian matrices, and the Arnoldi method is used for
non-hermitian matrices.
The `kwargs` are passed on to the function
[`KrylovKit.eigsolve()`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the KrylovKit.jl package to be loaded with `using KrylovKit`.
"""
struct KrylovKitMatrix{K<:NamedTuple}
    kw_nt::K # NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function KrylovKitMatrix(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitMatrix() requires that KrylovKit.jl is loaded, i.e. `using KrylovKit`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::KrylovKitMatrix)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "KrylovKitMatrix()")
    else
        print(io, "KrylovKitMatrix")
        show(io, s.kw_nt)
    end
end

"""
    ArpackEigs(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a sparse
matrix. It uses the Lanzcos method for hermitian problems, and the Arnoldi method for
non-hermitian problems, using the Arpack Fortran library. This is faster than
[`KrylovKitDirect()`](@ref), but it requires more memory and will only be useful if the
matrix fits into memory.

The `kwargs` are passed on to the function
[`Arpack.eigs()`](https://arpack.julialinearalgebra.org/stable/eigs/).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the Arpack.jl package to be loaded with `using Arpack`.
"""
struct ArpackEigs{K<:NamedTuple}
    kw_nt::K # NamedTuple
    # the inner constructor checks if Arpack is loaded
    function ArpackEigs(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :ArpackExt)
        if ext === nothing
            error("ArpackEigs() requires that Arpack.jl is loaded, i.e. `using Arpack`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::ArpackEigs)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "ArpackEigs()")
    else
        print(io, "ArpackEigs")
        show(io, s.kw_nt)
    end
end

"""
    LOBPCG(; kwargs...)

The Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).
Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a
sparse matrix. This is faster than [`KrylovKitDirect()`](@ref), but it requires more memory
and will only be useful if the matrix fits into memory.

LOBPCG is not suitable for non-hermitian eigenvalue problems.

The `kwargs` are passed on to the function
[`IterativeSolvers.lobpcg()`](https://iterativesolvers.julialinearalgebra.org/dev/eigenproblems/lobpcg/).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the IterativeSolvers.jl package to be loaded with `using IterativeSolvers`.
"""
struct LOBPCG{K<:NamedTuple}
    kw_nt::K # NamedTuple
    # the inner constructor checks if LinearSolvers is loaded
    function LOBPCG(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :IterativeSolversExt)
        if ext === nothing
            error("LOBPCG() requires that IterativeSolvers.jl is loaded, i.e. `using IterativeSolvers`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{typeof(kw_nt)}(kw_nt)
        end
    end
end
function Base.show(io::IO, s::LOBPCG)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "LOBPCG()")
    else
        print(io, "LOBPCG")
        show(io, s.kw_nt)
    end
end


"""
    LinearAlgebraEigen(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) using the dense-matrix
eigensolver from the `LinearAlgebra` standard library. This is only suitable for small
matrices.

The `kwargs` are passed on to function [`LinearAlgebra.eigen`](@ref
https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen).

# Keyword arguments
- `permute = true`: Whether to permute the matrix before diagonalization.
- `scale = true`: Whether to scale the matrix before diagonalization.
- `sortby`: The sorting order for the eigenvalues.

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
"""
struct LinearAlgebraEigen{K<:NamedTuple}
    kw_nt::K # NamedTuple
end
LinearAlgebraEigen(; kwargs...) = LinearAlgebraEigen(NamedTuple(kwargs))

function Base.show(io::IO, s::LinearAlgebraEigen)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "LinearAlgebraEigen()")
    else
        print(io, "LinearAlgebraEigen")
        show(io, s.kw_nt)
    end
end
