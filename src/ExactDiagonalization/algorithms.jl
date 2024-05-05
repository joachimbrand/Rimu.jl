abstract type AbstractAlgorithm{MatrixFree} end
ismatrixfree(::AbstractAlgorithm{MatrixFree}) where {MatrixFree} = MatrixFree

"""
    KrylovKitSolver(matrix_free::Bool; kwargs...)
    KrylovKitSolver(; matrix_free = false, kwargs...)

Algorithm for solving a large [`ExactDiagonalizationProblem`](@ref) to find a few
eigenvalues and vectors using the KrylovKit.jl package.
The Lanczos method is used for hermitian matrices, and the Arnoldi method is used for
non-hermitian matrices.

# Arguments
- `matrix_free = false`: Whether to use a matrix-free algorithm. If `false`, a sparse matrix
    will be instantiated. This is typically faster and recommended for small matrices,
    but requires more memory. If `true`, the matrix is not instantiated, which is useful for
    large matrices that would not fit into memory. The calculation will parallelise using
    threading and MPI if available by making use of [`PDVec`](@ref).
- `kwargs`: Additional keyword arguments are passed on to the function
    [`KrylovKit.eigsolve()`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).

!!! note
    Requires the KrylovKit.jl package to be loaded with `using KrylovKit`.
"""
struct KrylovKitSolver{MatrixFree} <: AbstractAlgorithm{MatrixFree}
    kw_nt::NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function KrylovKitSolver{MF}(; kwargs...) where MF
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitSolver requires that KrylovKit is loaded, i.e. `using KrylovKit`")
        else
            kw_nt = NamedTuple(kwargs)
            return new{MF}(kw_nt)
        end
    end
end
KrylovKitSolver(matrix_free::Bool; kwargs...) = KrylovKitSolver{matrix_free}(; kwargs...)
KrylovKitSolver(; matrix_free=true, kwargs...) = KrylovKitSolver(matrix_free; kwargs...)

function Base.show(io::IO, s::KrylovKitSolver)
    nt = (; matrix_free=ismatrixfree(s), s.kw_nt...)
    io = IOContext(io, :compact => true)
    print(io, "KrylovKitSolver")
    show(io, nt)
end


"""
    ArpackSolver(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a sparse
matrix. It uses the Lanzcos method for hermitian problems, and the Arnoldi method for
non-hermitian problems, using the Arpack Fortran library. This is faster than
[`KrylovKitSolver(; matrix_free=true)`](@ref), but it requires more memory and will only be
useful if the matrix fits into memory.

The `kwargs` are passed on to the function
[`Arpack.eigs()`](https://arpack.julialinearalgebra.org/stable/eigs/).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the Arpack.jl package to be loaded with `using Arpack`.
"""
struct ArpackSolver <: AbstractAlgorithm{false}
    kw_nt::NamedTuple
    # the inner constructor checks if Arpack is loaded
    function ArpackSolver(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :ArpackExt)
        if ext === nothing
            error("ArpackSolver() requires that Arpack.jl is loaded, i.e. `using Arpack`")
        else
            kw_nt = NamedTuple(kwargs)
            return new(kw_nt)
        end
    end
end
function Base.show(io::IO, s::ArpackSolver)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "ArpackSolver()")
    else
        print(io, "ArpackSolver")
        show(io, s.kw_nt)
    end
end

"""
    LOBPCGSolver(; kwargs...)

The Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).
Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a
sparse matrix.

LOBPCG is not suitable for non-hermitian eigenvalue problems.

The `kwargs` are passed on to the function
[`IterativeSolvers.lobpcg()`](https://iterativesolvers.julialinearalgebra.org/dev/eigenproblems/lobpcg/).

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
!!! note
    Requires the IterativeSolvers.jl package to be loaded with `using IterativeSolvers`.
"""
struct LOBPCGSolver <: AbstractAlgorithm{false}
    kw_nt::NamedTuple
    # the inner constructor checks if LinearSolvers is loaded
    function LOBPCGSolver(; kwargs...)
        ext = Base.get_extension(@__MODULE__, :IterativeSolversExt)
        if ext === nothing
            error("LOBPCGSolver() requires that IterativeSolvers.jl is loaded, i.e. `using IterativeSolvers`")
        else
            kw_nt = NamedTuple(kwargs)
            return new(kw_nt)
        end
    end
end
function Base.show(io::IO, s::LOBPCGSolver)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "LOBPCGSolver()")
    else
        print(io, "LOBPCGSolver")
        show(io, s.kw_nt)
    end
end


"""
    LinearAlgebraSolver(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) using the dense-matrix
eigensolver from the `LinearAlgebra` standard library. This is only suitable for small
matrices.

The `kwargs` are passed on to function [`LinearAlgebra.eigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen).

# Keyword arguments
- `permute = true`: Whether to permute the matrix before diagonalization.
- `scale = true`: Whether to scale the matrix before diagonalization.
- `sortby`: The sorting order for the eigenvalues.

See also [`ExactDiagonalizationProblem`](@ref), [`solve`](@ref).
"""
struct LinearAlgebraSolver <: AbstractAlgorithm{false}
    kw_nt::NamedTuple
end
LinearAlgebraSolver(; kwargs...) = LinearAlgebraSolver(NamedTuple(kwargs))

function Base.show(io::IO, s::LinearAlgebraSolver)
    io = IOContext(io, :compact => true)
    if isempty(s.kw_nt)
        print(io, "LinearAlgebraSolver()")
    else
        print(io, "LinearAlgebraSolver")
        show(io, s.kw_nt)
    end
end
