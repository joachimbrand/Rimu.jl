# # Example 4: Exact diagonalisation

# When working with smaller systems, or when multiple eigenvalues of a system are required,
# it's better to work with an exact diagonalization method. There are a few ways to go about
# this, each with its pros and cons. The purpose of this tutorial is to show off the methods
# as well as provide a few tips regarding them.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/exact-example.jl).
# Run it with `julia exact-example.jl`.

# We start by loading the required modules.

using Rimu
using LinearAlgebra # eigen, eigvals, and eigvecs
using Arpack        # eigs
using KrylovKit     # eigsolve
using LaTeXStrings

# ## Preliminaries

# In this example, we will look at a system of 6 particles in 7 sites, formulated in
# real space. Let's start by building the Hamiltonian. To create a momentum Fock state,
# we put all the particles in the middle of the address, as that corresponds to the zero
# momentum mode.

M = 7
N = 4
add = BoseFS(M, cld(M, 2) => N)
ham = HubbardMom1D(add)

# Before doing exact diagonalisation, it is a good idea to check the dimensionality of the
# Hamiltonian.

dimension(ham)

# Keep in mind that this is an estimate on the number of Fock states the Hamiltonian can act
# on, not the actual matrix size, but it can be used as a guide to decide whether a
# Hamiltonian is amenable to exact diagonalisation and which algorithm would be best suited
# to diagonalising it.

# ## The BasisSetRep

# As we'll see later, there are two ways to construct the matrices from Hamiltonians
# directly, but they both use `BasisSetRep` under the hood. The `BasisSetRep`, when called
# with a Hamiltonian and optionally a starting address, constructs the sparse matrix of the
# system and its basis. The starting address defaults to the one that was used to initialize
# the Hamiltonian. `BasisSetRep` only returns the part of the matrix that is accessible from
# this starting address through non-zero offdiagonal elements.

bsr = BasisSetRep(ham);

# To access the matrix or basis, access the `sm` and `basis` fields, respectively.

bsr.sm

#

bsr.basis

# When the basis is not needed, we can use `Matrix` or `sparse` directly.

Matrix(ham)

#

sparse(ham)

# ## Computing eigenvalues

# Now that we have a way of constructing matrices from Hamiltonians, we can use standard
# Julia functionality to diagonalize them.

# Let's begin by looking at the `eigen` function from the LinearAlgebra standard library.
# It operates on dense matrices and returns the full spectrum, hence it is only useful for
# small systems, or when all eigenvalues are required.

using LinearAlgebra

mat = Matrix(ham)
eig = eigen(mat)

# The values can be accesses like so:

eig.values

# The vectors are stored as columns in `eig.vectors`:

eig.vectors

# If you need the full spectrum, but would like to use less memory, consider using the
# in-place `eigen!`.

# For larger Hamiltonians, it is better to use an iterative solver. Here, there are several
# options. We will look at `eigs` from
# [`Arpack.jl`](https://github.com/JuliaLinearAlgebra/Arpack.jl) and `eigsolve` from
# [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/).

using Arpack
using KrylovKit

num_eigvals = 3

sm = sparse(ham)
vals_ar, vecs_ar = eigs(sm; which=:SR, nev=num_eigvals)
vals_ar

#

vals_kk, vecs_kk = eigsolve(sm, num_eigvals, :SR)
vals_kk

# Both solvers use the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm)
# for Hermitian matrices and the
# [Arnoldi algorithm](https://en.wikipedia.org/wiki/Arnoldi_iteration) for non-Hermitian
# ones. They both take the desired number of eigenpairs to compute and a symbol denoting
# which eigenpairs to look for. In this case, we use `:SR`, which stands for "smallest
# real part".

# If diagonalisation takes too long, you can use the `tol` keyword argument to `eigs` or
# `eigsolve`. Using drastically lower tolerances than default can still produce good results
# in practice. This, however, should be checked on a case-by-case basis.

# For very large Hamiltonians, where the sparse matrix would not fit into memory, you can
# use `eigsolve` matrix-free. While this method is by far the slowest of the ones discussed,
# its advantage is that it doesn't have to store the matrix - only a few copies of full
# vectors are needed.

# To use it, you first need a starting vector:

dvec = DVec(add => 1.0)

# Then, you pass that vector and the Hamiltonian to `eigsolve`. Since `eigsolve` has no way
# of knowing the Hamiltonian is Hermitian, we have to provide that information through the
# `issymmetric` or `ishermitian` keyword arguments. Make sure to only pass this argument
# when the Hamiltonian is actually symmetric. One way to check is to use the `LOStructure`:

LOStructure(ham)

#

vals_mf, vecs_mf = eigsolve(ham, dvec, num_eigvals, :SR; issymmetric=true)
vals_mf

# Keep in mind that if an eigenvector is orthogonal to `dvec`, KrylovKit will miss
# it. Consider the following example:

eigsolve(ham, vecs_mf[2], num_eigvals, :SR, issymmetric=true)[1]

# ## Reducing matrix size with symmetries

# As these matrices tend to get large quickly, memory is usually the bottleneck.
# There are currently two methods implemented to reduce the matrix size, `ParitySymmetry`
# and `TimeReversalSymmetry`. You should only use these where the relevant symmetries
# actually apply - no checks are performed to make sure they do. To demonstrate them, let's
# use a Hamiltonian where both do apply. Please consult the documentation for a more
# in-depth description of these options.

# The Hamiltonian presented in this example is compatible with the `ParitySymmetry`. Let's
# see how the matrix size is reduced when applying it.

size(sparse(ham))

#

size(sparse(ParitySymmetry(ham)))

# Its eigenvalues are a subset of the full [...]. To get the other half, we can pass the
# `even` keyword argument to it. When doing that, we need to make sure the starting address
# of the Hamiltonian is not even.

all_eigs = eigvals(Matrix(ham))

even_eigs = eigvals(Matrix(ParitySymmetry(ham)))

add_odd = BoseFS(M, cld(M, 2) => N - 3, cld(M, 2) - 1 => 2, cld(M, 2) + 2 => 1)
odd_eigs = eigvals(Matrix(ParitySymmetry(HubbardMom1D(add_odd); even=false)))

sort([even_eigs; odd_eigs]) ≈ all_eigs

# ## Computing observables

# Since operators in Rimu operate on `DVec`s, we need a way to convert the vectors
# coming from `eigen` or `eigs` to them. We can use `BasisSetRep.basis` to do that. As an
# example, let's look at the momentum density.

dvec = DVec(zip(bsr.basis, eigvecs(Matrix(ham))[:, 1]))

#

[dot(dvec, DensityMatrixDiagonal(i), dvec) for i in 1:7]
