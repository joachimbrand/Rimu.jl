# # Example 4: Exact diagonalisation

# When working with smaller systems, or when multiple eigenvalues of a system are required,
# one can use an exact diagonalisation method. There are a few ways to go about
# this, each with its pros and cons. The purpose of this tutorial is to show off the methods
# as well as provide a few tips regarding them.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/exact-example.jl).
# Run it with `julia exact-example.jl`.

# We start by loading Rimu.

using Rimu

# ## Introduction

# We will look at a bosonic system of 4 particles in 5 sites, formulated in momentum
# space. Let's start by building the Hamiltonian. To create a Fock state where all particles
# have zero momentum, we put all the particles in the mode at the centre of the address.

M = 5
N = 4
add = BoseFS(M, cld(M, 2) => N)
ham = HubbardMom1D(add)

# Before performing exact diagonalisation, it is a good idea to check the dimension of the
# Hamiltonian.

dimension(ham)

# Keep in mind that this is an estimate of the number of Fock states the Hamiltonian can act
# on, not the actual matrix size - the matrix size can sometimes be smaller. It can still be
# used as a guide to decide whether a Hamiltonian is amenable to exact diagonalisation and
# to determine which algorithm would be best suited to diagonalising it.

# ## The BasisSetRep

# As we'll see later, there are two ways to construct the matrices from Hamiltonians
# directly, but they both use [`BasisSetRep`](@ref) under the hood. The
# [`BasisSetRep`](@ref), when called with a Hamiltonian and optionally a starting address,
# constructs the sparse matrix of the system and its basis. The starting address defaults to
# the one that was used to initialize the Hamiltonian. [`BasisSetRep`](@ref) only returns
# the part of the matrix that is accessible from this starting address through non-zero
# offdiagonal elements.

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
# Julia functionality to diagonalise them.

# ### The built-in method

# Let's begin by looking at the
# [`eigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen),
# [`eigvecs`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigvecs),
# and
# [`eigvals`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigvals)
# functions from the LinearAlgebra standard library. They operate on dense matrices and
# return the full spectra, hence they are only useful for small systems or when all
# eigenvalues are required.

using LinearAlgebra

mat = Matrix(ham)
eig = eigen(mat);

# The values can be accessed like so:

eig.values

# The vectors are stored as columns in `eig.vectors`:

eig.vectors

# If you need the full spectrum, but would like to use less memory, consider using the
# in-place
# [`eigen!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen!).

# ### Iterative sparse solvers

# For larger Hamiltonians, it is better to use an iterative solver. There are several
# options. We will look at
# [`eigs`](https://arpack.julialinearalgebra.org/stable/api/#Arpack.eigs-Tuple{Any}) from
# [`Arpack.jl`](https://github.com/JuliaLinearAlgebra/Arpack.jl) and
# [`eigsolve`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve) from
# [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/).

# Let's start with Arpack's
# [`eigs`](https://arpack.julialinearalgebra.org/stable/api/#Arpack.eigs-Tuple{Any}). It is
# important to set the `nev` and `which` keyword arguments. `nev` sets the number of
# eigenpairs to find. `which` should in most cases be set to `:SR`, which will find the
# eigenvalues with the smallest real part.

using Arpack

num_eigvals = 3

sm = sparse(ham)
vals_ar, vecs_ar = eigs(sm; which=:SR, nev=num_eigvals)
vals_ar

# Using KrylovKit's
# [`eigsolve`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve) is
# similar, but the `nev` and `which` are given as positional arguments. Note that KrylovKit
# may sometimes return more than `nev` eigenpairs if it happens to find them.

using KrylovKit

vals_kk, vecs_kk = eigsolve(sm, num_eigvals, :SR)
vals_kk

# Both solvers use variants of the [Lanczos
# algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) for Hermitian matrices and the
# [Arnoldi algorithm](https://en.wikipedia.org/wiki/Arnoldi_iteration) for non-Hermitian
# ones. These may in some cases miss degenerate eigenpairs.

# If diagonalisation takes too long, you can reduce the tolerance by setting the `tol`
# keyword argument to `eigs` or `eigsolve`. Using drastically lower tolerances than the
# default can still produce good results in practice. This, however, should be checked on a
# case-by-case basis.

# ### The matrix-free method

# KrylovKit's
# [`eigsolve`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve)
# function is implemented in a way that does not require the linear operator and vector to
# be Julia arrays. Rimu leverages this functionality, which allows diagonalising
# Hamiltonians without ever needing to construct the matrix - all matrix elements are
# generated on the fly.

# While this method is by far the slowest of the ones discussed, it also uses drastically
# less memory. This allows us to diagonalise much larger Hamiltonians.

# To use this method, you first need a starting [`DVec`](@ref):

dvec = DVec(add => 1.0)

# Then, pass that vector and the Hamiltonian to
# [`eigsolve`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve). Since
# the function has no way of knowing the Hamiltonian is Hermitian, we have to provide that
# information through the `issymmetric` or `ishermitian` keyword arguments. Make sure to
# only pass this argument when the Hamiltonian is actually symmetric. To check that,
# look at the `LOStructure` trait:

LOStructure(ham)

#

vals_mf, vecs_mf = eigsolve(ham, dvec, num_eigvals, :SR; issymmetric=true)
vals_mf

# Keep in mind that if an eigenvector is orthogonal to `dvec`, KrylovKit will miss
# it. Consider the following example:

eigsolve(ham, vecs_mf[2], num_eigvals, :SR, issymmetric=true)[1]

# ## Reducing matrix size with symmetries

# As these matrices tend to get large quickly, memory is usually the bottleneck.  There are
# currently two methods implemented to reduce the matrix size, [`ParitySymmetry`](@ref) and
# [`TimeReversalSymmetry`](@ref). These symmetries work by performing a unitary transformation
# on the Hamiltonian which causes it to become block-diagonal. When building a matrix from a
# block-diagonal Hamiltonian, only the block that contains the starting address is
# constructed.

# You should only use these where the relevant symmetries actually apply - no checks are
# performed to make sure they do. There is also currently no way of using both at the same
# time. Please consult the documentation for a more in-depth description of these options.

# The Hamiltonian presented in this example is compatible with the
# [`ParitySymmetry`](@ref). Let's see how the matrix size is reduced when applying it.

size(sparse(ham))

#

size(sparse(ParitySymmetry(ham)))

# In this small example, the size reduction is modest, but for larger systems, you can
# expect to reduce the dimension of the matrix by about half.

all_eigs = eigvals(Matrix(ham))
even_eigs = eigvals(Matrix(ParitySymmetry(ham)))

# The eigenvalues of the transformed Hamiltonian are a subset of the full spectrum. To get
# the other half, we can pass the `even=false` keyword argument to it. When doing that, we
# need to make sure the starting address of the Hamiltonian is not symmetric under reversal:

add_odd = BoseFS(M, cld(M, 2) => N - 3, cld(M, 2) - 1 => 2, cld(M, 2) + 2 => 1)

#

odd_eigs = eigvals(Matrix(ParitySymmetry(HubbardMom1D(add_odd); even=false)))

# Now, let's check that combining the two sets of eigenvalues indeed recovers the whole
# spectrum.

sort([even_eigs; odd_eigs]) ≈ all_eigs

# ## Computing observables

# Since building a matrix from an operator only builds the part that is reachable from the
# starting address, we need to use a different approach when computing observables.

# To demonstrate this, we will use the [`DensityMatrixDiagonal`](@ref) operator, which in
# this case will give the momentum density.

# The idea here is to construct a [`DVec`](@ref) from the computed eigenvector and use it
# directly with the operator.

dvec = DVec(zip(bsr.basis, eigvecs(Matrix(ham))[:, 1]))

# The eigenvectors these methods produce are normalized, hence we can use the three-argument
# `dot` to compute the values of observables. Here we are computing the single particle 
# momentum density distribution, which is just the diagonal of the single-particle density
# matrix in momentum space:

[dot(dvec, DensityMatrixDiagonal(i), dvec) for i in 1:M]

#

using Test #hide
@test length(vals_mf) == length(vals_mf) ≥ num_eigvals   #hide
@test vals_ar[1:num_eigvals] ≈ vals_kk[1:num_eigvals]    #hide
@test vals_kk[1:num_eigvals] ≈ vals_mf[1:num_eigvals]    #hide
@test vals_ar[1:num_eigvals] ≈ eig.values[1:num_eigvals] #hide
nothing #hide
