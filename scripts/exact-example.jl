# # Example 4: Exact diagonalisation

# When working with smaller systems, or when multiple eigenvalues of a system are required,
# it's better to work with exact diagonalization method. There are a few ways to go about
# this, each with their pros and cons.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/exact-example.jl).
# Run it with `julia exact-example.jl`.

# We start by loading the required modules.

using Rimu
using LinearAlgebra
using Arpack
using KrylovKit
using Plots
using LaTeXStrings
gr() # hide
nothing # hide

# Things to mention
# * renormalization
# * real space -> most sparse, mom space -> less sparse, transcorrelated -> least sparse
# * matrices block - smaller than dimension (except if using potential)
# * full matrices are fastest and give full spectra
# * arpack is good when you can construct a sparse matrix - krylovkit is alternative and may
#   be faster
# * krylovkit is the recommended matrix-free method
# * deterministic fciqmc is not worth it

# ## Preliminaries

# In this example, we will look at a small example of three fermions in one dimension. To
# start, we define some convenience functions.

# In Rimu, momentum Fock states is stored such that the zero momentum state has a particle
# in the middle of the address. Consider the following example:

#momentum(FermiFS((1, 0, 0)))
#momentum(FermiFS((0, 1, 0)))
#momentum(FermiFS((0, 0, 1)))

# Below is a function that constructs a fock state with total momentum 1 and `M` momentum
# modes.

function init_address(M)
    return FermiFS2C(
        [i == cld(M, 2) || i == cld(M, 2) + 1 ? 1 : 0 for i in 1:M],
        [i == cld(M, 2) ? 1 : 0 for i in 1:M],
    )
end
nothing # hide

# The Hamiltonians in Rimu are dimensionless. We use the following function to convert the
# units to physical units (Q: physical units?). This function also handles
# renormalization, which can greatly improve convergence, as we'll see below.

function convert_units(g, M; renormalize=false)
    if renormalize
        g = g / (1 + g / (π^2 * M))
    end
    t = float(M^2/2)
    u = float(t*2/M*g)
    return (; u, t)
end
nothing # hide

# We know the energy of the three-fermion system for g = -10, so we save that in a constant.

const reference_energy = -15.151863462651115

# We will be comparing a lattice model, a lattice-renormalized model and the transcorrelated
# model in this example. Note that the `dispersion=continuum_dispersion` argument passed to
# `HubbardMom1D` is there to get a quadratic dispersion - without it the dispersion of the
# Hamiltonian would be a cosine, which would introduce an offset in the energy. As a first
# step, we can create three Hamiltonians.

M = 5
u, t = convert_units(-10, M)
u_ren, _ = convert_units(-10, M, renormalize=true)
address = init_address(M)
lattice = HubbardMom1D(address; u, t, dispersion=continuum_dispersion)
trcorr = Transcorrelated1D(address; v=u, t)
renorm = HubbardMom1D(address; u=u_ren, t, dispersion=continuum_dispersion)
nothing # hide

# ## The BasisSetRep

# In the first few examples, we will
# but they all use `BasisSetRep` under the hood. The `BasisSetRep`, when called with a Hamiltonian and optionally a starting address, constructs the sparse matrix of the system and its basis.

lattice_bsr = BasisSetRep(lattice)

# To access the matrix or basis, access the `sm` and `basis` fields, respectively.

lattice_bsr.sm

#

lattice_bsr.basis

# When the basis is not needed, we can use `Matrix` or `sparse` directly.

Matrix(trcorr)

#

sparse(renorm)

# ## Computing eigenvalues

# Now that we have a way of constructing matrices from Hamiltonians, we can use standard
# Julia functionality to diagonalize them. To make the computation more interesting, we
# will use a different method to diagonalize each Hamiltonian at different values of M and
# compare the results.

# For the first, we will use `LinearAlgebra`'s `eigvals` function. This is only recommended
# for very small or dense Hamiltonians, where this tends to be the fastest method.

lattice_energies = map(5:2:20) do M
    u, t = convert_units(-10, M)
    address = init_address(M)
    ham = HubbardMom1D(address; u, t, dispersion=continuum_dispersion)
    eigvals(Matrix(ham))[1]
end

# For the second, we will use
# [`Arpack.jl`](https://github.com/JuliaLinearAlgebra/Arpack.jl).  Note that the `which=:SR`
# argument tells Arpack to find the eigenvalues with the smallest real part first. `nev`
# sets the number of eigenvalues to find. This is best used with moderately sized
# Hamiltonians.

trcorr_energies = map(5:2:20) do M
    u, t = convert_units(-10, M)
    address = init_address(M)
    ham = Transcorrelated1D(address; v=u, t)
    eigs(sparse(ham); which=:SR, nev=1)[1][1]
end

# For the last method, we use the matrix-free method from
# [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/). Note that KrylovKit has no way of
# knowing the Hamiltonian in question is symmetric. We mitigate that by passing it the
# `issymmetric=true` keyword argument. It is only recommended to use this method when the
# matrix is too large to store in memory.

renorm_energies = map(5:2:20) do M
    u, t = convert_units(-10, M; renormalize=true)
    address = init_address(M)
    ham = HubbardMom1D(address; u, t, dispersion=continuum_dispersion)
    vec = DVec(address => 1.0)
    eigsolve(ham, vec, 1, :SR; issymmetric=true)[1][1]
end

# Now that we have the energies, let's compare how close to the reference energy they get.

p = plot(;
    title="Convergence", xscale=:log10, yscale=:log10, legend=:bottomleft,
    xlabel=L"M", ylabel=L"|E_0 - E_{\mathrm{ref}}|",
)
scatter!(p, 5:2:20, abs.(lattice_energies .- reference_energy); label="lattice")
scatter!(p, 5:2:20, abs.(trcorr_energies .- reference_energy); label="transcorrelated")
scatter!(p, 5:2:20, abs.(renorm_energies .- reference_energy); label="renormalized")

# ## Speeding up the computations

# As these matrices tend to get large quickly, memory is usually the bottleneck.

# There are currently two methods implemented to reduce the matrix size, `ParitySymmetry`
# and `TimeReversalSymmetry`. Keep in mind that you should only use these where the relevant
# symmetries actually apply - no checks are performed to make sure they do. To demonstrate
# them, let's use a Hamiltonian where both of these apply.

address = FermiFS2C((0,0,0,1,1,0,0), (0,0,1,1,0,0,0))
ham = HubbardMom1DEP(address)

mat_nosym = Matrix(ham)
mat_par = Matrix(ParitySymmetry(ham))
mat_tr = Matrix(TimeReversalSymmetry(ham))
mat_both = Matrix(ParitySymmetry(TimeReversalSymmetry(ham)))

println("Matrix size")
println("none:          ", size(mat_nosym))
println("parity:        ", size(mat_par))
println("time reversal: ", size(mat_tr))
println("both:          ", size(mat_both))
println("\nGroundstate energy:")
println("none:          ", eigvals(mat_nosym)[1])
println("parity:        ", eigvals(mat_par)[1])
println("time reversal: ", eigvals(mat_tr)[1])
println("both:          ", eigvals(mat_both)[1])

# Note that both of these symmetry accept a keyword argument `even` which controls whether
# even or odd symmetry is applied. Both options must be used and eigenvalues combined if the
# full spectrum of the Hamiltonian is desired.

# If diagonalisation time is a concern, use the `tol` keyword argument to `eigs` or
# `eigsolve`. Using drastically lower tolerances than default can still produce good results
# in practice. This, however, should be checked on a case-by-case basis.

# ## Computing observables

# Since all operators in Rimu operate on `DVec`s, we need a way to convert the vectors
# coming from `eigen` or `eigs` to them. We can use `BasisSetRep.basis` to do that. As an
# example, let's look at the momentum density.

M = 19
address = init_address(M)
bsr = BasisSetRep(HubbardMom1D(
    address; convert_units(-10, M)..., dispersion=continuum_dispersion
))

eigvec = eigs(sparse(bsr); nev=1, which=:SR)[2]
dvec = DVec(zip(bsr.basis, eigvec))
density_up = [dot(dvec, DensityMatrixDiagonal(i; component=1), dvec) for i in 1:M]
density_dn = [dot(dvec, DensityMatrixDiagonal(i; component=2), dvec) for i in 1:M]

p = plot(; xlabel=L"p", ylabel=L"ρ") # (Q: rho?))
bar!(p, density_up; label=L"↑")
bar!(p, density_dn; label=L"↓")
