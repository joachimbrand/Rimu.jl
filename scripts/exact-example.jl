# # Example 4: Exact diagonalisation

# When working with smaller systems, or when multiple eigenvalues of a system are required,
# it's better to work with an exact diagonalization method. There are a few ways to go about
# this, each with their pros and cons. The purpose of this tutorial to show off the methods
# as well as provide afew tips regarding them.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/exact-example.jl).
# Run it with `julia exact-example.jl`.

# We start by loading the required modules.

using Rimu
using LinearAlgebra # eigen, eigvals, and eigvecs
using Arpack        # eigs
using KrylovKit     # eigsolve
using Plots
using LaTeXStrings
gr() # hide
nothing # hide

# ## Preliminaries

# In this example, we will look at a small example, a three-fermion Hamiltonian in one
# dimension, formulated in momentum space. To start, we define some convenience functions.

# In Rimu, momentum Fock states are stored such that the zero momentum state has a particle
# in the middle of the address.

# Below is a function that constructs a fock state with total momentum 1 and ``M`` momentum
# modes.

function init_address(M)
    return FermiFS2C(
        [i == cld(M, 2) || i == cld(M, 2) + 1 ? 1 : 0 for i in 1:M],
        [i == cld(M, 2) ? 1 : 0 for i in 1:M],
    )
end
nothing # hide

# The Hamiltonians in Rimu are dimensionless. We use the following function to convert the
# units to physical units (``\frac{\hbar^2}{mL}``). This function also handles
# renormalization, which can greatly improve convergence with increasing ``M``, as we'll see
# below.

function convert_units(g, M; renormalize=false)
    if renormalize
        g = g / (1 + g / (π^2 * M))
    end
    t = float(M^2/2)
    u = float(t*2/M*g)
    return (; u, t)
end
nothing # hide

# We know the energy of the this system with ``g = -10``, so we save that in a constant.

const reference_energy = -15.151863462651115
nothing # hide

# We will be comparing a lattice model, a lattice-renormalized model and the transcorrelated
# model. Note that the `dispersion=continuum_dispersion` argument passed to
# `HubbardMom1D` is there to get a quadratic dispersion - without it the dispersion of the
# Hamiltonian would be a cosine, which would introduce an offset in the energy. As a first
# step, we can create the three Hamiltonians with 5 modes.

M = 5
u, t = convert_units(-10, M)
u_ren, _ = convert_units(-10, M, renormalize=true)
address = init_address(M)
lattice = HubbardMom1D(address; u, t, dispersion=continuum_dispersion)
trcorr = Transcorrelated1D(address; v=u, t)
renorm = HubbardMom1D(address; u=u_ren, t, dispersion=continuum_dispersion)
nothing # hide

# ## The BasisSetRep

# As we'll see later, there are way to construct the matrices from Hamiltonians directly,
# but they all use `BasisSetRep` under the hood. The `BasisSetRep`, when called with a
# Hamiltonian and optionally a starting address, constructs the sparse matrix of the system
# and its basis. The starting address defaults to the one that was used to initalize the
# Hamiltonian. `BasisSetRep` only returns the part of the matrix that is accessible from
# this starting address through non-zero offdiagonal elements.

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
# will use a different method to diagonalize each Hamiltonian at different values of `M` and
# compare the results.

# For the first, we will use `LinearAlgebra`'s `eigvals` (also see `eigs` and `eigvecs`)
# function. This is only recommended for very small or dense Hamiltonians. It tends
# to be the fastest method, but storing the full matrix requires the most memory.

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
# Hamiltonians, where enough memory is available to build the matrix.

# The transcorrelated Hamiltonian is non-Hermitian, so its eigenvalues will be complex.
# You should make sure their imaginary part is zero. If it's not, try increasing the
# `cutoff` parameter.

trcorr_energies = map(5:2:20) do M
    u, t = convert_units(-10, M)
    address = init_address(M)
    ham = Transcorrelated1D(address; v=u, t)
    real.(eigs(sparse(ham); which=:SR, nev=1)[1][1])
end

# For the last method, we use the matrix-free method from
# [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/). Note that KrylovKit has no way of
# knowing the Hamiltonian in question is symmetric. We mitigate that by passing it the
# `issymmetric=true` keyword argument. It is only recommended to use this method when the
# matrix is too large to store in memory as it tends to be much slower than the others.
# Another good option is to use KrylovKit on a sparse matrix. This can sometimes outperform
# `eigs`.

renorm_energies = map(5:2:20) do M
    u, t = convert_units(-10, M; renormalize=true)
    address = init_address(M)
    ham = HubbardMom1D(address; u, t, dispersion=continuum_dispersion)
    vec = DVec(address => 1.0)
    eigsolve(ham, vec, 1, :SR; issymmetric=true)[1][1]
end

# Now that we have the energies, let's compare how close to the reference energy they get.

p = scatter(
    5:2:20, abs.(lattice_energies .- reference_energy); label="lattice",
    title="Convergence", xscale=:log10, yscale=:log10, legend=:bottomleft,
    xlabel=L"M", ylabel=L"|E_0 - E_{\mathrm{ref}}|",
)
scatter!(p, 5:2:20, abs.(trcorr_energies .- reference_energy); label="transcorrelated")
scatter!(p, 5:2:20, abs.(renorm_energies .- reference_energy); label="renormalized")

# We can see that in this case, both lattice-renormalized and transcorrelated Hamiltonians
# give good results even with modest values of ``M``. Since the values converge as a power
# law, it's also possible to extrapolate the energies with a fitting package like
# [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl). Keep in mind, however, that for
# larger systems with very low ``M``, the convergence in both may not follow a power law.

# ## Speeding up the computations

# As these matrices tend to get large quickly, memory is usually the bottleneck.

# There are currently two methods implemented to reduce the matrix size, `ParitySymmetry`
# and `TimeReversalSymmetry`. You should only use these where the relevant symmetries
# actually apply - no checks are performed to make sure they do. To demonstrate them, let's
# use a Hamiltonian where both do apply. Please consult the documentation for a more
# in-depth description of these options.

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

# We see that the ground state for all four options is the same, up to numerical erros,
# while the dimension of the matrix is reduced by more than half.

# Note that both of these symmetries accept a keyword argument `even` which controls whether
# even or odd symmetry is applied. If the full spectrum of the Hamiltonian is desired,
# combine the eigenvalues obtained from using both options.

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

p = bar(-9:9, density_up; label=L"↑", xlabel=L"p", ylabel=L"n_p", yscale=:log10)
bar!(p, -9:9, density_dn; label=L"↓", yscale=:log10)
