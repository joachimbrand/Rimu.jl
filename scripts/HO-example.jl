# # Example 5: Harmonic oscillator

# This is an example calculation of the harmonic oscillator with
# contact interactions that are treated with first-order perturbation
# theory.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/HO-example.jl).
# Run it with `julia HO-example.jl`.

# Firstly, we load all needed modules.
using Rimu
using DataFrames
using LinearAlgebra

# First we define the system size for two particles in a 2D harmonic oscillator
# allowing `M` levels above the ground state in each dimension.
N = 2
D = 2
M = 3
S = ntuple(_ -> M + 1, D)
P = prod(S)

# The tuple `S` defines a grid of harmonic oscillator states in a Cartesian
# basis (`P` is the total number), but the underlying addresses are Fock states.
# Use the utility function [`fock_to_cart`](@ref) to convert a Fock address to 
# human-readable Cartesian quantum numbers for inspection.
addr = BoseFS(P, M + 1 => N)
fock_to_cart(addr, S)

# The output shows that both particles are in single-particle state ``n_x=3, n_y=0``.

# The harmonic oscillator Hamiltonian [`HOCartesianEnergyConserved`](@ref) handles 
# contact interactions with
# first-order perturbation theory, so the matrix representation will block
# according to the non-interacting energy of the basis states. Hence the first task is 
# to find all blocks of basis states with the same energy. The strength of the 
# interaction is not relevant at this point, just that it is non-zero.
# We use a dummy groundstate address to build the Hamiltonian 
H = HOCartesianEnergyConserved(BoseFS(P, 1 => N); S)

# and then a utility function [`get_all_blocks`](@ref) to find all blocks up
# to the maximum single-particle energy, which is `M` levels above the groundstate. 
# (Each level is a jump of ``\\hbar\\omega``.)
# This works by looping over all 
# possible states with `N` particles in Cartesian states defined by `S`.
E0 = N*D/2
block_df = get_all_blocks(H; max_energy = E0 + M);

# This outputs a list of blocks in `H` indexed by the noninteracting energy
# of all states in the block, and a single address that can be used to 
# rebuild the block for further analysis.
addr1 = block_df[7,:addr]
E = block_df[7,:block_E0]
# First, notice that all basis states have the same energy, defined by the block
basis1 = build_basis(H, addr1)
map(b -> Hamiltonians.noninteracting_energy(H, b), basis1)

# There are ``2^{D-1}`` blocks at each energy level, which are different due to 
# parity conservation, which is the only other symmetry in the Cartesian harmonic 
# oscillator. 
addr2 = block_df[4,:addr]
basis2 = build_basis(H, addr2)
basis1 == basis2

# However, since we have defined an isotropic harmonic oscillator, we
# should be able to build simultaneous eigenstates of the angular momentum operator
# ``L_z``, implemented with [`AxialAngularMomentumHO`](@ref)
Lz = AxialAngularMomentumHO(S)

# ``L_z`` does not conserve parity so we need both blocks. First combine the bases 
# of each block and convert to `DVecs`
dvs = map(b -> DVec(b => 1.0), vcat(basis1, basis2));
# and then compute overlaps
Lz_mat = [dot(v, Lz, w) for v in dvs, w in dvs]

# By diagonalising this matrix we obtain states of energy `E` and well-defined angular
# momentum
Lz_vals, Lz_vecs = eigen(Lz_mat)

# Finally we can consider the effect of interactions by looking at how states 
# in a single block are perturbed. We are interested in the energy shift due to 
# the interaction so we rebuild the Hamiltonian without the non-interacting energy
Hint = HOCartesianEnergyConserved(addr1; S, interaction_only = true)
ΔE = eigvals(Matrix(Hint, addr1))

# Two eigenstates in this block are unaffected by the interaction and three have a 
# non-zero energy shift.

# The default strength of the interaction is `g = 1.0`. Other interactions strengths
# can be obtained by using keyword argument `g` in `HOCartesianEnergyConserved` or by
# rescaling `ΔE` since the interactions are handled with first-order perturbation theory.

# `Rimu` also contains [`HOCartesianEnergyConservedPerDim`](@ref) which is a similar 
# Hamiltonian but with the stricter condition that the contact interaction only connects
# states that have the same total energy in each dimension, rather than conserving the 
# overall total energy. Both Hamiltonians can handle anisotropic systems by passing a tuple
# `S` whose elements are not all the same. This will alter which states are connected by the
# interaction, but assumes that the harmonic trapping frequencies in each dimension are 
# commensurate

# Finished!

using Test                                        #hide
@test nrow(block_df) == M*2^(D-1) + 1             #hide
@test E == E0 + M                                 #hide
@test Lz_vals ≈ [-3,-3,-1,-1,-1,1,1,1,3,3]        #hide