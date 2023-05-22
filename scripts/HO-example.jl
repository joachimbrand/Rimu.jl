# # Example 5: Degenerate perturbation theory in a harmonic oscillator basis

# `Rimu` can also handle non-lattice systems. This example shows how a harmonic 
# oscillator Hamiltonian is implemented, and how to calculate blocks of energy 
# eigenstates in the Hamiltonian and angular momentum eigenstates.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/HO-example.jl).
# Run it with `julia HO-example.jl`.

# First, load all needed modules.
using Rimu
using DataFrames
using LinearAlgebra

# Define the system size for ``N=2`` particles in a 2D harmonic oscillator
# allowing ``M=4`` levels in each dimension, including the groundstate.
N = 2
M = 4

# Use a tuple `S` to define the range of harmonic oscillator states in a Cartesian
# basis, in this isotropic case ``n_x,n_y=0,1,\ldots,M-1``.
S = (M, M)

# In `Rimu` the ``N``-particle states are still stored as Fock states
P = prod(S)
addr = BoseFS(P, M => N)
# where the numbering of the modes folds in the two spatial dimensions. 
# Use the utility function [`fock_to_cart`](@ref) to convert a Fock address to 
# human-readable Cartesian quantum numbers for inspection.
fock_to_cart(addr, S)
# The output shows that all ``N`` particles are in single-particle state ``n_x=M-1, n_y=0``.

# The harmonic oscillator Hamiltonian [`HOCartesianEnergyConserved`](@ref) handles 
# contact interactions with
# first-order perturbation theory, so the matrix representation will block
# according to the non-interacting energy of the basis states. The first task is 
# to find all blocks of basis states with the same energy. The strength of the 
# interaction is not relevant at this point, just that it is non-zero.
# Use a dummy groundstate address to build the Hamiltonian 
H = HOCartesianEnergyConserved(BoseFS(P, 1 => N); S)
# and then a utility function [`get_all_blocks`](@ref) to find all blocks. The blocks 
# are found by looping over all possible states with `N` particles in Cartesian states 
# defined by `S`. Note that this will only work for total energy 
# up to the maximum accessible by a single particle.
# The ``N``-particle groundstate energy for a 2D harmonic oscillator is 
# ``E_0 = N \hbar \omega`` and the maximum single-particle energy is ``E = (E_0 + M - 1) \hbar \omega``.
block_df = get_all_blocks(H; max_energy = N + M - 1)

# This outputs a list of blocks in `H` indexed by the noninteracting energy
# of all states in the block, and a single address that can be used to 
# rebuild the block for further analysis.
addr1 = block_df[7,:addr]
E = block_df[7,:block_E0]
# First, notice that all basis states have the same energy, defined by the block
basis1 = build_basis(H, addr1)
map(b -> Hamiltonians.noninteracting_energy(H, b), basis1)

# There are two blocks at each energy level (except the groundstate), which 
# are different due to parity conservation, which is the only other symmetry in the 
# Cartesian harmonic oscillator.
# The basis of this other block is different 
addr2 = block_df[4,:addr]
basis2 = build_basis(H, addr2);
basis1 ≠ basis2
# but its basis elements have the same energy
map(b -> Hamiltonians.noninteracting_energy(H, b), basis2)

# However, since this system is an isotropic harmonic oscillator, it is possible 
# to build simultaneous eigenstates of the angular momentum operator
# ``L_z``, implemented with [`AxialAngularMomentumHO`](@ref)
Lz = AxialAngularMomentumHO(S)

# ``L_z`` does not conserve parity so both blocks are required. First combine the bases 
# of each block and convert to `DVecs`
dvs = map(b -> DVec(b => 1.0), vcat(basis1, basis2));
# and then compute overlaps for the matrix elements of ``L_z``
Lz_mat = [dot(v, Lz, w) for v in dvs, w in dvs]

# By diagonalising this matrix the eigenstate have energy `E` and well-defined angular
# momentum
Lz_vals, Lz_vecs = eigen(Lz_mat)

# Finally, consider the effect of interactions by looking at how states 
# in a single block are perturbed. Only the energy shift due to the interaction is 
# relevant so now rebuild the Hamiltonian without the non-interacting energy
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

using Test                                      #hide
@test nrow(block_df) == 2*(M-1) + 1             #hide
@test E == N + M - 1                           #hide
@test Lz_vals ≈ [-3,-3,-1,-1,-1,1,1,1,3,3]      #hide
nothing                                         #hide
