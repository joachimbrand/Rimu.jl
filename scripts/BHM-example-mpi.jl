# In this example, we will demonstrate using Rimu with MPI.  A runnable script is located at
# [here](../../scripts/BHM-example-mpi.jl). Run it as `mpirun julia BHM-example.jl`.

# We start by importing `Rimu` and `Rimu.RMPI`, which contains MPI-replated
# functionality.
using Rimu
using Rimu.RMPI

# We will compute the ground-state of a Bose-Hubbard model in momentum space with 10 particles
# in 10 sites.

# First, we define the Hamiltonian. We want to start from an address with zero momentum.

address = BoseFS((0, 0, 0, 0, 10, 0, 0, 0, 0, 0))

# We will set the interaction strength `u` to `6`. The hopping strength `t` defaults to `1.0`.

hamiltonian = HubbardMom1D(address; u=6.0)

# Next, we construct the starting vector. Wrap a vector in `MPIData` to make it MPI
# distributed. We set the vector's style to [`IsDynamicSemistochastic`](@ref), which
# improves statistics and reduces the sign problem.

dvec = MPIData(DVec(address => 1.0; style=IsDynamicSemistochastic()))

# We set a reporting strategy. We will use [`ReportToFile`](@ref), which writes the reports
# directly to a file. This is useful for reducing memory use in long-running jobs, as we
# don't need to keep the results in memory. Setting `save_if=is_mpi_root()` will ensure only
# the root MPI rank will write to the file. The `chunk_size` parameter determines how often
# the data is saved to the file.

r_strat = ReportToFile(filename="result.arrow", save_if=is_mpi_root(), chunk_size=1000)

# Now, we can set other parameters as usual. We will perform the computation with 10_000
# walkers. We will also compute the projected energy.

s_strat = DoubleLogUpdate(targetwalkers=10_000)
post_step = ProjectedEnergy(hamiltonian, dvec)

# Finally, we can run the computation. The `@mpi_root` macro performs an action on the root
# rank only, which is useful for printing.

lomc!(hamiltonian, dvec; r_strat, s_strat, post_step, dτ=1e-4, laststep=10_000)
@mpi_root println("Finished!")
