# # Example 2: Rimu with MPI

# In this example, we will demonstrate using Rimu with MPI.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/BHM-example-mpi.jl).
# Run it with `mpirun julia BHM-example-mpi.jl`.

# We start by importing `Rimu` and `Rimu.RMPI`, which contains MPI-related
# functionality.
using Rimu
using Rimu.RMPI

# Note that it is not necessary to initialise the MPI library, as this is already done automatically
# when Rimu is loaded.

# We will compute the ground state of a Bose-Hubbard model in momentum space with 10
# particles in 10 sites.

# First, we define the Hamiltonian. We want to start from an address with zero momentum.
address = BoseFS(10, 5 => 10)

# We will set the interaction strength `u` to `6.0`. The hopping strength `t` defaults to
# `1.0`.
H = HubbardMom1D(address; u=6.0)

# Next, we construct the starting vector. We use a [`PDVec`](@ref), which is automatically
# MPI-distributed if MPI is available. We set the vector's stochastic style to
# [`IsDynamicSemistochastic`](@ref), which improves statistics and reduces the sign problem.
initial_vector = PDVec(address => 1.0; style=IsDynamicSemistochastic())

# We set a reporting strategy. We will use [`ReportToFile`](@ref), which writes the reports
# directly to a file. This is useful for reducing memory use in long-running jobs, as we
# don't need to keep the results in memory. It also allows us to inspect the results before
# the computation finishes and recover some data if it fails. Setting
# `save_if=is_mpi_root()` will ensure only the root MPI rank will write to the file. The
# `chunk_size` parameter determines how often the data is saved to the file. Progress
# messages are suppressed with `io=devnull`.
reporting_strategy = ReportToFile(
    filename="result.arrow",
    save_if=is_mpi_root(),
    reporting_interval=1,
    chunk_size=1000,
    io=devnull
)

# Now, we can set other parameters as usual. We will perform the computation with 10_000
# walkers. We will also compute the projected energy.
s_strat = DoubleLogUpdate(targetwalkers=10_000)
post_step = ProjectedEnergy(H, initial_vector)

# The [`@mpi_root`](@ref Main.Rimu.RMPI.@mpi_root) macro performs an action on the root rank
# only, which is useful for printing.
@mpi_root println("Running FCIQMC with ", mpi_size(), " rank(s).")

# Finally, we can run the computation.
lomc!(H, initial_vector; reporting_strategy, s_strat, post_step, dτ=1e-4, laststep=10_000);

using Test                                          #hide
@test isfile("result.arrow")                        #hide
dfr = load_df("result.arrow")                       #hide
qmcdata = last(dfr, 5000)                           #hide
qmc_shift, _ = mean_and_se(qmcdata.shift)           #hide
@test qmc_shift ≈ -6.5 atol=0.5                     #hide
rm("result.arrow", force=true)                      #hide
