# # Example 2: Rimu with MPI

# In this example, we will demonstrate using Rimu with
# [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). MPI is a standard for
# parallel and distributed computing, and it is widely used in high-performance computing.
# Rimu provides support for MPI to enable parallel computations on multiple nodes.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/BHM-example-mpi.jl).
# Run it with 2 MPI ranks with `mpirun -n 2 julia BHM-example-mpi.jl`.

# We start by importing `Rimu`.
using Rimu

# Note that it is not necessary to initialise the MPI library, as this is already done
# automatically when Rimu is loaded.

# We will compute the ground state of a Bose-Hubbard model in momentum space with 10
# particles in 10 sites.

# First, we define the Hamiltonian. We want to start from an address with zero momentum,
# which is located at mode 5 in the momentum grid. We put all 10 particles, all in the
# zero momentum mode.
address = BoseFS(10, 5 => 10)

# We will set the interaction strength `u` to `6.0`. The hopping strength `t` defaults to
# `1.0`.
H = HubbardMom1D(address; u=6.0)

# We set a reporting strategy. We will use [`ReportToFile`](@ref), which writes the reports
# directly to a file. This is useful for MPI calculations, as they will typically run
# non-interactively. The reports will be written to disk and can be inspected later. This
# has the additional benefit of reducing memory use in long-running jobs, as we don't need
# to keep the results in memory. It also allows us to inspect the results before the
# computation finishes and recover some data if it fails.

# The default settings will ensure that only the root MPI
# rank will write to the file, which is reasonable, and that data is saved in chunks of
# 1000 time steps. We choose to suppress progress messages with setting `io=devnull`.
reporting_strategy = ReportToFile(
    filename="result.arrow",
    io=devnull
)

# For running parallel computations with MPI, it is important that a compatible state vector
# is used. Here we explicitly set up an MPI-enabled state vector, [`PDVec`](@ref),
# which is automatically MPI-distributed over the available number of MPI ranks. In
# addition, threading will be used with all threads available to Julia.
initial_vector = PDVec(address => 1.0; style=IsDynamicSemistochastic())

# Now, we can set other parameters as usual. We will perform the computation with 10000
# walkers and for 10000 time steps. We will also compute the projected energy by passing a
# [`ProjectedEnergy`](@ref) object as a `post_step_strategy`.
problem = ProjectorMonteCarloProblem(H;
    start_at=initial_vector,
    reporting_strategy,
    post_step_strategy=ProjectedEnergy(H, initial_vector),
    target_walkers=10_000,
    time_step=1e-4,
    last_step=10_000
);

# The [`@mpi_root`](@ref) macro performs an action on the root rank only, which is useful
# for printing.
@mpi_root println("Running FCIQMC with ", mpi_size(), " rank(s).")

# Finally, we can run the computation.
simulation = solve(problem);

@mpi_root println("Simulation success = ", simulation.success)

# Once the calculation is done, the results are available in the arrow file on disk.

# In a typical workflow, the simulation results would be loaded from disk and analysed in
# the REPL or with a separate script. The arrow file can be loaded into a `DataFrame`
# with metadata using the [`load_df`](@ref) function.

using Test                                          #hide
@test isfile("result.arrow")                        #hide
dfr = load_df("result.arrow")                       #hide
qmcdata = last(dfr, 5000)                           #hide
qmc_shift, _ = mean_and_se(qmcdata.shift)           #hide
@test qmc_shift ≈ -6.5 atol=0.5                     #hide
rm("result.arrow", force=true)                      #hide
