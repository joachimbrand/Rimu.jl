# # Example: 1D Bose-Hubbard Model

# This is an example calculation finding the ground state of
# a 1D Bose-Hubbard chain with 6 particles in 6 lattice sites.
# The Julia run-able script is in [`scripts/BHM-example.jl`](../../scripts/BHM-example.jl).

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation;

using Rimu

# Now we define the physical problem:
# Setting the number of lattice sites `m = 6`;
# and the number of particles `n = 6`:
m = n = 6
# Generating a configuration that particles are evenly distributed:
aIni = nearUniform(BoseFS{n,m})
# where `BoseFS` is used to create a bosonic system.
# The Hamiltonian is defined based on the configuration `aIni`,
# with additional onsite interaction strength `u = 6.0`
# and the hopping strength `t = 1.0`:
Ĥ = HubbardReal1D(aIni; u = 6.0, t = 1.0)


# Now let's setup the Monte Carlo settings.
# The number of walkers to use in this Monte Carlo run:
targetwalkers = 1_000
# The number of time steps before doing statistics,
# i.e. letting the walkers to sample Hilbert and to equilibrate:
steps_equilibrate = 1_000
# And the number of time steps used for getting statistics,
# e.g. time-average of shift, projected energy, walker numbers, etc.:
steps_measure = 1_000


# Set the size of a time step
dτ = 0.001
# and we report QMC data every k-th step,
# setting `k = 1` means we record QMC data every step:
k = 1

# Now we prepare initial state and allocate memory.
# The initial address is defined above as `aIni = nearUniform(Ĥ)`.
# Define the initial number of walkers per rank:
nIni = 1
# Putting the `nIni` number of walkers into the initial address `aIni`
svec = DVec(aIni => nIni)
# Let's plant a seed for the random number generator to get consistent result:
Rimu.ConsistentRNG.seedCRNG!(17)

# Now let's setup all the FCIQMC strategies.

# Passing dτ and total number of time steps into params:
params = RunTillLastStep(dτ = dτ, laststep = steps_equilibrate + steps_measure)
# Strategy for updating the shift:
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
# Strategy for reporting info:
r_strat = ReportDFAndInfo(k = k, i = 100)
# Strategy for updating dτ:
t_strat = ConstantTimeStep()
# set up the calculation and reporting of the projected energy
# in this case we are projecting onto the starting vector,
# which contains a single configuration
post_step = ProjectedEnergy(Ĥ, copy(svec))

# Print out info about what we are doing:
println("Finding ground state for:")
println(Ĥ)
println("Strategies for run:")
println(params, s_strat)
println(t_strat)

# Finally, we can start the main FCIQMC loop:
df, state = lomc!(Ĥ,svec;
            params = params,
            laststep = steps_equilibrate + steps_measure,
            s_strat = s_strat,
            r_strat = r_strat,
            τ_strat = t_strat,
            post_step = post_step,
            threading = false, # only for reproducible runs
)
println("Writing data to disk...")

# Saving output data stored in `df` into a `.arrow` file which can be read in later:
save_df("fciqmcdata.arrow", df)

# Now let's look at the calculated energy from the shift:
# Loading the equilibrated data:
qmcdata = last(df,steps_measure)
using Rimu.StatsTools

# For the shift, it's easy to use `mean_and_se` from `Rimu.StatsTools`
(qmcShift,qmcShiftErr) = mean_and_se(qmcdata.shift)

# For the projected energy, it a bit more complicated as it's a ratio of two means:
r = ratio_of_means(qmcdata.hproj,qmcdata.vproj)
rwe = ratio_with_errs(r)

# Here we use the 95% CI for the lower and upper error bars:
(eProj,eProjErrLower,eProjErrUpper) = (rwe.ratio, rwe.err2_l, rwe.err2_u)

println("Energy from $steps_measure steps with $targetwalkers walkers:
         Shift: $qmcShift ± $qmcShiftErr;
         Projected Energy: $eProj ± ($eProjErrLower, $eProjErrUpper)")

# Finished !
println("Finished!")
