# # Example: 1D Bose-Hubbard Model

# This is an example calculation finding the ground state of
# a 1D Bose-Hubbard chain with 6 particles in 6 lattice sites.

# A runnable script for this example is located 
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/Ex1-BHM.jl).
# Run it with `julia Ex1-BHM.jl`.

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation;

using Rimu

# Now we define the physical problem:
# Setting the number of lattice sites `m = 6`;
# and the number of particles `n = 6`:
m = n = 6
# Generating a configuration that particles are evenly distributed:
aIni = near_uniform(BoseFS{n,m})
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
steps_measure = 2_000


# Set the size of a time step
dτ = 0.001
# and we report QMC data every k-th step,
# set the interval to record QMC data:
reporting_interval = 1

# Now we prepare initial state and allocate memory.
# The initial address is defined above as `aIni = near_uniform(Ĥ)`.
# Putting one of walkers into the initial address `aIni`
svec = DVec(aIni => 1)
# Let's plant a seed for the random number generator to get consistent result:
seedCRNG!(17)

# Now let's setup all the FCIQMC strategies.

# Passing dτ and total number of time steps into params:
params = RunTillLastStep(dτ = dτ, laststep = steps_equilibrate + steps_measure)
# Strategy for updating the shift:
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
# Strategy for reporting info:
r_strat = ReportDFAndInfo(reporting_interval = reporting_interval, info_interval = 100)
# Strategy for updating dτ:
τ_strat = ConstantTimeStep()
# set up the calculation and reporting of the projected energy
# in this case we are projecting onto the starting vector,
# which contains a single configuration
post_step = ProjectedEnergy(Ĥ, copy(svec))

# Print out info about what we are doing:
println("Finding ground state for:")
println(Ĥ)
println("Strategies for run:")
println(params, s_strat)
println(τ_strat)

# Finally, we can start the main FCIQMC loop:
df, state = lomc!(Ĥ,svec;
            params,
            laststep = steps_equilibrate + steps_measure,
            s_strat,
            r_strat,
            τ_strat,
            post_step,
            threading = false, # only for reproducible runs
)

# Here is how to save the output data stored in `df` into a `.arrow` file,
# which can be read in later:
println("Writing data to disk...")
save_df("fciqmcdata.arrow", df)

# Now let's look at the calculated energy from the shift.
# Loading the equilibrated data
qmcdata = last(df,steps_measure);

# compute the average shift and its standard error
se = shift_estimator(qmcdata)

# For the projected energy, it a bit more complicated as it's a ratio of two means:
pe = projected_energy(qmcdata)

# The result is a ratio distribution. Let's get its median and lower and upper error bars
# for a 95% confidence interval
v = val_and_errs(pe; p=0.95)

println("Energy from $steps_measure steps with $targetwalkers walkers:
         Shift: $(se.mean) ± $(se.err);
         Projected Energy: $(v.val) ± ($(v.val_l), $(v.val_u))")

# Finished !
println("Finished!")
