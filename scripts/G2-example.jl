# # Example: Two-body correlation function

# This is an example calculation of the two-body correlation function G_2.

# A runnable script for this example is located 
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/BHM-example.jl).
# Run it with `julia BHM-example.jl`.

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation, and `DataFrames` for output

using Rimu
using DataFrames
using Printf

# We use the same Hamiltonian as the previous examples,
# a Bose-Hubbard model with 6 particles in 6 sites, with
# strong interactions (we expect a Mott insulating state).
m = n = 6
aIni = near_uniform(BoseFS{n,m})
Ĥ = HubbardReal1D(aIni; u = 6.0, t = 1.0)

# Now we define the operators for the observables we wish to calculate
dvals = 0:m-1
G2list = ([G2RealCorrelator(d) for d in dvals]...,)
# This is a tuple of `G2RealCorrelator`s, which are subtyped to 
# `AbstractHamiltonian`, but with less functionality than a full Hamiltonian.
# It calculates the two-body correlation function on a lattice
# ```math
#     \\hat{G}^{(2)}(d) = \\frac{1}{M} \\sum_i^M \\hat{n}_i (\\hat{n}_{i+d} - \\delta_{0d}).
# ```
# with normalisation
# ```math
#     \\sum_{d=0}^{M-1} \\langle \\hat{G}^{(2)}(d) \\rangle = \\frac{N (N-1)}{M}.
# ```

# Observables are calculated using the "replica trick" which runs independent 
# copies of the model. We enable this by defining a `ReplicaStrategy`, in 
# this case, `AllOverlaps`. At each timestep, in addition to calculating FCIQMC 
# variables like the shift, this strategy calculates the overlaps of our operators 
# with the wavefunction from each pair of replicas. 
num_reps = 3
replica = AllOverlaps(num_reps; operator = G2list)


# We will deliberately use a small number of walkers to amplify
# the effect of the statistical tools that we will test.
targetwalkers = 100
# We need a reasonable number of timesteps to get good statistics
steps_equilibrate = 1_000
steps_measure = 5_000


# Other FCIQMC parameters are the same as before
dτ = 0.001
reporting_interval = 1
svec = DVec(aIni => 1)
seedCRNG!(17)

# Other FCIQMC strategies are the same as before

# Passing dτ and total number of time steps into params:
params = RunTillLastStep(dτ = dτ, laststep = steps_equilibrate + steps_measure)
# Strategy for updating the shift:
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
# Strategy for reporting info:
r_strat = ReportDFAndInfo(reporting_interval = reporting_interval, info_interval = 100)
# Strategy for updating dτ:
τ_strat = ConstantTimeStep()

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
            replica,
            threading = false, # only for reproducible runs
)

# The output `DataFrame` has FCIQMC statistics for each replica 
# (e.g. shift) 
println(filter(startswith("shift_"), names(df)))
# as well as vector-vector overlaps (e.g. `c1_dot_c2`)
println(filter(contains("dot"), names(df)))
# and operator overlaps (e.g. `c1_Op1_c2`) between the replicas.
println(filter(contains("Op"), names(df)))

# The vector-vector and operator overlaps go into calculating the Rayleigh quotient 
# for an observable
# ```math
#     \\langle \\hat{G}^{(2)}(d) \\rangle = \\frac{\\sum_{a<b} \\mathbf{c}_a^\\dagger \\hat{G}^{(2)}(d) \\mathbf{c}_b}{\\sum_{a<b} \\mathbf{c}_a^\\dagger \\cdot \\mathbf{c}_b \\}
# ```
# The sum over all replica pairs (a,b), especially in the denominator, helps to avoid 
# errors from poor sampling if the number of walkers is too low.

# We use the function `rayleigh_replica_estimator` to calculate the Rayleigh quotient 
# using all replicas in `df`, returning a `RatioBlockingResult` using `MonteCarloMeasurements`.

# Now we can calculate the correlation function for each value of `d`
println("Two-body correlator from $num_reps replicas:")
for d in dvals
    r = rayleigh_replica_estimator(df; op_name = "Op$(d+1)", skip=steps_equilibrate)
    println("   G2($d) = $(@sprintf("%.3f", r.f)) ± $(@sprintf("%.3f", r.σ_f))")
    # push!(dfg2, (; d, G2=r.f, err=r.σ_f))
end

# As expected, the onsite correlation at ``d=0`` is low since this is 
# a Mott insulating state with unit filling fraction, and is highest at 
# ``d=3`` which is the longest possible separation with periodic boundary conditions.

# Since we ran multiple independent replicas, we also have multiple estimates of 
# the shift energy
println("Shift energy from $num_reps replicas:")
for i in 1:num_reps
    se = shift_estimator(df; shift="shift_$i", skip=steps_equilibrate)
    println("   Replica $i: $(@sprintf("%.2f", se.mean)) ± $(@sprintf("%.2f", se.err))")
end

# Finished !
println("Finished!")
