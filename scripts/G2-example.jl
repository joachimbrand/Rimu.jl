# # Example: G_2 correlation function

# This is an example calculation of the two-body correlation 
# function G_2.
# The Julia run-able script is in [`scripts/BHM-example.jl`](../../../scripts/G2-example.jl).

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation, and `DataFrames` for output

using Rimu
using DataFrames

# We use the same Hamiltonian as the previous examples,
# a Bose-Hubbard model with 6 particles in 6 sites, with
# strong interactions (we expect a Mott insulating state).
m = n = 6
aIni = near_uniform(BoseFS{n,m})
Ĥ = HubbardReal1D(aIni; u = 6.0, t = 1.0)

# Now we define the operators for the observables we wish to calculate
G2list = ([G2RealCorrelator(d) for d in 0:m-1]...,)
# This is a tuple of `G2RealCorrelator`s, which are subtyped to 
# `AbstractHamiltonian`, but with less functionality than a full Hamiltonian.

# Observables are calculated using the "replica trick" which runs independent 
# copies of the model. We enable this by defining a `ReplicaStrategy`, in 
# this case, `AllOverlaps`. At each timestep, in addition to calculating FCIQMC 
# variables like the shift, this strategy calculates the overlaps of our operators 
# with the wavefunction from each pair of replicas. 
num_reps = 2
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
# (shift, norm, etc...) as well as vector-vector overlaps "c1_dot_c2" 
# and operator overlaps "c1_Op1_c2" between the replicas.
println(names(df))

# We can look at the shift energy for each replica, skipping the equilibration steps.
se1 = shift_estimator(df; shift="shift_1", skip=steps_equilibrate)
se2 = shift_estimator(df; shift="shift_2", skip=steps_equilibrate)

println("Energy from $steps_measure steps with $targetwalkers walkers:
         Replica 1: $(se1.mean) ± $(se1.err);
         Replica 2: $(se2.mean) ± $(se2.err);")

# Now we look at the G_2 correlation function, which is defined 
# by a Rayleigh quotient.
# G2 is normalised to `n*(n-1)/m` when summed over all `d`, so we can 
# correct for that by passing a scalar
Anorm = m^2 / n*(n-1)
# We calculate the correlation function for each value of `d` and 
# save to a `DataFrame`.
dfg2 = DataFrame()
for d in dvals
    r = rayleigh_replica_estimator(df; op_ol = "Op$(d+1)", skip=steps_equilibrate, Anorm)
    push!(dfg2, (; d, G2=r.mean, err=r.err))
end

# The results:
show(dfg2)

# As expected, the onsite correlation `(d=0)` is low since this is 
# a Mott insulating state, and the correlation between sites at distance
# `d` quickly rises to the long-range value


# Finished !
println("Finished!")
