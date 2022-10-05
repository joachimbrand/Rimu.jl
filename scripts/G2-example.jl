# # Example 3: Calculating observables

# This is an example calculation of the two-body correlation function G_2.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/G2-example.jl).
# Run it with `julia G2-example.jl`.

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation, and `DataFrames` for output

using Rimu
using Random
using DataFrames

# We use the same Hamiltonian as the first example,
# a Bose-Hubbard model with 6 particles in 6 sites, with
# strong interactions (we expect a Mott insulating state).
m = n = 6
aIni = near_uniform(BoseFS{n,m})
H = HubbardReal1D(aIni; u = 6.0, t = 1.0)

# Now we define the operators for the observables we wish to calculate
dvals = 0:m-1
G2list = ([G2RealCorrelator(d) for d in dvals]...,)
# This is a tuple of [`G2RealCorrelator`](@ref)s, which are subtyped to
# [`AbstractHamiltonian`](@ref), but with less functionality than a full Hamiltonian.
# It calculates the two-body correlation function on a lattice
# ```math
#     \hat{G}^{(2)}(d) = \frac{1}{M} \sum_i^M \hat{n}_i (\hat{n}_{i+d} - \delta_{0d}).
# ```
# with normalisation
# ```math
#     \sum_{d=0}^{M-1} \langle \hat{G}^{(2)}(d) \rangle = \frac{N (N-1)}{M}.
# ```

# Observables are calculated using the "replica trick" whereby several
# copies or "replicas" of the model are run simultaneously. We enable this by defining
# a [`ReplicaStrategy`](@ref). Each replica has its own state and FCIQMC is effectively
# performed independently on each one.
# For calculating observables, we use [`AllOverlaps`](@ref) for the `ReplicaStrategy`.
# At each timestep, after the necessary FCIQMC variables are calculated for each replica,
# (e.g. shift, norm etc.), this strategy calculates the overlaps of every operator with
# the wavefunctions from each pair of replicas.
num_reps = 3
replica = AllOverlaps(num_reps; operator = G2list)

# We need a reasonable number of timesteps to get good statistics, and we are running
# multiple replicas, so we will only use a small number of walkers:
steps_equilibrate = 1_000
steps_measure = 5_000
targetwalkers = 100;

# Other FCIQMC parameters and strategies are the same as before
dτ = 0.001
reporting_interval = 1
svec = DVec(aIni => 1)
Random.seed!(17)
params = RunTillLastStep(dτ = dτ, laststep = steps_equilibrate + steps_measure)
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
r_strat = ReportDFAndInfo(reporting_interval = reporting_interval, info_interval = 100)
τ_strat = ConstantTimeStep();

# Now we run the main FCIQMC loop:
df, state = lomc!(H, svec;
            params,
            laststep = steps_equilibrate + steps_measure,
            s_strat,
            r_strat,
            τ_strat,
            replica,
            threading = false, # only for reproducible runs
);

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
#     \langle \hat{G}^{(2)}(d) \rangle = \frac{\sum_{a<b} \mathbf{c}_a^\dagger \cdot \hat{G}^{(2)}(d) \cdot \mathbf{c}_b}{\sum_{a<b} \mathbf{c}_a^\dagger \cdot \mathbf{c}_b }
# ```
# The sum over all replica pairs (a,b), especially in the denominator, helps to avoid
# errors from poor sampling if the number of walkers is too low.

# We use the function [`rayleigh_replica_estimator`](@ref)
# to calculate the Rayleigh quotient using all replicas in `df`, returning a
# `RatioBlockingResult` using `MonteCarloMeasurements`. Using the keyword `skip`
# will ignore the initial equilibration steps.

# Now we can calculate the correlation function for each value of `d`
println("Two-body correlator from $num_reps replicas:")
for d in dvals
    r = rayleigh_replica_estimator(df; op_name = "Op$(d+1)", skip=steps_equilibrate)
    println("   G2($d) = $(r.f) ± $(r.σ_f)")
end

# As expected, the onsite correlation at ``d=0`` is low since this is
# a Mott insulating state with unit filling fraction, and is highest at
# ``d=3`` which is the longest possible separation with periodic boundary conditions.

# Since we ran multiple independent replicas, we also have multiple estimates of
# the shift energy
println("Shift energy from $num_reps replicas:")
for i in 1:num_reps
    se = shift_estimator(df; shift="shift_$i", skip=steps_equilibrate)
    println("   Replica $i: $(se.mean) ± $(se.err)")
end

# Finished !

using Test                                  #hide
r = rayleigh_replica_estimator(df; op_name = "Op1", skip=steps_equilibrate) #hide
@test r.f ≈ 0.2 rtol=0.1                    #hide
nothing                                     #hide
