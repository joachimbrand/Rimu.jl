# # Example 3: Calculating observables

# This is an example calculation of the two-body correlation function ``G_2``.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/G2-example.jl).
# Run it with `julia G2-example.jl`.

# First, we load the reqired packages. `Rimu` for FCIQMC calculation, and `DataFrames` for
# maniplating the output.
using Rimu
using Random
using DataFrames

# We use the same Hamiltonian as the first example, a Bose-Hubbard model with 6 particles in
# 6 sites, with strong interactions (we expect a Mott insulating state).
m = n = 6
initial_address = near_uniform(BoseFS{n,m})
H = HubbardReal1D(initial_address; u = 6.0, t = 1.0)

# Now, we define the operators for the observables we wish to calculate.
dvals = 0:m-1
G2list = ((G2RealCorrelator(d) for d in dvals)...,)

# This is a tuple of [`G2RealCorrelator`](@ref)s, subtypes of
# [`AbstractHamiltonian`](@ref).
# It calculates the density-density correlation function on a lattice
# ```math
#     \hat{G}^{(2)}(d) = \frac{1}{M} \sum_i^M \hat{n}_i (\hat{n}_{i+d} - \delta_{0d}).
# ```
# with normalisation
# ```math
#     \sum_{d=0}^{M-1} \langle \hat{G}^{(2)}(d) \rangle = \frac{N (N-1)}{M}.
# ```

# Observables that are defined by expectation values are calculated using the
# "replica trick". Thereby several independent copies or "replicas"
# of the state vector are propagated simultaneously. The reason is to have two (or more)
# stochastically independent copies of the state vector available such that we can
# calculate bias-free overlaps. We enable this by defining a
# [`ReplicaStrategy`](@ref). Each replica has its own state and FCIQMC is effectively
# performed independently on each one.  For calculating observables, we use
# [`AllOverlaps`](@ref) for the [`ReplicaStrategy`](@ref). At each timestep, after the
# FCIQMC step is performed on, this strategy calculates the overlaps of every operator with
# the wavefunctions from each pair of replicas.

# To obtain an unbiased result, at least two replicas should be used. One can also use more
# than two to improve the statistics. This is particularly helpful when the walker number is
# low.
n_replicas = 3
replica_strategy = AllOverlaps(n_replicas; operator=G2list)

# Other FCIQMC parameters and strategies can be set in the same way as before.
steps_equilibrate = 1_000
steps_measure = 5_000
target_walkers = 100;
time_step = 0.001

Random.seed!(17); #hide

# Now, we run FCIQMC. Note that passing an initial vector is optional - if we only pass the
# style, a vector with the appropriate style is created automatically.
problem = ProjectorMonteCarloProblem(H;
    style=IsDynamicSemistochastic(),
    time_step,
    last_step = steps_equilibrate + steps_measure,
    target_walkers,
    replica_strategy,
)
result = solve(problem)
df = DataFrame(result);

# The output `DataFrame` has FCIQMC statistics for each replica (e.g. shift, norm),
println(filter(startswith("shift_"), names(df)))

# as well as vector-vector overlaps (e.g. `c1_dot_c2`),
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

# We use the function [`rayleigh_replica_estimator`](@ref) to calculate the Rayleigh
# quotient using all replicas in `df`, returning a [`RatioBlockingResult`](@ref
# Main.StatsTools.RatioBlockingResult). Using the keyword `skip` will ignore the initial
# equilibration steps.

# Now, we can calculate the correlation function for each value of ``d``.
println("Two-body correlator from $n_replicas replicas:")
for d in dvals
    r = rayleigh_replica_estimator(df; op_name = "Op$(d+1)", skip=steps_equilibrate)
    println("   G2($d) = $(r.f) ± $(r.σ_f)")
end

# As expected, the onsite correlation at ``d=0`` is low since this is a Mott insulating
# state with unit filling fraction, and is close to ``1.0`` for all other values of the
# displacement ``d``.

# Since we ran multiple independent replicas, we also have multiple estimates of the shift
# energy.
println("Shift energy from $n_replicas replicas:")
for i in 1:n_replicas
    se = shift_estimator(df; shift="shift_$i", skip=steps_equilibrate)
    println("   Replica $i: $(se.mean) ± $(se.err)")
end

using Test                                  #hide
r = rayleigh_replica_estimator(df; op_name="Op1", skip=steps_equilibrate) #hide
@test r.f ≈ 0.2 rtol=0.1                    #hide
nothing                                     #hide
