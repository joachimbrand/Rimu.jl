# # Example 1: 1D Bose-Hubbard Model

# This is an example calculation finding the ground state of a 1D Bose-Hubbard chain with 6
# particles in 6 lattice sites.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/BHM-example.jl).
# Run it with `julia BHM-example.jl`.

# First, we load Rimu and Plots.

using Rimu
using Plots

# ## Setting up the model

# We start by defining the physical problem. First, we generate an initial configuration
# which will be used as a starting point of our computation. In this example, we use a
# bosonic Fock state with 6 particles evenly distributed in 6 lattice sites.
initial_address = near_uniform(BoseFS{6,6})

# The Hamiltonian is constructed by initializing a struct with an initial address and model
# parameters. Here, we use the Bose Hubbard model in one-dimensional real space.
H = HubbardReal1D(initial_address; u = 6.0, t = 1.0)

# ## Parameters of the calculation

# Now, let's setup the Monte Carlo calculation. We need to decide the number of walkers to
# use in this Monte Carlo run, which is equivalent to the average one-norm of the
# coefficient vector. Higher values will result in better statistics, but require more
# memory and computing power.
targetwalkers = 1_000;

# FCIQMC takes a certain number of steps to equllibrate, after which the observables will
# fluctuate around a mean value. In this example, we will devote 1000 steps to equilibration
# and take an additional 2000 steps for measurement.
steps_equilibrate = 1_000;
steps_measure = 2_000;
laststep = steps_equilibrate + steps_measure

# Next, we pick a time step size. FCIQMC does not have a time step error, but the
# time step needs to be small enough, or the computation might diverge. If the time step is
# too small, however, the computation might take a long time to equilibrate. The
# appropriate time step size is problem-dependent and is best determined through
# experimentation.
dτ = 0.001;

# ## Defining an observable

# Now, let's set up an observable to measure. Here we will measure the projected energy. In
# additon to the shift, the projected energy is a second estimator for the energy. It
# usually produces better statistics than the shift.

# We first need to define a projector. Here, we use the function
# [`default_starting_vector`](@ref) to generate a vector with only a single occupied
# configuration. We will use the same vector as the starting vector for the FCIQMC
# calculation.
initial_vector = default_starting_vector(initial_address; style=IsDynamicSemistochastic())

# The choice of the `style` argument already determines the FCIQMC algorithm to
# use. [`IsDynamicSemistochastic`](@ref) is usually the best choice as it reduces noise and
# improves the sign problem.

# Observables that can be calculated by projection of the fluctuating quantum state onto
# a constant vector are passed into the [`lomc!`](@ref) function with the `post_step`
# keyword argument.
post_step = ProjectedEnergy(H, initial_vector)

# ## Running the calculation

# In this example, we seed the random number generator in order to get reproducible results.
# This should not be done for actual computations.
using Random
Random.seed!(17);

# Finally, we can start the FCIQMC run.
df, state = lomc!(
    H, initial_vector;
    laststep,
    dτ,
    targetwalkers,
    post_step,
);

# Here, `df` is a `DataFrame` containing the time series data, while `state` contains the
# internal state of FCIQMC, which can be used to continue computations.

# ## Analysing the results

# We can plot the norm of the coefficient vector as a function of the number of steps.
hline(
    [targetwalkers];
    label="targetwalkers", xlabel="steps", ylabel="norm",
    color=2, linestyle=:dash, margin = 1Plots.cm
)
plot!(df.steps, df.norm, label="norm", color=1)

# After an initial equilibriation period, the norm fluctuates around the target number of
# walkers.

# Now, let's look at using the shift to estimate the ground state energy of `H`. The mean of
# the shift is a useful estimator of the energy. Calculating the error bars is a bit more
# involved as autocorrelations have to be removed from the time series. This can be done
# with the function [`shift_estimator`](@ref), which performs a blocking analysis.
se = shift_estimator(df; skip=steps_equilibrate)

# Here, `se` contains the calculated mean and standard errors of the shift, as well as some
# additional information related to the blocking analysis.

# Computing the error of the projected energy is a bit more complicated, as it's a ratio of
# fluctuating variables. Thankfully, the complications are handled by the following
# function.
pe = projected_energy(df; skip=steps_equilibrate)

# The result is a ratio distribution. We extract its median and the edges of the 95%
# confidence interval.
v = val_and_errs(pe; p=0.95)

# Let's visualise these estimators together with the time series of the shift.
plot(df.steps, df.shift, ylabel="energy", xlabel="steps", label="shift", margin = 1Plots.cm)

plot!(x->se.mean, df.steps[steps_equilibrate+1:end], ribbon=se.err, label="shift mean")
plot!(
    x -> v.val, df.steps[steps_equilibrate+1:end], ribbon=(v.val_l,v.val_u),
    label="projected energy",
)
lens!([steps_equilibrate, laststep], [-5.1, -2.9]; inset=(1, bbox(0.2, 0.25, 0.6, 0.4)))

# In this case the projected energy and the shift are close to each other and the error bars
# are hard to see.

# The problem was just a toy example, as the dimension of the Hamiltonian is rather small:
dimension(H)

# In this case, it's easy (and more efficient) to calculate the exact ground state energy
# using standard linear algebra. Read more about Rimu's capabilities for exact
# diagonalisation in the example "Exact diagonalisation".

using LinearAlgebra
exact_energy = eigvals(Matrix(H))[1]

# We finish by comparing our FCIQMC results with the exact computation.

println(
    """
    Energy from $steps_measure steps with $targetwalkers walkers:
    Shift: $(se.mean) ± $(se.err)
    Projected Energy: $(v.val) ± ($(v.val_l), $(v.val_u))
    Exact Energy: $exact_energy
    """
)


using Test                                      #hide
@test se.mean ≈ -4.0215 rtol=0.1;               #hide
