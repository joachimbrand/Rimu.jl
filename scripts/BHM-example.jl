# # Example 1: 1D Bose-Hubbard Model

# This is an example calculation finding the ground state of
# a 1D Bose-Hubbard chain with 6 particles in 6 lattice sites.

# A runnable script for this example is located
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/BHM-example.jl).
# Run it with `julia BHM-example.jl`.

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation;

using Rimu
using Plots # for plotting

# ## Setting up the model

# Now we define the physical problem:
# Generating a configuration where 6 particles are evenly distributed in 6 lattice sites:
aIni = near_uniform(BoseFS{6,6})
# where `BoseFS` is used to create a bosonic Fock state.
# The Hamiltonian is defined by specifying the model and the parameters.
# Here we use the Bose Hubbard model in one dimension and real space:
Ĥ = HubbardReal1D(aIni; u = 6.0, t = 1.0)

# ## Parameters of the calculation

# Now let's setup the Monte Carlo calculation.
# We need to decide the number of walkers to use in this Monte Carlo run, which is
# equivalent to the average one-norm of the coefficient vector:
targetwalkers = 1_000;

# It is good practice to equilibrate the time series before taking statistics.
steps_equilibrate = 1_000;
steps_measure = 2_000;

# The appropriate size of the time step is problem dependent.
dτ = 0.001;

# ## Defining an observable

# Now let's set up an observable to measure. Here we will measure the projected energy.
# In additon to the `shift`, the projected energy is a second estimator for the energy.

# We first need to define a projector. Here we use the function `default_starting_vector`
# to generate a vector with only a single occupied configuration, the same vector that we
# will use as starting vector for the FCIQMC calculation.
svec = default_starting_vector(aIni; style=IsStochasticInteger())
# The choice of the `style` argument already determines the FCIQMC algorithm to use
# (while it is irrelevant for the projected energy).

# Observables are passed into the `lomc!` function with the `post_step` keyword argument.
post_step = ProjectedEnergy(Ĥ, svec)

# ## Running the calculation

# Seeding the random number generator is sometimes useful in order to get reproducible
# results
using Random
Random.seed!(17);

# Finally, we can start the main FCIQMC loop:
df, state = lomc!(Ĥ, svec;
            laststep = steps_equilibrate + steps_measure, # total number of steps
            dτ,
            targetwalkers,
            post_step,
);

# `df` is a `DataFrame` containing the time series data.

# ## Analysing the results

# We can plot the norm of the coefficient vector as a function of the number of steps:
hline([targetwalkers], label="targetwalkers", color=:red, linestyle=:dash)
plot!(df.steps, df.norm, label="norm", ylabel="norm", xlabel="steps")
# After some equilibriation steps, the norm fluctuates around the target number of walkers.

# Now let's look at estimating the energy from the shift.
# The mean of the shift is a useful estimator of the shift. Calculating the error bars
# is a bit more involved as correlations have to be removed from the time series.
# The following code does that:
se = shift_estimator(df; skip=steps_equilibrate)

# For the projected energy, it a bit more complicated as it's a ratio of two means:
pe = projected_energy(df; skip=steps_equilibrate)

# The result is a ratio distribution. Let's get its median and lower and upper error bars
# for a 95% confidence interval
v = val_and_errs(pe; p=0.95)

# Let's visualise these estimators together with the time series of the shift
plot(df.steps, df.shift, ylabel="energy", xlabel="steps", label="shift")

plot!(x->se.mean, df.steps[steps_equilibrate+1:end], ribbon=se.err, label="shift mean")
plot!(
    x -> v.val, df.steps[steps_equilibrate+1:end], ribbon=(v.val_l,v.val_u),
    label="projected_energy",
)
# In this case the projected energy and the shift are close to each other an the error bars
# are hard to see on this scale.

# The problem was just a toy example, as the dimension of the Hamiltonian is rather small:
dimension(Ĥ)

# In this case is easy (and more efficient) to calculate the exact ground state energy
# using standard linear algebra:
using LinearAlgebra
exact_energy = eigvals(Matrix(Ĥ))[1]

# Comparing our results for the energy:
println("Energy from $steps_measure steps with $targetwalkers walkers:
         Shift: $(se.mean) ± $(se.err)
         Projected Energy: $(v.val) ± ($(v.val_l), $(v.val_u))
         Exact Energy: $exact_energy")


using Test                                      #hide
@test se.mean ≈ -4.0215 rtol=0.1;               #hide
