# Projector Monte Carlo / FCIQMC

The purpose of Projector Monte Carlo is to stochastically sample the ground state, i.e. the 
eigenvector corresponding to the lowest eigenvalue of a quantum Hamiltonian, or more generally, 
a very large matrix. Rimu implements a flavor of Projector Monte Carlo called 
Full Configuration Interaction Quantum Monte Carlo (FCIQMC).

## `ProjectorMonteCarloProblem`

To run a projector Monte Carlo simulation you set up a problem with `ProjectorMonteCarloProblem`
and solve it with `solve`. Alternatively you can initialize a `PMCSimulation` struct, `step!` 
through time steps, and `solve!` it to completion. 

```@docs; canonical=false
ProjectorMonteCarloProblem
init
solve
solve!
step!
```

After `solve` or `solve!` have been called the returned `PMCSimulation` contains the results of 
the projector Monte Carlo calculation.

### `PMCSimulation` and report as a `DataFrame`

```@docs; canonical=false
Rimu.PMCSimulation
```

The `DataFrame` returned from `DataFrame(::PMCSimulation)` contains the time series data from 
the projector Monte Carlo simulation that is of primary interest for analysis. Depending on the 
`reporting_strategy` and other options passed as keyword arguments to 
`ProjectorMonteCarloProblem` it can have different numbers of rows and columns. The rows 
correspond to the reported time steps (Monte Carlo steps). There is at least one column with the name `:step`. Further columns are usually present with additional data reported from the simulation.

For the default option `algorithm = FCIQMC(; shift_strategy, time_step_strategy)` with a single
replica (`n_replicas = 1`) and single spectral state, the fields `:shift`, `:norm`, `:len` will 
be present as well as others depending on the `style` argument and the `post_step_strategy`.

If multiple replicas or spectral states are requested, then the relevant field names in the 
`DataFrame` will have a suffix identifying the respective replica simulation, e.g. the `shift`s will be reported as `shift_1`, `shift_2`, ... 

Many tools for analysing the time series data obtained from a 
[`ProjectorMonteCarloProblem`](@ref) are contained in the [Module `StatsTools`](@ref).
