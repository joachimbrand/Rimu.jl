# Example: Bose-Hubbard Model

This is an example calculation finding the ground state of
a 1D Bose-Hubbard chain with 6 particles in 6 lattice site.
The Julia run-able script is in [`scripts/BHM-example.jl`](../../scripts/BHM-example.jl).

Firstly, we load all needed modules.
`Rimu` for FCIQMC calculation (obviously);
`Feather` for saving output data in a `DataFrame` using `DataFrames`:

```@example BHM-example
using Rimu
using Feather
using DataFrames
```

Now we define the physical problem:
Setting the number of lattice sites `m = 6`;
and the number of particles `n = 6`:

```@example BHM-example
m = n = 6
```

Generating a configuration that particles are evenly distributed:

```@example BHM-example
aIni = nearUniform(BoseFS{n,m})
```

The Hamiltonian is defined based on the configuration `aIni`,
with additional onsite interaction strength `u = 6.0`
and the hopping strength `t = 1.0`:

```@example BHM-example
Ĥ = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
```

Now let's setup the Monte Carlo settings.
The number of walkers to use in this Monte Carlo run:

```@example BHM-example
targetwalkers = 1_000
```

The number of time steps before doing statistics,
i.e. letting the walkers to sample Hilbert and to equilibrate:

```@example BHM-example
steps_equilibrate = 1_000
```

And the number of time steps used for getting statistics,
e.g. time-average of shift, projected energy, walker numbers, etc.:

```@example BHM-example
steps_measure = 1_000
```

Set the size of a time step

```@example BHM-example
dτ = 0.001
```

and we report QMC data every k-th step,
setting `k = 1` means we record QMC data every step:

```@example BHM-example
k = 1
```

Now we prepare initial state and allocate memory.
The initial address is defined above as `aIni = nearUniform(Ĥ)`.
Define the initial number of walkers per rank:

```@example BHM-example
nIni = 1
```

Putting the `nIni` number of walkers into the initial address `aIni`,
and set the DVec size to be targetwalkers*10:

```@example BHM-example
svec = DVec(Dict(aIni => nIni), targetwalkers*10)
```

Let's plant a seed for the random number generator to get consistent result:

```@example BHM-example
Rimu.ConsistentRNG.seedCRNG!(17)
```

Now let's setup all the FCIQMC strategies.

Passing dτ and total number of time steps into params:

```@example BHM-example
params = RunTillLastStep(dτ = dτ, laststep = steps_equilibrate + steps_measure)
```

Strategy for updating the shift:

```@example BHM-example
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
```

Strategy for reporting info and setting projectors:

```@example BHM-example
r_strat = ReportDFAndInfo(k = k, i = 100, projector = UniformProjector())
```

Strategy for updating dτ:

```@example BHM-example
t_strat = ConstantTimeStep()
```

Print out info about what we are doing:

```@example BHM-example
println("Finding ground state for:")
println(Ĥ)
println("Strategies for run:")
println(params, s_strat)
println(t_strat)
```

Finally, we can start the main FCIQMC loop:

```@example BHM-example
r_strat = ReportDFAndInfo(k = k, projector = UniformProjector(), writeinfo = false) # hide
df = lomc!(Ĥ,svec;
            params = params,
            laststep = steps_equilibrate + steps_measure,
            s_strat = s_strat,
            r_strat = r_strat,
            τ_strat = t_strat)
```

Saving output data stored in `df.df` into a `.feather` file which can be read in later:

```@example BHM-example
println("Writing data to disk...")
Feather.write("fciqmcdata.feather", df.df)
```

Now do some quick statistics:

```@example BHM-example
(qmcShift,qmcShiftErr,qmcEnergy,qmcEnergyErr) = autoblock(df.df,start=steps_equilibrate)
println("Energy from $steps_measure steps with $targetwalkers walkers:
Shift: $qmcShift ± $qmcShiftErr
E_proj:$qmcEnergy ± $qmcEnergyErr")
```

Finished !

```@example BHM-example
println("Finished!")
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
