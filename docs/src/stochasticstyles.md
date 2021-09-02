# Module `StochasticStyles`

This module contains the implementations of [`StochasticStyle`](@ref)s, which control how the
stochastic matrix-vector multiplication is performed.

```@docs
StochasticStyle
```

## Available `StochasticStyle`s

```@docs
StyleUnknown
IsStochasticInteger
IsDeterministic
IsStochasticWithThreshold
IsDynamicSemistochastic
Rimu.StochasticStyles.IsStochastic2Pop
Rimu.StochasticStyles.IsExplosive
```

## The `StochasticStyle` interface

```@docs
step_stats
fciqmc_col!
update_dvec!
CompressionStrategy
compress!
default_style
```

## Spawning strategies and convenience functions

The following functions and types are unexported, but are useful when defining new styles.

```@docs
Rimu.StochasticStyles.diagonal_step!
Rimu.StochasticStyles.projected_deposit!
Rimu.StochasticStyles.SpawningStrategy
Rimu.StochasticStyles.Exact
Rimu.StochasticStyles.SingleSpawn
Rimu.StochasticStyles.WithReplacement
Rimu.StochasticStyles.WithoutReplacement
Rimu.StochasticStyles.Bernoulli
Rimu.StochasticStyles.DynamicSemistochastic
Rimu.StochasticStyles.spawn!
```
