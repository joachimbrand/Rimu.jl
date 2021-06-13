
## Module `Rimu/StatsTools`

The  module `Rimu/StatsTools` contains helper function for analysis and post
processing of Monte Carlo data.

### Usage

The module is not exported by default. In oder to use its functions without
explicitly specifying the submodule, import the module with

```julia
using Rimu, Rimu.StatsTools
```

### Blocking analysis

For blocking analysis of a single time series use [`blocking_analysis`](@ref),
or [`mean_and_se`](@ref).

For blocking analysis of a couple of time series where the ration of means is
the quantity of interest use [`ratio_of_means`](@ref).

#### Exported
```@autodocs
Modules = [StatsTools]
Pages = ["StatsTools.jl", "blocking.jl", "ratio_of_means.jl", "convenience.jl",
  "variances.jl", "growth_witness.jl", "reweighting.jl"
]
Private = false
```

#### Additional docstrings
```@autodocs
Modules = [StatsTools]
Pages = ["StatsTools.jl", "blocking.jl", "ratio_of_means.jl", "convenience.jl",
  "variances.jl", "growth_witness.jl", "reweighting.jl"
]
Public = false
```

#### Index
```@index
Pages   = ["statstools.md"]
```
