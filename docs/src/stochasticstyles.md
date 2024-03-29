# Module `StochasticStyles`

```@docs
StochasticStyles
```

## Available `StochasticStyle`s

```@docs
StyleUnknown
```
```@autodocs
Modules = [StochasticStyles]
Pages = ["styles.jl"]
```

## The `StochasticStyle` interface

```@docs
StochasticStyle
step_stats
apply_column!
default_style
```

## Compression strategies

```@docs
CompressionStrategy
NoCompression
StochasticStyles.ThresholdCompression
compress!
```

## Spawning strategies and convenience functions

The following functions and types are unexported, but are useful when defining new styles.

```@autodocs
Modules = [StochasticStyles]
Pages = ["spawning.jl"]
Order = [:function,:method,:type]
```

## Index
```@index
Pages   = ["stochasticstyles.md"]
```
