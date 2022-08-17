"""
    IsStochasticInteger{T=Int}() <: StochasticStyle{T}

FCIQMC algorithm with integer walkers as in
[Booth *et al.* (2009)](https://doi.org/10.1063/1.3193710). During the vector matrix product
each individual diagonal and spawning step is rounded stochastically to a nearby integer
value.

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticInteger{T<:Integer} <: StochasticStyle{T} end
IsStochasticInteger() =  IsStochasticInteger{Int}()

function step_stats(::IsStochasticInteger{T}) where {T}
    z = zero(T)
    return (
        (:spawn_attempts, :spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0, z, z, z, z, z),
    )
end
function fciqmc_col!(
    ::IsStochasticInteger, w, ham, add, num::Real, shift, dτ
)
    clones, deaths, zombies, ann_diag = diagonal_step!(w, ham, add, num, dτ, shift, 0, true)
    attempts, spawns, ann_offdiag = spawn!(WithReplacement(), w, ham, add, num, dτ)
    return (attempts, spawns, deaths, clones, zombies, ann_diag + ann_offdiag)
end

"""
    IsStochastic2Pop{T=Complex{Int}}() <: StochasticStyle{T}

Stochastic propagation with complex walker numbers representing two populations of integer
walkers.

When using this style, make sure to set a complex target number walkers in the
[`ShiftStrategy`](@ref)!

This style is experimental.

See also [`StochasticStyle`](@ref).
"""
struct IsStochastic2Pop{T<:Complex{<:Integer}} <: StochasticStyle{T} end
IsStochastic2Pop() = IsStochastic2Pop{Complex{Int}}()

function step_stats(::IsStochastic2Pop{T}) where {T}
    z = zero(T)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(z, z, z, z, z)
    )
end
function fciqmc_col!(::IsStochastic2Pop, w, ham, add, val, shift, dτ)
    offdiags = offdiagonals(ham, add)
    spawns = deaths = clones = zombies = ann_o = ann_d = 0 + 0im
    # off-diagonal real.
    s, a = spawn!(WithReplacement(), w, offdiags, add, real(val), dτ)
    spawns += s; ann_o += a
    # off-diagonal complex: complex dτ ensures spawning to the correct population.
    s, a = spawn!(WithReplacement(), w, offdiags, add, imag(val), dτ * im)
    spawns += s; ann_o += a

    clones, deaths, zombies, ann_d = diagonal_step!(w, ham, add, val, dτ, shift, 0, true)

    return (spawns, deaths, clones, zombies, ann_o + ann_d)
end

"""
    IsDeterministic{T=Float64}(compression=NoCompression()) <: StochasticStyle{T}

Propagate with deterministic vector matrix multiplications. Stochastic compression of the
resultant vector (after annihilations) can be triggered by setting the optional
`compression` argument to a relevant [`CompressionStrategy`](@ref).

See also [`StochasticStyle`](@ref).
"""
struct IsDeterministic{T<:AbstractFloat,C<:CompressionStrategy} <: StochasticStyle{T}
    compression::C
end
function IsDeterministic{T}(compression::C=NoCompression()) where {T,C}
    return IsDeterministic{T,C}(compression)
end
IsDeterministic(args...) = IsDeterministic{Float64}(args...)

CompressionStrategy(s::IsDeterministic) = s.compression

function Base.show(io::IO, s::IsDeterministic{T}) where {T}
    if s.compression isa NoCompression
        print(io, "IsDeterministic{$T}()")
    else
        print(io, "IsDeterministic{$T}($(s.compression))")
    end
end

function step_stats(::IsDeterministic)
    return (:exact_steps,), MultiScalar(0,)
end
function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    for i in axes(ham, 1) # iterate through off-diagonal rows of `ham`
        i == add && continue
        deposit!(w, i, -dτ * ham[i, add] * num, add => num)
    end
    deposit!(w, add, (1 + dτ * (shift - ham[add, add])) * num, add => num) # diagonal
    return (1,)
end
function fciqmc_col!(::IsDeterministic, w, ham, add, val, shift, dτ)
    diagonal_step!(w, ham, add, val, dτ, shift)
    spawn!(Exact(), w, ham, add, val, dτ)
    return (1,)
end

"""
    IsStochasticWithThreshold{T=Float64}(threshold=1.0) <: StochasticStyle{T}

Stochastic propagation with floating point walker numbers. During the vector matrix product
each individual diagonal and spawning result is rounded stochastically if smaller than
`threshold` (before annihilations).

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticWithThreshold{T<:AbstractFloat} <: StochasticStyle{T}
    threshold::T
end
IsStochasticWithThreshold(args...) = IsStochasticWithThreshold{Float64}(args...)
IsStochasticWithThreshold{T}(t=1.0) where {T} = IsStochasticWithThreshold{T}(T(t))

function step_stats(::IsStochasticWithThreshold{T}) where {T}
    z = zero(T)
    return (
        (:spawn_attempts, :spawns),
        MultiScalar(0, z)
    )
end
function fciqmc_col!(s::IsStochasticWithThreshold, w, ham, add, val, shift, dτ)
    diagonal_step!(w, ham, add, val, dτ, shift)
    attempts, spawns, _ = spawn!(WithReplacement(s.threshold), w, ham, add, val, dτ)
    return (attempts, spawns)
end

"""
    IsDynamicSemistochastic{T=Float64}(; kwargs...) <: StochasticStyle{T}

QMC propagation with floating-point walker numbers and reduced noise. All possible spawns
(offdiagonal elements in vector-matrix multiplication)
are performed deterministically when number of walkers in a configuration is high, as
controlled by the `rel_threshold` and `abs_threshold` keywords. Stochastic selection of
spawns is controlled  by the `spawning` keyword.

By default, a stochastic vector compression is applied after annihilations are completed.
This behaviour can be changed to on-the-fly projection (as in [`IsStochasticInteger`](@ref)
or [`IsStochasticWithThreshold`](@ref)) by setting a threshold to `spawning` and
setting `compression` to [`NoCompression`](@ref).

## Parameters:

* `rel_threshold = 1.0`: If the walker number on a configuration times this threshold
  is greater than the number of offdiagonals, spawning is done deterministically. Should be
  set to 1 or more for best performance.

* `abs_threshold = Inf`: If the walker number on a configuration is greater than this value,
  spawning is done deterministically. Can be set to e.g.
  `abs_threshold = 0.1 * target_walkers`.

* `proj_threshold = 1.0`: Values below this number are stochastically projected to this
  value or zero. See also [`ThresholdCompression`](@ref).

* `spawning = WithReplacement()`: [`SpawningStrategy`](@ref) to use for the non-exact
  spawns. See [`WithReplacement`](@ref) for behaviour and possible arguments.

* `compression = ThresholdCompression(proj_threshold)`: [`CompressionStrategy`](@ref) used
  to compress the vector after a step. Overrides `proj_threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsDynamicSemistochastic{
    T<:AbstractFloat,C<:CompressionStrategy,S<:DynamicSemistochastic
} <: StochasticStyle{T}
    compression::C
    spawning::S
end
function IsDynamicSemistochastic{T}(
    ;
    proj_threshold=1.0, strength=1.0, rel_threshold=1.0, abs_threshold=Inf,
    compression=ThresholdCompression(proj_threshold),
    spawning=WithReplacement(),
) where {T}
    s = DynamicSemistochastic(spawning, rel_threshold, abs_threshold)
    return IsDynamicSemistochastic{T,typeof(compression),typeof(s)}(compression, s)
end
IsDynamicSemistochastic(; kwargs...) = IsDynamicSemistochastic{Float64}(; kwargs...)

function Base.show(io::IO, s::IsDynamicSemistochastic{T,C,S}) where {T,C,S}
    print(io, "IsDynamicSemistochastic{$T,", nameof(C), ",", nameof(S), "}()")
end

CompressionStrategy(s::IsDynamicSemistochastic) = s.compression

function step_stats(::IsDynamicSemistochastic{T}) where {T}
    z = zero(T)
    return (
        (:exact_steps, :inexact_steps, :spawn_attempts, :spawns),
        MultiScalar(0, 0, 0, z),
    )
end
function fciqmc_col!(s::IsDynamicSemistochastic, w, ham, add, val, shift, dτ)
    diagonal_step!(w, ham, add, val, dτ, shift)
    exact, inexact, attempts, spawns, _ = spawn!(s.spawning, w, ham, add, val, dτ)
    return (exact, inexact, attempts, spawns)
end

"""
    IsExplosive{T=Float64}(;
        splatter_factor,
        explosion_threshold,
        compression
    ) <: StochasticStyle{T}

QMC propagation with explosive walkers. Walkers with small walker numbers do not perform the
standard death/spawning steps. Instead, a walker will either die completely and spawn with a
greater magnitude (refered to as explosion below), or stay unchanged and not spawn. The
probabilty of exploding is controlled by the shift and `dτ`.

Walkers with high walker numbers spawn as if [`IsDynamicSemistochastic`](@ref) was used.

Like [`IsDynamicSemistochastic`](@ref), the vector is compressed after all spawning is
performed.

This style is experimental.

## Parameters

* `splatter_factor = 1.0`: The spawning strength to use with exploded walkers.

* `explosion_threshold = 1.0`: Entries smaller or equal than this value will attempt to
  explode.

* `proj_threshold = 1.0`: Threshold to use in vector compression. See
  [`ThresholdCompression`](@ref).

* `compression = ThresholdCompression(proj_threshold)`: [`CompressionStrategy`](@ref) to use
  to compress the vector. Overrides `proj_threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsExplosive{T<:AbstractFloat,C<:CompressionStrategy} <: StochasticStyle{T}
    splatter_factor::T
    explosion_threshold::T
    compression::C
end
function IsExplosive{T}(
    ;
    splatter_factor=one(T),
    explosion_threshold=one(T),
    proj_threshold=one(T),
    compression=ThresholdCompression(proj_threshold),
) where {T}
    return IsExplosive(T(splatter_factor), T(explosion_threshold), compression)
end
IsExplosive(; kwargs...) = IsExplosive{Float64}(; kwargs...)

CompressionStrategy(s::IsExplosive) = s.compression

function step_stats(::IsExplosive{T}) where {T}
    z = zero(T)
    return (
        (:explosions, :ticks, :normal_steps, :spawn_attempts,
         :explosive_spawns, :normal_spawns,
         ),
        MultiScalar(0, 0, 0, 0, z, z),
    )
end
function fciqmc_col!(s::IsExplosive{T}, w, ham, add, val, shift, dτ) where {T}
    explosions = normal_steps = ticks = 0
    explosive_spawns = normal_spawns = zero(T)
    clones = deaths = zombies = zero(T)
    attempts = 0

    pd = dτ * (diagonal_element(ham, add) - shift)
    if abs(val) ≤ s.explosion_threshold && 0 ≤ pd < 1
        if cRand() < pd
            _, _, attempts, explosive_spawns, _ = spawn!(
                DynamicSemistochastic(strat=WithReplacement(zero(T), s.splatter_factor)),
                w, ham, add, val / pd, dτ,
            )
            explosions = 1
        else
            deposit!(w, add, val, add => val)
            ticks = 1
        end
    else
        _ = diagonal_step!(w, ham, add, val, dτ, shift)
        _, _, attempts, normal_spawns, _ = spawn!(
            DynamicSemistochastic(strat=WithReplacement()), w, ham, add, val, dτ,
        )
        normal_steps = 1
    end

    return (
        explosions, ticks, normal_steps, attempts, explosive_spawns, normal_spawns,
    )
end

default_style(::Type{T}) where {T<:Integer} = IsStochasticInteger{T}()
default_style(::Type{T}) where {T<:AbstractFloat} = IsDeterministic{T}()
default_style(::Type{T}) where {T<:Complex{<:Integer}} = IsStochastic2Pop{T}()
