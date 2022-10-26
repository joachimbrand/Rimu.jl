"""
    projected_deposit!(w, add, val, parent, threshold=0)

Like [`deposit!`](@ref), but performs threshold projection before spawning. If `eltype(w)`
is an `Integer`, values are stochastically rounded.

Returns the value deposited and the number.
"""
@inline function projected_deposit!(w, add, val, parent, threshold=0)
    return projected_deposit!(valtype(w), w, add, val, parent, threshold)
end
# Non-integer
@inline function projected_deposit!(::Type{T}, w, add, value, parent, thresh) where {T}
    # ensure type stability.
    threshold = T(thresh)
    val = T(value)

    absval = abs(val)
    if absval < threshold
        if rand() < abs(val) / threshold
            val = sign(val) * threshold
        else
            val = zero(val)
        end
    end
    if !iszero(val)
        deposit!(w, add, val, parent)
    end

    return abs(val)
end
# Round to integer
@inline function projected_deposit!(::Type{T}, w, add, val, parent, _) where {T<:Integer}
    intval = T(sign(val)) * floor(T, abs(val) + rand())
    if !iszero(intval)
        deposit!(w, add, intval, parent)
    end
    return abs(intval)
end
# Complex/Int
@inline function projected_deposit!(
    ::Type{T}, w, add, val, parent, _
) where {I<:Integer,T<:Complex{I}}
    r_val, i_val = reim(val)

    r_intval = I(sign(r_val)) * floor(I, abs(r_val) + rand())
    i_intval = I(sign(i_val)) * floor(I, abs(i_val) + rand())
    intval = r_intval + im * i_intval

    if !iszero(intval)
        deposit!(w, add, intval, parent)
    end
    return abs(r_intval) + im * abs(i_intval)
end

"""
    diagonal_step!(w, ham, add, val, dτ, shift, threshold=0) -> (clones, deaths, zombies)

Perform diagonal step on a walker `add => val`. Optional argument `threshold` sets the
projection threshold. If `eltype(w)` is an `Integer`, the `val` is rounded to the nearest
integer stochastically.
"""
@inline function diagonal_step!(w, ham, add, val, dτ, shift, threshold=0)
    pd = dτ * (diagonal_element(ham, add) - shift)
    new_val = (1 - pd) * val
    res = projected_deposit!(w, add, new_val, add => val, threshold)

    return clones_deaths_zombies(pd, res, val)
end

@inline function clones_deaths_zombies(pd::Real, res::Real, val::Real)
    clones = deaths = zombies = zero(res)
    if pd < 0
        clones = abs(res - val)
    elseif pd < 1
        deaths = abs(res - val)
    else
        deaths = abs(val)
        zombies = abs(res)
    end
    return (clones, deaths, zombies)
end
@inline function clones_deaths_zombies(pd::Complex, res::Complex, val::Complex)
    clones = deaths = zombies = zero(res)
    pd_re, pd_im = reim(pd)
    res_re, res_im = reim(res)
    val_re, val_im = reim(val)
    if pd_re < 0
        clones += abs(res_re - val_re)
    elseif pd_re < 1
        deaths += abs(res_re - val_re)
    else
        deaths += abs(val_re)
        zombies += abs(res_re)
    end
    if pd_im < 0
        clones += abs(res_im - val_im) * im
    elseif pd_im < 1
        deaths += abs(res_im - val_im) * im
    else
        deaths += abs(val_im) * im
        zombies += abs(res_im) * im
    end
    return (clones, deaths, zombies)
end

"""
    SpawningStrategy

A `SpawningStrategy` is used to control how spawns (multiplies with off-diagonal part of the
column vector) are performed and can be passed to some of the [`StochasticStyle`](@ref)s as
keyword arguments.

The following concrete implementations are provided:

* [`Exact`](@ref): Perform exact spawns. Used by [`IsDeterministic`](@ref).

* [`WithReplacement`](@ref): The default stochastic spawning strategy. Spawns are chosen with
  replacement.

* [`DynamicSemistochastic`](@ref): Behave like [`Exact`](@ref) when the number of spawns
  performed is high, and like a different substrategy otherwise. Used by
  [`IsDynamicSemistochastic`](@ref).

* [`SingleSpawn`](@ref): Perform a single spawn only. Used as a building block for other
  strategies.

* [`WithoutReplacement`](@ref): Similar to [`WithReplacement`](@ref), but ensures each spawn
  is only performed once. Only to be used as a substrategy of
  [`DynamicSemistochastic`](@ref).

* [`Bernoulli`](@ref): Each spawn is attempted with a fixed probability. Only to be used as
  a substrategy of [`DynamicSemistochastic`](@ref).


## Interface

In order to implement a new `SpawningStrategy`, define a method for [`spawn!`](@ref).
"""
abstract type SpawningStrategy end

"""
    spawn!(s::SpawningStrategy, w, ham::AbstractHamiltonian, add, val, dτ)
    spawn!(s::SpawningStrategy, w, offdiags::AbstractOffdiagonals, add, val, dτ)

Perform stochastic spawns to `w` from address `add` with `val` walkers. `dτ` is a factor
multiplied to every spawns, while `val` also controls the number of spawns performed.

This function should be overloaded in the second form, with `offdiags` as an argument.

See [`SpawningStrategy`](@ref).
"""
@inline function spawn!(s::SpawningStrategy, w, ham, add, val, dτ)
    return spawn!(s, w, offdiagonals(ham, add), add, val, dτ)
end

"""
    Exact(threshold=0.0) <: SpawningStrategy

Perform an exact spawning step.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawns.
"""
struct Exact{T} <: SpawningStrategy
    threshold::T
end
Exact() = Exact(0.0)

@inline function spawn!(s::Exact, w, offdiags::AbstractVector, add, val, dτ)
    T = valtype(w)
    spawns = zero(valtype(w))
    factor = -dτ * val
    for (new_add, mat_elem) in offdiags
        spw = projected_deposit!(
            w, new_add, factor * mat_elem, add => val, s.threshold
        )
        spawns += spw
    end
    return (length(offdiags), spawns)
end

"""
    SingleSpawn(threshold=0.0) <: SpawningStrategy

Perform a single spawn. Useful as a building block for other stochastic styles.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawns.

"""
struct SingleSpawn{T} <: SpawningStrategy
    threshold::T
    strength::T
end
SingleSpawn(threshold=0.0) = SingleSpawn(threshold, zero(threshold))

@inline function spawn!(s::SingleSpawn, w, offdiags::AbstractVector, add, val, dτ)
    if iszero(val)
        z = zero(valtype(w))
        return (z, z)
    else
        new_add, prob, mat_elem = random_offdiagonal(offdiags)
        new_val = -val * mat_elem * dτ / prob
        spw = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
        return 1, spw
    end
end

"""
    WithReplacement(threshold=0.0, strength=1.0) <: SpawningStrategy

[`SpawningStrategy`](@ref) where spawn targets are sampled with replacement. This is the
default spawning strategy for most of the [`StochasticStyle`](@ref)s.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.
* `strength` sets the number of spawns to perform, e.g. if `val=5` and `strength=2`, 10
  spawns will be performed.

[`spawn!`](@ref) with this strategy returns the number of spawns.
"""
struct WithReplacement{T} <: SpawningStrategy
    threshold::T
    strength::T
end
function WithReplacement(threshold=0.0, strength=one(threshold))
    t, s = promote(threshold, strength)
    return WithReplacement{typeof(t)}(t, s)
end

@inline function spawn!(s::WithReplacement, w, offdiags::AbstractVector, add, val, dτ)
    spawns = zero(valtype(w))
    num_attempts = max(floor(Int, abs(val) * s.strength), 1)
    magnitude = val / num_attempts
    factor = magnitude * dτ

    for _ in 1:num_attempts
        new_add, prob, mat_elem = random_offdiagonal(offdiags)
        new_val = -mat_elem * factor / prob
        spw = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
        spawns += spw
    end
    return (num_attempts, spawns)
end

"""
    WithoutReplacement(threshold=0.0, strength=1.0) <: SpawningStrategy

[`SpawningStrategy`](@ref) where spawn targets are sampled without replacement. This
strategy needs to allocate a temporary array during spawning, which makes is significantly
less efficient than [`WithReplacement`](@ref).

If the number of spawn attempts is greater than the number of offdiagonals, this functions
like [`Exact`](@ref), but is less efficient. For best performance, this strategy is to be
used as a substrategy of [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.
* `strength` sets the number of spawns to perform, e.g. if `val=5` and `strength=2`, 10
  spawns will be performed.

[`spawn!`](@ref) with this strategy returns the number of spawns.
"""
struct WithoutReplacement{T} <: SpawningStrategy
    threshold::T
    strength::T
end
function WithoutReplacement(threshold=0.0, strength=one(threshold))
    t, s = promote(threshold, strength)
    return WithoutReplacement{typeof(t)}(t, s)
end

@inline function spawn!(s::WithoutReplacement, w, offdiags::AbstractVector, add, val, dτ)
    spawns = zero(valtype(w))
    num_attempts = max(floor(Int, abs(val) * s.strength), 1)

    if abs(num_attempts) ≤ 1
        spawn!(SingleSpawn(s.threshold), w, offdiags, add, val, dτ)
    else
        magnitude = val / num_attempts

        num_offdiags = length(offdiags)
        prob = 1 / num_offdiags

        for i in sample(1:num_offdiags, num_attempts; replace=false)
            new_add, mat_elem = offdiags[i]
            new_val = -mat_elem * magnitude / prob * dτ
            spw = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
            spawns += spw
        end
    end
    return (num_attempts, spawns)
end

"""
    Bernoulli(threshold=0.0, strength=1.0) <: SpawningStrategy

Perform Bernoulli sampling. A spawn is attempted on each offdiagonal element with a
probability that results in an expected number of spawns equal to the number of walkers on
the spawning configuration. This is significantly less efficient than
[`WithReplacement`](@ref).

If the number of spawn attempts is greater than the number of offdiagonals, this functions
like [`Exact`](@ref), but is less efficient. For best performance, this strategy is to be
used as a substrategy of [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold.
* `strength` sets the number of spawns to perform, e.g. if `val=5` and `strength=2`, 10
  spawns will be performed on average.

[`spawn!`](@ref) with this strategy returns the number of spawns.
"""
struct Bernoulli{T} <: SpawningStrategy
    threshold::T
    strength::T
end

function Bernoulli(threshold=0.0, strength=one(threshold))
    t, s = promote(threshold, strength)
    return Bernoulli{typeof(t)}(t, s)
end

@inline function spawn!(s::Bernoulli, w, offdiags::AbstractVector, add, val, dτ)
    spawns = zero(valtype(w))
    # General case.
    num_offdiags = length(offdiags)
    prob = abs(val) * s.strength / num_offdiags
    num_attempts = 0
    for i in 1:num_offdiags
        if rand() > prob
            new_add, mat_elem = offdiags[i]
            new_val = -mat_elem / prob * dτ * val
            spw = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
            spawns += spw
            num_attempts += 1
        end
    end
    return (num_attempts, spawns)
end


"""
    DynamicSemistochastic(; strat, rel_threshold, abs_threshold) <: SpawningStrategy

[`SpawningStrategy`](@ref) that behaves like `strat` when the number of walkers is low, but
performs exact steps when it is high. What "high" means is controlled by the two thresholds
described below.

## Parameters

* `strat = WithReplacement()`: a [`SpawningStrategy`](@ref) to use when the multiplication
  is not performed exactly. If the `strat` has a `threshold` different from zero, all spawns
  will be projected to that threshold.

* `rel_threshold = 1.0`: When deciding on whether to perform an exact spawn, this value is
  multiplied to the number of walkers. Should be set to 1 or more for best performance. This
  threshold is affected by `strat.strength`.

* `abs_threshold = Inf`: When deciding on whether to perform an exact spawn,
  `min(abs_threshold, num_offdiagonals)` is used. This threshold is affected by
  `strat.strength`.

See e.g. [`WithoutReplacement`](@ref) for a description of the `strat.threshold` and
`strat.strength` parameters.

[`spawn!`](@ref) with this strategy returns the numbers of exact and inexact spawns, and the
number of spawns.
"""
Base.@kwdef struct DynamicSemistochastic{T,S<:SpawningStrategy} <: SpawningStrategy
    strat::S = WithReplacement()
    rel_threshold::T = 1.0
    abs_threshold::T = Inf
end

@inline function spawn!(s::DynamicSemistochastic, w, offdiags::AbstractVector, add, val, dτ)
    # assumes that s.strat.strength and s.strat.threshold are defined
    # special-case substrategies that don't fit the pattern!
    thresh = min(s.abs_threshold, length(offdiags))
    amount = s.strat.strength * abs(val) * s.rel_threshold
    if amount ≥ thresh
        # Exact multiplication.
        attempts, spawns = spawn!(
            Exact(s.strat.threshold), w, offdiags, add, val, dτ
        )
        return (1, 0, attempts, spawns)
    else
        # Regular spawns.
        attempts, spawns = spawn!(s.strat, w, offdiags, add, val, dτ)
        return (0, 1, attempts, spawns)
    end
end

# bypass branching code for Exact() sub-strategy
@inline function spawn!(s::DynamicSemistochastic{<:Any, <:Exact}, args...)
    return (1, 0, spawn!(s.strat, args...)...)
end
