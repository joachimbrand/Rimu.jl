"""
    projected_deposit!(w, add, val, parent, threshold=0)

Like [`deposit!`](@ref), but performs threshold projection before spawning. If `eltype(w)`
is an `Integer`, values are stochastically rounded.

Returns the value deposited.
"""
@inline function projected_deposit!(w, add, val, parent, threshold=0)
    return projected_deposit!(valtype(w), w, add, val, parent, threshold)
end
# Non-integer
@inline function projected_deposit!(
    ::Type{T}, w, add, value, parent, threshold
) where {T<:Real}
    thresh = T(threshold)
    val = T(value)
    absval = abs(val)
    if absval < thresh
        if rand() < absval / thresh
            val = sign(val) * thresh
        else
            val = zero(T)
        end
    end
    if !iszero(val)
        deposit!(w, add, val, parent)
    end

    return val
end
# Round to integer
@inline function projected_deposit!(
    ::Type{T}, w, add, val, parent, threshold=0
) where {T<:Integer}
    if !iszero(threshold)
        throw(ArgumentError("Thresholding not supported for integer spawns"))
    end

    new_val = T(sign(val)) * floor(T, abs(val) + rand())
    if !iszero(new_val)
        deposit!(w, add, new_val, parent)
    end
    return new_val
end
# Complex/Int
@inline function projected_deposit!(
    ::Type{T}, w, add, val, parent, threshold=0
) where {I<:Integer,T<:Complex{I}}
    if !iszero(threshold)
        throw(ArgumentError("Thresholding not supported for integer spawns"))
    end

    val_re, val_im = reim(val)

    new_val_re = I(sign(val_re)) * floor(I, abs(val_re) + rand())
    new_val_im = I(sign(val_im)) * floor(I, abs(val_im) + rand())
    new_val = new_val_re + im * new_val_im

    if !iszero(new_val)
        deposit!(w, add, new_val, parent)
    end
    return new_val_re + im * new_val_im
end

"""
    diagonal_step!(w, op, add, val, threshold=0) -> (clones, deaths, zombies)

Perform diagonal step on a walker `add => val`. Optional argument `threshold` sets the
projection threshold. If `eltype(w)` is an `Integer`, the `val` is rounded to the nearest
integer stochastically.
"""
@inline function diagonal_step!(w, op, add, val, threshold=0)
    new_val = diagonal_element(op, add) * val
    res = projected_deposit!(w, add, new_val, add => val, threshold)
    return clones_deaths_zombies(res, typeof(res)(val))
end

@inline function clones_deaths_zombies(res::T, val::T) where {T<:Real}
    clones = deaths = zombies = zero(T)
    if res > val
        # walker number increased
        clones = abs(res - val)
    elseif sign(res) ≠ sign(val)
        # walker number decreased so much that sign changed
        deaths = abs(val)
        zombies = abs(res)
    else
        # walker number decreased, but not too much
        deaths = abs(res - val)
    end
    return (clones, deaths, zombies)
end
@inline function clones_deaths_zombies(res::Complex, val::Complex)
    res_re, res_im = reim(res)
    val_re, val_im = reim(val)

    clones_re, deaths_re, zombies_re = clones_deaths_zombies(res_re, val_re)
    clones_im, deaths_im, zombies_im = clones_deaths_zombies(res_im, val_im)
    clones = clones_re + im * clones_im
    deaths = deaths_re + im * deaths_im
    zombies = zombies_re + im * zombies_im

    return (clones, deaths, zombies)
end

"""
    SpawningStrategy

A `SpawningStrategy` is used to control how spawns (multiplies with off-diagonal part of the
column vector) are performed and can be passed to some of the [`StochasticStyle`](@ref)s as
keyword arguments.

The following concrete implementations are provided:

* [`Exact`](@ref): Perform exact spawns. Used by [`IsDeterministic`](@ref).

* [`WithReplacement`](@ref): The default stochastic spawning strategy. Spawns are chosen
  with replacement.

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
    spawn!(s::SpawningStrategy, w, op::AbstractHamiltonian, add, val, boost)
    spawn!(s::SpawningStrategy, w, offdiags::AbstractOffdiagonals, add, val, boost)

Perform stochastic spawns to `w` from address `add` with `val` walkers. `val * boost`
controls the number of spawns performed.

This function should be overloaded in the second form, with `offdiags` as an argument.

See [`SpawningStrategy`](@ref).
"""
@inline function spawn!(s::SpawningStrategy, w, op::AbstractOperator, add, val, boost=1)
    return spawn!(s, w, offdiagonals(op, add), add, val, boost)
end

"""
    Exact(threshold=0.0) <: SpawningStrategy

Perform an exact spawning step.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct Exact{T} <: SpawningStrategy
    threshold::T

    Exact(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::Exact, w, offdiags::AbstractVector, add, val, boost=1)
    T = valtype(w)
    spawns = sum(offdiags; init=zero(T)) do (new_add, mat_elem)
        abs(projected_deposit!(
            w, new_add, val * mat_elem, add => val, s.threshold
        ))
    end
    return (length(offdiags), spawns)
end

"""
    SingleSpawn(threshold=0.0) <: SpawningStrategy

Perform a single spawn. Useful as a building block for other stochastic styles.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts (always 1)
and the number of spawns.
"""
struct SingleSpawn{T} <: SpawningStrategy
    threshold::T

    SingleSpawn(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::SingleSpawn, w, offdiags::AbstractVector, add, val, boost=1)
    if iszero(val)
        return (1, zero(valtype(w)))
    else
        new_add, prob, mat_elem = random_offdiagonal(offdiags)
        new_val = val * mat_elem / prob
        spawns = abs(projected_deposit!(w, new_add, new_val, add => val, s.threshold))
        return (1, spawns)
    end
end

"""
    WithReplacement(threshold=0.0) <: SpawningStrategy

[`SpawningStrategy`](@ref) where spawn targets are sampled with replacement. This is the
default spawning strategy for most of the [`StochasticStyle`](@ref)s.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct WithReplacement{T} <: SpawningStrategy
    threshold::T

    WithReplacement(threshold::T=0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::WithReplacement, w, offdiags::AbstractVector, add, val, boost=1)
    spawns = zero(valtype(w))
    num_attempts = max(floor(Int, abs(val) * boost), 1)
    magnitude = val / num_attempts

    for _ in 1:num_attempts
        new_add, prob, mat_elem = random_offdiagonal(offdiags)
        new_val = mat_elem * magnitude / prob
        spawns += abs(projected_deposit!(w, new_add, new_val, add => val, s.threshold))
    end
    return (num_attempts, spawns)
end

"""
    WithoutReplacement(threshold=0.0) <: SpawningStrategy

[`SpawningStrategy`](@ref) where spawn targets are sampled without replacement. This
strategy needs to allocate a temporary array during spawning, which makes it significantly
less efficient than [`WithReplacement`](@ref).

If the number of spawn attempts is greater than the number of offdiagonals, this functions
like [`Exact`](@ref), but is less efficient. For best performance, this strategy is to be
used as a substrategy of [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct WithoutReplacement{T} <: SpawningStrategy
    threshold::T

    WithoutReplacement(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::WithoutReplacement, w, offdiags::AbstractVector, add, val, boost=1)
    spawns = zero(valtype(w))
    num_attempts = max(floor(Int, abs(val) * boost), 1)

    if abs(num_attempts) ≤ 1
        spawn!(SingleSpawn(s.threshold), w, offdiags, add, val)
    else
        magnitude = val / num_attempts

        num_offdiags = length(offdiags)
        prob = 1 / num_offdiags

        for i in sample(1:num_offdiags, num_attempts; replace=false)
            new_add, mat_elem = offdiags[i]
            new_val = mat_elem * magnitude / prob
            spawns += abs(projected_deposit!(w, new_add, new_val, add => val, s.threshold))
        end
    end
    return (num_attempts, spawns)
end

"""
    Bernoulli(threshold=0.0) <: SpawningStrategy

Perform Bernoulli sampling. A spawn is attempted on each offdiagonal element with a
probability that results in an expected number of spawns equal to the number of walkers on
the spawning configuration. This is significantly less efficient than
[`WithReplacement`](@ref).

If the number of spawn attempts is greater than the number of offdiagonals, this functions
like [`Exact`](@ref), but is less efficient. For best performance, this strategy is to be
used as a substrategy of [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct Bernoulli{T} <: SpawningStrategy
    threshold::T

    Bernoulli(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::Bernoulli, w, offdiags::AbstractVector, add, val, boost=1)
    spawns = zero(valtype(w))
    # General case.
    num_offdiags = length(offdiags)
    prob = abs(val) * boost / num_offdiags
    num_attempts = 0
    for i in 1:num_offdiags
        if rand() < prob
            new_add, mat_elem = offdiags[i]
            new_val = mat_elem / prob * val
            spawns += abs(projected_deposit!(w, new_add, new_val, add => val, s.threshold))
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
  threshold is affected by the `boost` argument to [`spawn!`](@ref).

* `abs_threshold = Inf`: When deciding on whether to perform an exact spawn,
  `min(abs_threshold, num_offdiagonals)` is used. This threshold is not affected by
  the `boost` argument to [`spawn!`](@ref).

See e.g. [`WithoutReplacement`](@ref) for a description of the `strat.threshold` parameter.

[`spawn!`](@ref) with this strategy returns the numbers of exact and inexact spawns, the
number of spawn attempts and the number of spawns.
"""
Base.@kwdef struct DynamicSemistochastic{T,S<:SpawningStrategy} <: SpawningStrategy
    strat::S = WithReplacement()
    rel_threshold::T = 1.0
    abs_threshold::T = Inf
end

@inline function spawn!(s::DynamicSemistochastic, w, offdiags::AbstractVector, add, val, boost)
    # assumes that s.strat.threshold is defined
    # special-case substrategies that don't fit the pattern?
    thresh = min(s.abs_threshold, length(offdiags))
    amount = boost * abs(val) * s.rel_threshold
    if amount ≥ thresh
        # Exact multiplication.
        attempts, spawns = spawn!(Exact(s.strat.threshold), w, offdiags, add, val)
        return (1, 0, attempts, spawns)
    else
        # Regular spawns.
        attempts, spawns = spawn!(s.strat, w, offdiags, add, val, boost)
        return (0, 1, attempts, spawns)
    end
end

# bypass branching code for Exact() sub-strategy
@inline function spawn!(
    s::DynamicSemistochastic{<:Any,<:Exact}, w, od::AbstractVector, args...
)
    return (1, 0, spawn!(s.strat, w, od, args...)...)
end
