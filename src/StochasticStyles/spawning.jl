"""
    projected_deposit!(w, add, val, parent, threshold=0)

Like [`deposit!`](@ref), but performs threshold projection before spawning. If `eltype(w)` is
an `Integer`, values are stochastically rounded.

Returns the value deposited and the number of annihilations.
"""
function projected_deposit!(w, add, val, parent, threshold=0)
    projected_deposit!(valtype(w), w, add, val, parent, threshold)
end
# Non-integer
function projected_deposit!(::Type{T}, w, add, val, parent, threshold) where T
    absval = abs(val)
    if absval < threshold
        if cRand() < abs(val) / threshold
            val = sign(val) * threshold
        else
            val = zero(val)
        end
    end
    annihilations = zero(T)
    if !iszero(val)
        prev = w[add]
        if sign(prev) ≠ sign(val)
            annihilations = min(abs(prev), abs(val))
        end
        deposit!(w, add, val, parent)
    end

    return abs(val), annihilations
end
# Round to integer
function projected_deposit!(::Type{T}, w, add, val, parent, _) where {T<:Integer}
    intval = T(sign(val)) * floor(T, abs(val) + cRand())
    annihilations = zero(T)
    if !iszero(intval)
        prev = w[add]
        if sign(prev) ≠ sign(intval)
            annihilations = min(abs(prev), abs(intval))
        end
        deposit!(w, add, intval, parent)
    end
    return abs(intval), annihilations
end
# TODO complex

"""
    diagonal_step!(w, ham, add, val, dτ, shift, threshold=0)

Perform diagonal step on a walker `add => val`. Optional argument `threshold` sets the
projection threshold. If `eltype(w)` is an `Integer`, the `val` is rounded stochastically.
"""
function diagonal_step!(w, ham, add, val, dτ, shift, threshold=0)
    clones = deaths = zombies = zero(valtype(w))

    pd = dτ * (diagonal_element(ham, add) - shift)
    new_val = (1 - pd) * val
    res, annihilations = projected_deposit!(w, add, new_val, add => val, threshold)
    if pd < 0
        clones = abs(res - val)
    elseif pd < 1
        deaths = abs(res - val)
    else
        deaths = abs(val)
        zombies = abs(res)
    end
    return (clones, deaths, zombies, annihilations)
end

"""
    SpawningStrategy

A `SpawningStrategy` is used to control how spawns are peformed.

The following methods are implemented:

* [`ExactSpawning`](@ref)
* [`SpawningWithReplacement`](@ref)
* [`SpawningWithoutReplacement`](@ref)
* [`BernoulliSpawning`](@ref)
* [`DynamicSemistochasticSpawning`](@ref)
"""
abstract type SpawningStrategy end

"""
    spawn!(w, ham::AbstractHamiltonian, add, val, dτ)
    spawn!(s::SpawningStrategy, w, offdiags::AbstractOffdiagonals, add, val, dτ)

Perform stochastic spawns to `w` from address `add` with `val` walkers. `dτ` is a factor
multiplied to every spawns, while `val` also controls the number of spawns performed.

This function should be overloaded in the second form, with `offdiags` as an argument.
"""
function spawn!(w, ham, add, val, dτ)
    spawn_strat = SpawningStrategy(StochasticStyle(w))
    return spawn!(spawn_strat, w, offdiagonals(ham, add), add, val, dτ)
end
function spawn!(s::SpawningStrategy, w, ham, add, val, dτ)
    return spawn!(s, w, offdiagonals(ham, add), add, val, dτ)
end

"""
    Exact(threshold=0.0) <: SpawningStrategy

Perform an exact spawning step. `threshold` sets the projection threshold.

[`spawn!`](@ref) with this strategy returns the number of spawns and annihilations.
"""
Base.@kwdef struct Exact{T} <: SpawningStrategy
    threshold::T = 0.0
end

step_stats(::Exact, ::Type{T}) where {T} = (:spawns => T, :annihilations => T)

function spawn!(s::Exact, w, offdiags::AbstractVector, add, val, dτ)
    spawns = annihilations = zero(valtype(w))
    factor = -dτ * val
    for (new_add, mat_elem) in offdiags
        spw, ann = projected_deposit!(w, new_add, factor * mat_elem, add => val, s.threshold)
        spawns += spw
        annihilations += ann
    end
    return (spawns, annihilations)
end

"""
    WithReplacement <: SpawningStrategy

Perform semistochastic spawns from a walker `add => val`. If `val` exceeds the number of possible spawns, the spawning is done exactly (see [`Exact`](@ref)).

## Parameters

* `threshold` sets the projection threshold.
* `strength` sets the number of spawns to perform, e.g. if `val=5` and `strength=2`, 10
  spawns will be performed.

[`spawn!`](@ref) with this strategy returns the number of spawns and annihilations.
"""
Base.@kwdef struct WithReplacement{T} <: SpawningStrategy
    threshold::T = 0.0
    strength::T = 1.0
end

step_stats(::WithReplacement, ::Type{T}) where {T} = (:spawns => T, :annihilations => T)

function spawn!(s::WithReplacement, w, offdiags::AbstractVector, add, val, dτ)
    spawns = annihilations = zero(valtype(w))
    num_spawns = max(floor(Int, abs(val) * s.strength), 1)
    magnitude = val / num_spawns

    for _ in 1:num_spawns
        new_add, prob, mat_elem = random_offdiagonal(offdiags)
        new_val = -mat_elem * magnitude / prob * dτ
        spw, ann = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
        spawns += spw
        annihilations += ann
    end
    return (spawns, annihilations)
end


"""
    WithoutReplacement <: SpawningStrategy

Only to be used with [`DynamicSemistochastic`](@ref).

Perform semistochastic spawns from a walker `add => val`. Spawn targets are sampled without
replacement. If `val` exceeds the number of possible spawns, the spawning is done exactly (see [`Exact`](@ref)).

## Parameters

* `threshold` sets the projection threshold.
* `strength` sets the number of spawns to perform, e.g. if `val=5` and `strength=2`, 10
  spawns will be performed.

[`spawn!`](@ref) with this strategy returns the number of spawns and annihilations.
"""
Base.@kwdef struct WithoutReplacement{T} <: SpawningStrategy
    threshold::T = 0.0
    strength::T = 1.0
end

step_stats(::WithoutReplacement, ::Type{T}) where {T} = (:spawns => T, :annihilations => T)

function spawn!(s::WithoutReplacement, w, offdiags::AbstractVector, add, val, dτ)
    spawns = annihilations = zero(valtype(w))
    num_spawns = max(floor(Int, abs(val) * s.strength), 1)
    magnitude = val / num_spawns

    num_offdiags = length(offdiags)
    prob = 1 / num_offdiags

    for i in sample(1:num_offdiags, num_spawns; replace=false)
        new_add, mat_elem = offdiags[i]
        new_val = -mat_elem * magnitude / prob * dτ
        spw, ann = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
        spawns += spw
        annihilations += ann
    end
    return (spawns, annihilations)
end

"""
    Bernoulli <: SpawningStrategy

Only to be used with [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold.
* `strength` sets the number of spawns to perform, e.g. if `val=5` and `strength=2`, 10
  spawns will be performed.

[`spawn!`](@ref) with this strategy returns the number of spawns and annihilations.
"""
Base.@kwdef struct Bernoulli{T} <: SpawningStrategy
    threshold::T = 0.0
    strength::T = 1.0
end

step_stats(::Bernoulli, ::Type{T}) where {T} = (:spawns => T, :annihilations => T)

function spawn!(s::Bernoulli, w, offdiags::AbstractVector, add, val, dτ)
    spawns = annihilations = zero(valtype(w))

    num_offdiags = length(offdiags)
    prob = abs(val) * s.strength / num_offdiags

    for i in 1:num_offdiags
        if cRand() > prob
            new_add, mat_elem = offdiags[i]
            new_val = -mat_elem / prob * dτ * val
            spw, ann = projected_deposit!(w, new_add, new_val, add => val, s.threshold)
            spawns += spw
            annihilations += ann
        end
    end
    return (spawns, annihilations)
end


"""
    DynamicSemistochastic(; start, rel_threshold, abs_threshold) <: SpawningStrategy

[`SpawningStrategy`](@ref) that behaves like `strat` when the number of walkers is low, but
performs exact steps when it is high.

* `strat = WithReplacement()`: a [`SpawningStrategy`](@ref) to use when the
  multiplication is not performed exactly.

* `rel_threshold = 1.0`: When deciding on whether to perform an exact spawn, this value is
  multiplied to the number of walkers. Should be set to 1 or more for best performance.

* `abs_threshold = Inf`: When deciding on whether to perform an exact spawn,
  `min(abs_threshold, num_offdiagonals)` is used.

"""
Base.@kwdef struct DynamicSemistochastic{T,S<:SpawningStrategy} <: SpawningStrategy
    strat::S = WithReplacement()
    rel_threshold::T = 1.0
    abs_threshold::T = Inf
end

step_stats(::DynamicSemistochastic, ::Type{T}) where {T} =
    (
        :exact_steps => Int,
        :inexact_steps => Int,
        :spawns => T,
        :annihilations => T,
    )

function spawn!(s::DynamicSemistochastic, w, offdiags::AbstractVector, add, val, dτ)
    thresh = min(s.abs_threshold, length(offdiags))
    amount = s.strat.strength * abs(val) * s.rel_threshold
    if amount ≥ thresh
        # Exact multiplication.
        spawns, annihilations = spawn!(Exact(s.strat.threshold), w, offdiags, add, val, dτ)
        return (1, 0, spawns, annihilations)
    else
        # Regular spawns.
        spawns, annihilations = spawn!(s.strat, w, offdiags, add, val, dτ)
        return (0, 1, spawns, annihilations)
    end
end
