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
        (:spawn_attempts, :spawns, :deaths, :clones, :zombies),
        MultiScalar(0, z, z, z, z),
    )
end
function apply_column!(::IsStochasticInteger, w, op, add, val::Real, boost=1)
    clones, deaths, zombies = diagonal_step!(w, op, add, val)
    attempts, spawns = spawn!(WithReplacement(), w, op, add, val, boost)
    return (attempts, spawns, deaths, clones, zombies)
end

"""
    IsStochastic2Pop{T=Complex{Int}}() <: StochasticStyle{T}

Stochastic propagation with complex walker numbers representing two populations of integer
walkers.

When using this style, make sure to set a complex number as target walkers in the
[`ShiftStrategy`](@ref Main.ShiftStrategy)!

This style is experimental.

See also [`StochasticStyle`](@ref).
"""
struct IsStochastic2Pop{T<:Complex{<:Integer}} <: StochasticStyle{T} end
IsStochastic2Pop() = IsStochastic2Pop{Complex{Int}}()

function step_stats(::IsStochastic2Pop{T}) where {T}
    z = zero(T)
    return (
        (:spawns, :deaths, :clones, :zombies),
        MultiScalar(z, z, z, z)
    )
end
function apply_column!(::IsStochastic2Pop, w, op, add, val, boost=1)
    offdiags = offdiagonals(op, add)
    spawns = deaths = clones = zombies = 0 + 0im
    # off-diagonal real.
    s, a = spawn!(WithReplacement(), w, offdiags, add, real(val), boost)
    spawns += s
    # off-diagonal complex.
    s, a = spawn!(WithReplacement(), w, offdiags, add, im * imag(val), boost)
    spawns += s

    clones, deaths, zombies = diagonal_step!(w, op, add, val)

    return (spawns, deaths, clones, zombies)
end

const FloatOrComplexFloat = Union{AbstractFloat, Complex{<:AbstractFloat}}

"""
    IsDeterministic{T=Float64}(compression=NoCompression()) <: StochasticStyle{T}

Propagate with deterministic vector matrix multiplications. Stochastic compression of the
resultant vector (after annihilations) can be triggered by setting the optional
`compression` argument to a relevant [`CompressionStrategy`](@ref).

See also [`StochasticStyle`](@ref).
"""
struct IsDeterministic{T<:FloatOrComplexFloat,C<:CompressionStrategy} <: StochasticStyle{T}
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
function apply_column!(::IsDeterministic, w, op::AbstractMatrix, add, val, boost=1)
    w .+= op[:, add] .* val
    return (1,)
end
function apply_column!(::IsDeterministic, w, op, add, val, boost=1)
    diagonal_step!(w, op, add, val)
    spawn!(Exact(), w, op, add, val)
    return (1,)
end

"""
    IsStochasticWithThreshold{T=Float64}(threshold=1.0) <: StochasticStyle{T}

Stochastic propagation with floating point walker numbers. During the vector matrix product
each individual diagonal and spawning result is rounded stochastically if smaller than
`threshold` (before annihilations). For a more customizable stochastic style, see
[`IsDynamicSemistochastic`](@ref).

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticWithThreshold{T<:AbstractFloat} <: StochasticStyle{T}
    threshold::T
end
IsStochasticWithThreshold(args...) = IsStochasticWithThreshold{Float64}(args...)
IsStochasticWithThreshold{T}(t=1.0) where {T} = IsStochasticWithThreshold{T}(T(t))

function step_stats(::IsStochasticWithThreshold{T}) where {T}
    return ((:spawn_attempts, :spawns), MultiScalar(0, zero(T)))
end
function apply_column!(s::IsStochasticWithThreshold, w, op, add, val, boost=1)
    diagonal_step!(w, op, add, val, s.threshold)
    attempts, spawns = spawn!(WithReplacement(s.threshold), w, op, add, val, boost)
    return (attempts, spawns)
end

"""
    IsDynamicSemistochastic{T=Float64}(; kwargs...) <: StochasticStyle{T}

QMC propagation with floating-point walker numbers and reduced noise. All possible spawns
(offdiagonal elements in vector-matrix multiplication) are performed deterministically when
number of walkers in a configuration is high, as controlled by the `rel_spawning_threshold`
and `abs_spawning_threshold` keywords. Stochastic selection of spawns is controlled by the
`spawning` keyword.

By default, a stochastic vector compression is applied after annihilations are completed.
This behaviour can be changed to on-the-fly projection (as in [`IsStochasticInteger`](@ref)
or [`IsStochasticWithThreshold`](@ref)) by setting `late_compression=false`, or modifying
`spawning` and `compression`. See parameters below for a more detailed explanation.

## Parameters:

* `threshold = 1.0`: Values below this number are stochastically projected to this
  value or zero. See also [`ThresholdCompression`](@ref).

* `late_compression = true`: If this is set to `true`, stochastic vector compression is
  performed after all the spawns are performed. If it is set to `false`, values are
  stochastically projected as they are being spawned. `late_compression=true` is equivalent
  to setting `compression=`[`ThresholdCompression`](@ref)`(threshold)` and
  `spawning=`[`WithReplacement`](@ref)`()`.  `late_compression=false` is equivalent to
  `compression=`[`NoCompression`](@ref)`()` and
  `spawning=WithReplacement(threshold)`.

* `rel_spawning_threshold = 1.0`: If the walker number on a configuration times this
  threshold is greater than the number of offdiagonals, spawning is done
  deterministically. Should be set to 1 or more for best performance.

* `abs_spawning_threshold = Inf`: If the walker number on a configuration is greater than
  this value, spawning is done deterministically. Can be set to e.g.  `abs_spawning_threshold = 0.1 *
  target_walkers`.

* `spawning = WithReplacement()`: [`SpawningStrategy`](@ref) to use for the non-exact
  spawns.

* `compression = ThresholdCompression(threshold)`: [`CompressionStrategy`](@ref) used
  to compress the vector after a step. Overrides `threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsDynamicSemistochastic{
    T<:AbstractFloat,C<:CompressionStrategy,S<:DynamicSemistochastic
} <: StochasticStyle{T}
    proj_threshold::T
    compression::C
    spawning::S
end
function IsDynamicSemistochastic{T}(
    ;
    threshold=1.0, rel_spawning_threshold=1.0, abs_spawning_threshold=Inf,
    late_compression=true,
    compression=late_compression ? ThresholdCompression(threshold) : NoCompression(),
    spawning=late_compression ? WithReplacement() : WithReplacement(threshold),
) where {T}
    ds_spawning = DynamicSemistochastic(
        spawning, rel_spawning_threshold, abs_spawning_threshold
    )
    proj_threshold = T(spawning.threshold)
    return IsDynamicSemistochastic(proj_threshold, compression, ds_spawning)
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
function apply_column!(s::IsDynamicSemistochastic, w, op, add, val, boost=1)
    diagonal_step!(w, op, add, val, s.proj_threshold)
    exact, inexact, attempts, spawns = spawn!(s.spawning, w, op, add, val, boost)
    return (exact, inexact, attempts, spawns)
end

default_style(::Type{T}) where {T<:Integer} = IsStochasticInteger{T}()
default_style(::Type{T}) where {T<:AbstractFloat} = IsDeterministic{T}()
default_style(::Type{T}) where {T<:Complex{<:AbstractFloat}} = IsDeterministic{T}()
default_style(::Type{T}) where {T<:Complex{<:Integer}} = IsStochastic2Pop{T}()
