"""
    IsStochasticInteger{T=Int}() <: StochasticStyle{T}

Trait for generalised vector of configurations indicating stochastic propagation as seen in
the original FCIQMC algorithm.

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticInteger{T<:Integer} <: StochasticStyle{T} end
IsStochasticInteger() = IsStochasticInteger{Int}()

function step_stats(::IsStochasticInteger)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0, 0, 0, 0, 0),
    )
end
function fciqmc_col!(
    ::IsStochasticInteger, w, ham, add, num::Real, shift, dτ
)
    clones, deaths, zombies, ann_diag = diagonal_step!(w, ham, add, num, dτ, shift)
    spawns, ann_offdiag = spawn!(WithReplacement(), w, ham, add, num, dτ)
    return (spawns, deaths, clones, zombies, ann_diag + ann_offdiag)
end

struct IsStochastic2Pop{T<:Complex{<:Integer}} <: StochasticStyle{T} end
IsStochastic2Pop() = IsStochastic2Pop{Complex{Int}}()

function step_stats(::IsStochastic2Pop)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(ntuple(_ -> 0 + 0im, Val(5))),
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

    clones, deaths, zombies, ann_d = diagonal_step!(w, ham, add, val, dτ, shift)

    return (spawns, deaths, clones, zombies, ann_o + ann_d)
end

"""
    IsStochastic2PopOld{T=Complex{Int}}() <: StochasticStyle{T}

Trait for generalised vector of configurations indicating stochastic propagation with
complex walker numbers representing two populations of integer walkers.

See also [`StochasticStyle`](@ref).
"""
struct IsStochastic2PopOld{T<:Complex{<:Integer}} <: StochasticStyle{T} end
IsStochastic2PopOld() = IsStochastic2PopOld{Complex{Int}}()

function step_stats(::IsStochastic2PopOld)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(ntuple(_ -> 0 + 0im, Val(5))),
    )
end
function fciqmc_col!(::IsStochastic2PopOld, w, ham, add, cnum::Complex, cshift, dτ)
    # version for complex integer psips
    # off-diagonal: spawning psips
    spawns::typeof(cnum) = deaths = clones = zombies = annihilations = zero(cnum)
    # stats reported are complex, for each component separately
    hops = offdiagonals(ham,add)
    # real psips first
    num = real(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(Int, pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(real(w[naddress])) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(real(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            deposit!(w, naddress, nspawns, add => cnum)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # now imaginary psips
    num = imag(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(Int, pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = im*convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(imag(w[naddress])) * sign(imag(nspawns)) < 0 # record annihilations
            annihilations += min(abs(imag(w[naddress])),abs(imag(nspawns)))
        end
        if !iszero(nspawns)
            deposit!(w, naddress, nspawns, add => cnum)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += im*abs(nspawns)
        end
    end

    # diagonal death / clone
    shift = real(cshift) # use only real part of shift for now
    dME = diagonal_element(ham,add)
    pd = dτ * (dME - shift) # real valued so far
    cnewdiagpop = (1-pd)*cnum # now it's complex
    # treat real part
    newdiagpop = real(cnewdiagpop)
    num = real(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) + 0im # now real integer type
    if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    end
    deposit!(w, add, ndiags, add => num) # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += abs(real(ndiags) - num)
    elseif pd < 1
        deaths += abs(real(ndiags) - num)
    else
        zombies += abs(real(ndiags))
    end
    # treat imaginary part
    newdiagpop = imag(cnewdiagpop)
    num = imag(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = im*convert(typeof(num),ndiag) # now complex integer type
    if sign(imag(w[add])) ≠ sign(imag(ndiag)) # record annihilations
        annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    end
    deposit!(w, add, ndiags, add => cnum)
    if  pd < 0 # record event statistics
        clones += im*abs(imag(ndiags) - num)
    elseif pd < 1
        deaths += im*abs(imag(ndiags) - num)
    else
        zombies += im*abs(imag(ndiags))
    end

    # imaginary part of shift leads to spawns across populations
    cspawn = im*dτ*imag(cshift)*cnum # to be spawned as complex number with signs

    # real part - to be spawned into real walkers
    rspawn = real(cspawn) # float with sign
    nspawn = trunc(Int, rspawn) # deal with integer part separately
    cRand() < abs(rspawn - nspawn) && (nspawn += Int(sign(rspawn))) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn)
    if sign(real(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(real(w[add])),abs(nspawn))
    end
    deposit!(w, add, cnspawn, add => cnum)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    # imag part - to be spawned into imaginary walkers
    ispawn = imag(cspawn) # float with sign
    nspawn = trunc(Int, ispawn) # deal with integer part separately
    cRand() < abs(ispawn - nspawn) && (nspawn += Int(sign(ispawn))) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
    if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(imag(w[add])),abs(nspawn))
    end
    deposit!(w, add, cnspawn, add => cnum)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    return (spawns, deaths, clones, zombies, annihilations)
    # note that w is not returned
end

"""
    IsDeterministic{T=Float64}() <: StochasticStyle{T}

Trait for generalised vector of configuration indicating deterministic propagation of
walkers. The optional [`compression`](@ref) argument can set a
[`CompressionStrategy`](@ref).

See also [`StochasticStyle`](@ref).
"""
struct IsDeterministic{T<:AbstractFloat,C<:CompressionStrategy} <: StochasticStyle{T}
    compression::C
end
IsDeterministic{T}(compression::C=NoCompression()) where {T,C} = IsDeterministic{T,C}(compression)
IsDeterministic(args...) = IsDeterministic{Float64}(args...)

CompressionStrategy(s::IsDeterministic) = s.compression

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

Trait for generalised vector of configurations indicating stochastic propagation with real
walker numbers and cutoff `threshold`.

During stochastic propagation, walker numbers small than `threshold` will be stochastically
projected to either zero or `threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticWithThreshold{T<:AbstractFloat} <: StochasticStyle{T}
    threshold::T
end
IsStochasticWithThreshold(t=1.0) = IsStochasticWithThreshold{typeof(float(t))}(float(t))

function step_stats(::IsStochasticWithThreshold)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0.0, 0.0, 0.0, 0.0, 0.0)
    )
end
function fciqmc_col!(s::IsStochasticWithThreshold, w, ham, add, val, shift, dτ)
    deaths, clones, zombies, ann_d = diagonal_step!(w, ham, add, val, dτ, shift)
    spawns, ann_o = spawn!(WithReplacement(s.threshold, 1.0), w, ham, add, val, dτ)
    return (spawns, deaths, clones, zombies, ann_d + ann_o)
end

"""
    IsDynamicSemistochastic{T=Float64}(; kwargs...) <: StochasticStyle{T}

QMC propagation with non-integer walker numbers and reduced noise. All possible spawns are
performed deterministically when number of walkers in a configuration is high. Unlike with
[`IsStochasticInteger`](@ref) or [`IsStochasticWithThreshold`](@ref), where spawns are
projected on the fly, stochastic vector compression is applied after spawning and diagonal
death steps.

Note: if you want `IsDynamicSemistochastic` to project spawns as they are being performed, set a threshold to `spawning`, and set `compression` to [`NoCompression`](@ref).

## Parameters:

* `rel_threshold = 1.0`: If the walker number on a configuration times this threshold
  is greater than the number of offdiagonals, spawning is done deterministically. Should be
  set to 1 or more for best performance.

* `abs_threshold = Inf`: If the walker number on a configuration is greater than this value,
  spawning is done deterministically. Can be set to e.g.
  `abs_threshold = 0.1 * target_walkers`.

* `proj_threshold = 1.0`: Values below this number are stochastically projected to this
  value or zero. See also [`IsStochasticWithThreshold`](@ref).

* `spawning = WithReplacement()`: [`SpawningStrategy`](@ref) to use for the non-exact spawns.

* `compression = ThresholdCompression(proj_threshold)`: [`CompressionStartegy`](@ref) used
  to compress the vector after a step. Overrides `proj_threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsDynamicSemistochastic{
    T<:AbstractFloat,C<:CompressionStrategy,S<:DynamicSemistochastic
} <: StochasticStyle{T}
    spawning::S
    compression::C
end
function IsDynamicSemistochastic{T}(
    ;
    proj_threshold=1.0, strength=1.0, rel_threshold=1.0, abs_threshold=Inf,
    compression=ThresholdCompression(proj_threshold),
    spawning=WithReplacement(),
) where {T}
    s = DynamicSemistochastic(spawning, rel_threshold, abs_threshold)
    return IsDynamicSemistochastic{T,typeof(compression),typeof(s)}(s, compression)
end
IsDynamicSemistochastic(; kwargs...) = IsDynamicSemistochastic{Float64}(; kwargs...)

CompressionStrategy(s::IsDynamicSemistochastic) = s.compression

function step_stats(::IsDynamicSemistochastic)
    return (
        (:exact_steps, :inexact_steps, :spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
end
function fciqmc_col!(s::IsDynamicSemistochastic, w, ham, add, val, shift, dτ)
    clones, deaths, zombies, ann_d = diagonal_step!(w, ham, add, val, dτ, shift)
    exact, inexact, spawns, ann_o = spawn!(s.spawning, w, ham, add, val, dτ)
    return (exact, inexact, spawns, deaths, clones, zombies, ann_d + ann_o)
end

"""
    IsExplosive{T=Float64}(; splatter_factor, explosion_threshold, compression) <: StochasticStyle{T}

QMC propagation with explosive walkers. Walkers with small walker numbers do not perform the
standard death/spawning steps. Instead, a walker will either die completely and spawn with a
greater magnitude (refered to as explosion below), or stay unchanged and not spawn. The
probabilty of exploding is controlled by the shift and `dτ`.

Walkers with high walker numbers spawn as if [`IsDynamicSemistochastic`](@ref) was used.

Like [`IsDynamicSemistochastic`](@ref), the vector is compressed after all spawning is
performed.

## Parameters

* `splatter_factor = 1.0`: The spawning strength to use with exploded walkers.

* `explosion_threshold = 1.0`: Entries smaller or equal than this value will attempt to
  explode.

* `proj_threshold = 1.0`: Threshold to use in vector compression.

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

function step_stats(::IsExplosive)
    return (
        (:explosions, :ticks, :normal_steps,
         :explosive_spawns, :normal_spawns,
         :clones, :deaths, :zombies
         ),
        MultiScalar(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
end
function fciqmc_col!(s::IsExplosive{T}, w, ham, add, val, shift, dτ) where {T}
    explosions = normal_steps = ticks = 0
    explosive_spawns = normal_spawns = 0.0
    clones = deaths = zombies = 0.0

    pd = dτ * (diagonal_element(ham, add) - shift)
    if abs(val) ≤ s.explosion_threshold && 0 ≤ pd < 1
        if cRand() < pd
            _, _, explosive_spawns, _ = spawn!(
                DynamicSemistochastic(strat=WithReplacement(zero(T), s.splatter_factor)),
                w, ham, add, val / pd, dτ,
            )
            explosions = 1
        else
            deposit!(w, add, val, add => val)
            ticks = 1
        end
    else
        clones, deaths, zombies, _ = diagonal_step!(w, ham, add, val, dτ, shift)
        _, _, normal_spawns, _ = spawn!(
            DynamicSemistochastic(strat=WithReplacement()), w, ham, add, val, dτ,
        )
        normal_steps = 1
    end

    return (
        explosions, ticks, normal_steps,
        explosive_spawns, normal_spawns,
        clones, deaths, zombies,
    )
end

default_style(::Type{T}) where {T<:Integer} = IsStochasticInteger{T}()
default_style(::Type{T}) where {T<:AbstractFloat} = IsDeterministic{T}()
default_style(::Type{T}) where {T<:Complex{<:Integer}} = IsStochastic2Pop{T}()
