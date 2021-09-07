# estimators for ratio of means of two time series (including blocking)

"""
    RatioBlockingResult(ratio, f, σ_f, δ_y, k, success)
Result of [`ratio_of_means()`](@ref).

### Fields:
- `ratio::P`: ratio with uncertainties propagated by MonteCarloMeasurements
- `f::T`: ratio of means
- `σ_f::T`: std from linear propagation
- `δ_y::T`: coefficient of variation for denominator (≤ 0.1 for normal approx)
- `k::Int`: k-1 blocking steps were used to uncorrelate time series
- `blocks::Int`: number of data values after blocking
- `success::Bool`: false if any of the blocking steps failed

Has methods for [`val_and_errs`](@ref), [`val`](@ref), [`errs`](@ref).

Note: to compute statistics on the `RatioBlockingResult`, use functions `pmedian`,
`pquantile`, `pmiddle`, `piterate`, `pextrema`, `pminimum`, `pmaximum`, `pmean`, and `pcov`.
"""
struct RatioBlockingResult{T,P}
    ratio::P    # ratio with uncertainties propagated by MonteCarloMeasurements
    f::T        # ratio of means
    σ_f::T      # std from linear propagation
    δ_y::T      # coefficient of variation for denominator (≤ 0.1 for normal approx)
    k::Int      # k-1 blocking steps were used to uncorrelate time series
    blocks::Int # number of data values after blocking
    success::Bool # false if any of the blocking steps failed
end
# make it behave like Particles
# Base.iterate(r::RatioBlockingResult, args...) = iterate(r.ratio, args...)
# Base.length(r::RatioBlockingResult) = length(r.ratio)
# Base.eltype(r::RatioBlockingResult) = eltype(r.ratio)
MacroTools.@forward RatioBlockingResult.ratio MonteCarloMeasurements.pmedian, MonteCarloMeasurements.pquantile
MacroTools.@forward RatioBlockingResult.ratio MonteCarloMeasurements.pmiddle, MonteCarloMeasurements.piterate, MonteCarloMeasurements.pextrema
MacroTools.@forward RatioBlockingResult.ratio MonteCarloMeasurements.pminimum, MonteCarloMeasurements.pmaximum
MacroTools.@forward RatioBlockingResult.ratio MonteCarloMeasurements.pmean, MonteCarloMeasurements.pcov

import Statistics: median, quantile, mean, cov
import Base: iterate, extrema, minimum, maximum
@deprecate median(r::RatioBlockingResult) pmedian(r)
@deprecate quantile(r::RatioBlockingResult, args...) pquantile(r, args...)
@deprecate iterate(r::RatioBlockingResult, args...) piterate(r, args...)
@deprecate extrema(r::RatioBlockingResult) pextrema(r)
@deprecate minimum(r::RatioBlockingResult) pminimum(r)
@deprecate maximum(r::RatioBlockingResult) pmaximum(r)
@deprecate mean(r::RatioBlockingResult) pmean(r)
@deprecate cov(r::RatioBlockingResult) pcov(r)

function MonteCarloMeasurements.pmedian(r::RatioBlockingResult{<:Complex})
    complex(pmedian(real(r.ratio)), pmedian(imag(r.ratio)))
end
function MonteCarloMeasurements.pquantile(r::RatioBlockingResult{<:Complex}, args...)
    throw(ErrorException("""
    `pquantile(r, args...)` called with `Complex` data type.
    Try `pquantile(real(r.ratio), args...)` and `pquantile(imag(r.ratio), args...)`!
    """
    ))
end

function Base.show(io::IO, r::RatioBlockingResult{T,P}) where {T<:Real, P}
    q = pquantile(r.ratio, [0.16,0.5,0.84])
    qr95 = pquantile(r.ratio, [0.025,0.975])
    println(io, "RatioBlockingResult{$T,$P}")
    println(io, f"  ratio = \%g(q[2]) ± (\%g(q[3]-q[2]), \%g(q[2]-q[1])) (MC)")
    println(io, f"  95% confidence interval: [\%g(qr95[1]), \%g(qr95[2])]) (MC)")
    println(io, f"  linear error propagation: \%g(r.f) ± \%g(r.σ_f)")
    println(io, f"  |δ_y| = |\%g(r.δ_y)| (≤ 0.1 for normal approx)")
    if r.success
        println(io, "  Blocking successful with $(r.blocks) blocks after $(r.k-1) transformations (k = $(r.k)).")
    else
        println(io, "  Blocking unsuccessful.")
    end
end

function Base.show(io::IO, r::RatioBlockingResult{T,P}) where {T<:Complex, P}
    qr = pquantile(real(r.ratio), [0.16,0.5,0.84])
    qi = pquantile(imag(r.ratio), [0.16,0.5,0.84])
    println(io, "RatioBlockingResult{$T,$P}")
    println(io, f"  ratio = \%g(qr[2]) ± (\%g(qr[3]-qr[2]), \%g(qr[2]-qr[1])) + [\%g(qi[2]) ± (\%g(qi[3]-qi[2]), \%g(qi[2]-qi[1]))]*im (MC)")
    qr95 = pquantile(real(r.ratio), [0.025,0.975])
    qi95 = pquantile(imag(r.ratio), [0.025,0.975])
    println(io, f"  95% confidence interval real: [\%g(qr95[1]), \%g(qr95[2])] (MC)")
    println(io, f"  95% confidence interval imag: [\%g(qi95[1]), \%g(qi95[2])] (MC)")
    println(io, "  linear error propagation: ($(r.f)) ± ($(r.σ_f))")
    println(io, f"  |δ_y| = \%g(abs(r.δ_y)) (≤ 0.1 for normal approx)")
    if r.success
        println(io, "  Blocking successful with $(r.blocks) blocks after $(r.k-1) transformations (k = $(r.k)).")
    else
        println(io, "  Blocking unsuccessful.")
    end
end

"""
    ratio_of_means(num, denom; α=0.01, corrected=true, mc_samples=nothing, skip=0) -> r
Estimate the ratio of `mean(num)/mean(denom)` assuming that `num` and `denom` are possibly
correlated time series, skipping the first `skip` elements. A blocking analysis with
m-test is used to uncorrelate the time series, see [`blocking_analysis()`](@ref). The
remaining standard error and correlation of the means is propagated using
`MonteCarloMeasurements`. The results are reported as a [`RatioBlockingResult`](@ref).

Robust estimates for the ratio are obtained from `pmedian(r)` and confidence intervals from
`pquantile()`, e.g. `pquantile(r, [0.025, 0.975])` for the 95% confidence interval.

Estimates from linear uncertainty propagation are returned as `r.f` and `r.σ_f` using
[`x_by_y_linear()`](@ref).

The keyword `mc_samples` controls the number of samples used for error propagation
by `MonteCarloMeasurements`. Use `nothing` for the default and `Val(1000)` to set the number
to 1000 samples in a type-consistent way.

Note: to compute statistics on the [`RatioBlockingResult`](@ref), use functions `pmedian`,
`pquantile`, `pmiddle`, `piterate`, `pextrema`, `pminimum`, `pmaximum`, `pmean`, and `pcov`.
"""
function ratio_of_means(num, denom; α=0.01, corrected=true, mc_samples=nothing, skip=0)
    num = @view num[skip+1:end]
    denom = @view denom[skip+1:end]
    # determine how many blocking steps are needed to uncorrelate data
    bt_num = blocking_analysis(num; α)
    bt_den = blocking_analysis(denom; α)
    # choose largst k from mtest on blocking analyses on numerator and denominator
    ks = (bt_num.k, bt_den.k)
    success = any(k->k<0, ks) ? false : true
    k = max(ks...)

    # MC and linear error propagation
    r, f, σ_f, δ_y, blocks = ratio_estimators(num, denom, k; corrected, mc_samples)

    # more accurately before blocking
    f = bt_num.mean/bt_den.mean
    return RatioBlockingResult(r, f, σ_f, δ_y, k, blocks, success)
end

"""
    x_by_y_linear(μ_x,μ_y,σ_x,σ_y,ρ) -> f, σ_f
Linear error propagation for ratio `f = x/y` assuming `x` and `y` are correlated normal
random variables and assuming the ratio can be approximated as a normal distribution.
See [wikipedia](https://en.wikipedia.org/wiki/Propagation_of_uncertainty) and
[Díaz-Francés, Rubio (2013)](http://link.springer.com/10.1007/s00362-012-0429-2).
```math
σ_f = \\sqrt{\\frac{σ_x}{μ_y}^2 + \\frac{μ_x σ_y}{μ_y^2}^2 - \\frac{2 ρ μ_x}{μ_y^3}}
```
"""
function x_by_y_linear(μ_x, μ_y, σ_x, σ_y, ρ)
    f = μ_x/μ_y
    # σ_f = abs(f)*sqrt((σ_x/μ_x)^2 + (σ_y/μ_y)^2 - 2*ρ/(μ_x*μ_y))
    σ_f = sqrt((σ_x/μ_y)^2 + (μ_x*σ_y/μ_y^2)^2 - 2*ρ*μ_x/μ_y^3)
    return f, σ_f
end

"""
    particles(samples, d)
    particles(::Nothing, d)
    particles(::Val{T}, d) where T
Return `Particles` object from  `MonteCarloMeasurements` using  a type-stable constructor
if possible. Pass `nothing` for the default number of particles or `Val(1_000)` for using
1000 particles in a type-stable manner. If `d` is a `Particles` object it is passed through
without re-sampling.
"""
particles(samples, d::Distribution) = particles(Val(samples),d)
particles(::Nothing, d::Distribution) = Particles(d)
particles(::Val{T}, d::Distribution) where T = Particles{eltype(d),T}(Random.GLOBAL_RNG,d)
function particles(samples, m::Measurements.Measurement)
    particles(samples, Normal(Measurements.value(m),Measurements.uncertainty(m)))
end
particles(_, p::Particles) = p # don't re-sample if it is already a Particles object
"""
    particles(samples, (μ_x, μ_y, var_x, var_y, ρ))
Return bivariate `Particles` object from `MonteCarloMeasurements` taking into account
correlations. Zero variance parameters are supported.
"""
function particles(samples, (μ_x, μ_y, var_x, var_y, ρ))
    if var_x < √eps(var_x) || var_y < √eps(var_y)
        return [    # ignore covariance
            particles(samples, Normal(μ_x, √var_x)),
            particles(samples, Normal(μ_y, √var_y))
        ]
    else
        ρ = min(ρ, abs(√(var_x * var_y) - √eps(var_y)))
        particles(samples, MvNormal([μ_x,μ_y],[var_x ρ; ρ var_y]))
    end
end

"""
    ratio_estimators(x, y, [k]; corrected=true, mc_samples=10_000) -> (; r, f, σ_f, δ_y, n)
Estimators for the ratio of means `mean(x)/mean(y)`.
If `k` is given, `k-1` blocking steps are performed to remove internal correlations in
the time series `x` and `y`. Otherwise these are assumed to be free of internal
correlations. Correlations between `x` and `y` may be present and are taken into
account.

### Return values:
- `r::Particles` is the Monte Carlo sampled ratio estimator, see [`Particles`](@ref)
- `f = mean(x)/mean(y)`
- `σ_f` standard deviation of `f` from linear error propagation (normal approximation)
- `δ_y = std(y)/mean(y)` coefficient of variation; < 0.1 for normal approximation to work
- `n`: number of uncorrelated data used for uncertainty estimation
"""
function ratio_estimators(x, y; corrected = true, mc_samples = nothing)
    n = length(x)
    @assert n == length(y)
    μ_x = mean(x)
    var_x = var(x; corrected)/n # variance of mean
    μ_y = mean(y)
    var_y = var(y; corrected)/n # variance of mean
    ρ = cov(x, y; corrected)/n # estimated correlation of sample means μ_x and μ_y

    # Monte Carlo sampling of correlated normal distribution of sample means for x and y
    x_y_ps = particles(mc_samples, (μ_x, μ_y, var_x, var_y, ρ))
    # Note: type instability creeps in here through `Particles`
    r = x_y_ps[1]/x_y_ps[2] # MC sampled ratio of means

    # linear error propagation
    f, σ_f = x_by_y_linear(μ_x, μ_y, √var_x, √var_y, ρ)

    # coefficient of variation, should be <0.1 for normal approximation
    # [Kuethe(2000), Diaz-Frances & Rubio (2013)]
    δ_y = √var_y/μ_y
    return (; r, f, σ_f, δ_y, n)
end

function ratio_estimators(num, denom, k; corrected = true, mc_samples = nothing)
    for i in 1:(k-1) # decorrelate time series by `k-1` blocking steps
        num = blocker(num)
        denom = blocker(denom)
    end
    return ratio_estimators(num, denom; corrected, mc_samples)
end

function ratio_estimators(
    x::AbstractVector{<:Complex}, y::AbstractVector{<:Complex};
    corrected = true, mc_samples = nothing,
)
    n = length(x)
    @assert n == length(y)
    μ_x = mean(x)
    var_x = var(x; corrected)/n # variance of mean
    μ_y = mean(y)
    var_y = var(y; corrected)/n # variance of mean
    ρ = cov(x, y; corrected)/n # estimated correlation of sample means μ_x and μ_y

    Σ = [
        var(real(x); corrected)          cov(real(x), imag(x); corrected) cov(real(x), real(y); corrected) cov(real(x), imag(y); corrected)
        cov(imag(x), real(x); corrected) var(imag(x); corrected)          cov(imag(x), real(y); corrected) cov(imag(x), imag(y); corrected)
        cov(real(y), real(x); corrected) cov(real(y), imag(x); corrected) var(real(y); corrected)          cov(real(y), imag(y); corrected)
        cov(imag(y), real(x); corrected) cov(imag(y), imag(x); corrected) cov(imag(y), real(y); corrected) var(imag(y); corrected)
    ]/n
    # Monte Carlo sampling of correlated normal distribution of sample means for x and y
    x_y_ps = particles(
        mc_samples, MvNormal([real(μ_x), imag(μ_x), real(μ_y), imag(μ_y)], Σ)
    )
    r = (x_y_ps[1] + im*x_y_ps[2]) / (x_y_ps[3] + im*x_y_ps[4]) # MC sampled ratio of means

    # linear error propagation
    f, σ_f = x_by_y_linear(μ_x, μ_y, √var_x, √var_y, ρ)

    # coefficient of variation, should be <0.1 for normal approximation
    # [Kuethe(2000), Diaz-Frances & Rubio (2013)]
    δ_y = √var_y/μ_y
    return (; r, f, σ_f, δ_y, n)
end
