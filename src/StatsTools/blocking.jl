# blocking analysis of single time series
# `blocking_analysis()`
# `mean_and_se()`

"""
    BlockingResult(mean, err, err_err, p_cov, k, blocks)
Result of [`blocking_analysis`](@ref).

### Fields:
- `mean`: sample mean
- `err`: standard error (estimated standard deviation of the mean)
- `err_err`: estimated uncertainty of `err`
- `p_cov`: estimated pseudo covariance of `mean`, relevant for complex time series
- `k::Int`: k-1 blocking steps were used to uncorrelate time series
- `blocks::Int`: number of uncorrelated values after blocking

Has methods for [`NamedTuple`](@ref), [`val_and_errs`](@ref), [`val`](@ref), [`errs`](@ref),
[`mean_and_se`](@ref), `Measurements.:±`,
[`MonteCarloMeasurements.Particles`](@ref), and
`Statistics.cov` for `Complex` data.

**Example:**
```jldoctest; setup = :(Random.seed!(1234))
julia> blocking_analysis(smoothen(randn(2^10), 2^5))
BlockingResult{Float64}
  mean = -0.026 ± 0.029
  with uncertainty of ± 0.003638545517264226
  from 32 blocks after 5 transformations (k = 6).
```
"""
struct BlockingResult{T}
    mean::T
    err::Float64
    err_err::Float64
    p_cov::T # pseudo covariance for complex normal distribution
    k::Int
    blocks::Int
end
# constructor from NamedTuple of vectors returned by `blocks_with_m`
function BlockingResult(nt, k)
    T = eltype(nt.mean)
    if k < 0 # blocking failed
        return BlockingResult{T}(nt.mean[1], NaN, NaN, T(NaN), k, 0)
    end
    return BlockingResult{T}(
        nt.mean[1], nt.std_err[k], nt.std_err_err[k], nt.p_cov[k], k, nt.blocks[k]
    )
end

function Base.show(io::IO, r::BlockingResult{T}) where {T}
    println(io, "BlockingResult{$T}")
    println(io, "  mean = $(Measurements.measurement(r))")
    println(io, "  with uncertainty of ± $(r.err_err)")
    T <: Complex && println(io, "  cov = $(cov(r))")
    if r.k > 0
        println(io, "  from $(r.blocks) blocks after $(r.k-1) transformations (k = $(r.k)).")
    else
        println(io, "  Blocking unsuccessful. k = $(r.k). Try using more time steps!")
    end
end

"""
    cov(r::BlockingResult{<:Complex})
Return the covariance matrix of the multivariate normal distribution approximating
the uncertainty of the blocking result `r` of a complex time series.
See (https://en.wikipedia.org/wiki/Complex_normal_distribution).
"""
function Statistics.cov(r::BlockingResult{<:Complex})
    v_xx = real(r.err^2 + r.p_cov) / 2
    v_xy = imag(r.p_cov) / 2
    v_yy = real(r.err^2 - r.p_cov) / 2
    return [v_xx v_xy; v_xy v_yy]
end

"""
    measurement(r::BlockingResult)
    Measurements.±(r::BlockingResult)
Convert a `BlockingResult` into a `Measurement` for linear error propagation with
[`Measurements`](https://github.com/JuliaPhysics/Measurements.jl).

Limitation: Does not account for covariance in complex `BlockingResult`.
Consider using [`MonteCarloMeasurements.Particles(r)`](@ref)!
"""
function Measurements.measurement(r::BlockingResult{<:Real})
    return measurement(r.mean, r.err)
end
function Measurements.measurement(r::BlockingResult{<:Complex})
    Σ = cov(r) # real valued covariance matrix
    cm = complex(measurement(real(r.mean), √Σ[1, 1]), measurement(imag(r.mean), √Σ[2, 2]))
    return cm
end

"""
    MonteCarloMeasurements.Particles(r::BlockingResult; mc_samples = 2000)
    MonteCarloMeasurements.±(r::BlockingResult)
Convert a `BlockingResult` into a `Particles` object for nonlinear error propagation with
[`MonteCarloMeasurements`](https://github.com/baggepinnen/MonteCarloMeasurements.jl).
"""
function MonteCarloMeasurements.Particles(r::BlockingResult{<:Real}; mc_samples=2000)
    return Particles(mc_samples, Normal(r.mean, r.err))
end
function MonteCarloMeasurements.Particles(r::BlockingResult{<:Complex}; mc_samples=2000)
    Σ = cov(r) # real valued covariance matrix
    ps = Particles(mc_samples, MvNormal([real(r.mean), imag(r.mean)], Σ))
    return complex(ps[1], ps[2])
end
MonteCarloMeasurements.:±(r::BlockingResult) = Particles(r)

"""
    blocking_analysis(v::AbstractVector; α = 0.01, corrected = true, skip=0, warn=true)
    -> BlockingResult(mean, err, err_err, p_cov, k, blocks)
Compute the sample mean `mean` and estimate the standard deviation of the mean
(standard error) `err` of a correlated time series. It uses the blocking algorithm from
Flyvberg and Peterson [JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480)
and the M test of Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304) at significance level
``1-α``.

Use `skip` to skip the first `skip` elements in `v`.
`corrected` controls whether bias correction for variances is used. If decorrelating the
time series fails according to the M test, `NaN` is returned as the standard error and `-1`
for `k`. The keyword argument `warn` controls whether a warning message is logged.

The summary result is returned as a [`BlockingResult`](@ref). `k - 1` is the number of
blocking transformations required to pass the hypothesis test for an uncorrelated time
series and `err_err` the estimated standard error or `err`.

The detailed results from each reblocking step can be obtained with
[`blocking_analysis_data`](@ref).

See [`BlockingResult`](@ref), [`shift_estimator`](@ref), [`ratio_of_means`](@ref),
[`blocking_analysis_data`](@ref).
"""
function blocking_analysis(v::AbstractVector; kwargs...)
    return _blocking_analysis_data(v; kwargs...)[1]
end

@inline function _blocking_analysis_data(
    v::AbstractVector;
    α=0.01, corrected::Bool=true, skip=0, warn=true
)
    T = float(eltype(v))
    v = @view v[skip+1:end]
    if length(v) < 2
        if iszero(length(v))
            warn && @error "Attempted blocking on an empty vector."
            return BlockingResult(zero(T), NaN, NaN, T(NaN), -1, 0), (;)
        else # treat like failed M test
            warn && @warn "Attempted blocking on a vector of length $(length(v))."
            return BlockingResult(T(v[1]), NaN, NaN, T(NaN), -1, 1), (;)
        end
    end
    nt = blocks_with_m(v; corrected)
    k = mtest(nt.mj; α)
    if warn && k < 0
        @warn "Hypothesis testing failed during `blocking_analysis`. Try using more time steps!" k
    end
    return BlockingResult(nt, k), nt
end

"""
    blocking_analysis_data(v::AbstractVector; kwargs...) ->
    (; br::BlockingResult, df::DataFrame)
Perform a [`blocking_analysis`](@ref) and return the summary result `br` as well as a
`DataFrame` `df` with information about the standard error in each blocking step.

For a description of the keyword arguments see [`blocking_analysis`](@ref).

## Example
```julia-repl
julia> data = smoothen(rand(10_000), 2^6); # generate correlated data

julia> br, df = blocking_analysis_data(data)
(br = BlockingResult{Float64}
  mean = 0.5088 ± 0.0029
  with uncertainty of ± 0.00023454488294744232
  from 78 blocks after 7 transformations (k = 8).
, df = 13×6 DataFrame
 Row │ blocks  mean      std_err      std_err_err  p_cov       mj
     │ Int64   Float64   Float64      Float64      Float64     Float64
─────┼──────────────────────────────────────────────────────────────────────
   1 │  10000  0.508806  0.000375044  2.6521e-6    1.40658e-7  9715.08
   2 │   5000  0.508806  0.000528547  5.28599e-6   2.79361e-7  4778.14
   3 │   2500  0.508806  0.000743386  1.05152e-5   5.52622e-7  2298.64
   4 │   1250  0.508806  0.00104064   2.08212e-5   1.08293e-6  1056.24
   5 │    625  0.508806  0.00144177   4.08121e-5   2.07871e-6   427.949
   6 │    312  0.508736  0.00194209   7.78707e-5   3.77171e-6   128.711
   7 │    156  0.508736  0.00247921   0.00014081   6.14647e-6    17.3075
   8 │     78  0.508736  0.00291063   0.000234545  8.47174e-6     0.731386
   9 │     39  0.508736  0.00284613   0.000326474  8.10046e-6     0.901054
  10 │     19  0.508241  0.0026998    0.000449967  7.28892e-6     2.85915
  11 │      9  0.507939  0.00359907   0.000899766  1.29533e-5     1.08644
  12 │      4  0.509327  0.00440559   0.00179857   1.94092e-5     0.0370381
  13 │      2  0.509327  0.00432708   0.00305971   1.87237e-5     0.125)

julia> using StatsPlots; unicodeplots();

julia> plot([br.k,br.k],[0.0,maximum(df.std_err.+df.std_err_err)], label="m test");

julia> @df df plot!(
           1:length(:std_err), :std_err;
           err=:std_err_err, xlabel="k", label="std err",
           title="std err vs blocking steps"
       )
               ⠀⠀⠀⠀⠀⠀⠀⠀⠀std err vs blocking steps⠀⠀⠀⠀⠀⠀⠀⠀
               ┌────────────────────────────────────────┐
    0.00423501 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀│ m test
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⢸⠀⠀⠀⠀│ std err
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⢸⠀⠀⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⢸⠀⠀⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡠⢺⠒⠒⢺⠀⠀⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⡀⠀⠀⡆⣀⠤⡗⠉⠀⢸⠀⠀⢸⡆⠀⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡧⠤⠔⡗⠊⠉⡏⠀⠀⡇⠀⠀⢸⠀⠀⢸⢣⠀⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⡇⠀⠀⠁⠀⠀⠁⠀⠀⠁⠀⠀⠀⠀⠀⢸⠸⡀⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠴⠁⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠀⡇⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠊⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣦⠀│
               │⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⠀│
               │⠀⠀⠀⢀⣀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀│
               │⠀⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀│
   -0.00012335 │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠧⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│
               └────────────────────────────────────────┘
               ⠀0.64⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀k⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀13.36⠀
```
A vertical line at `k==8` indicates the blocking step identified by hypothesis testing
to decorrelate the time series data. The decorrelation length can thus be estimated at
``2^{k-1} = 2^7 = 128``. Note that the data was correlated with a sliding window of ``2^6``
steps.

See [`blocking_analysis`](@ref), [`BlockingResult`](@ref).
"""
function blocking_analysis_data(v::AbstractVector; kwargs...)
    br, nt = _blocking_analysis_data(v; kwargs...)
    return (; br, df=DataFrame(nt))
end

"""
    mean_and_se(v::AbstractVector; α = 0.01, corrected::Bool=true, skip=0) -> mean, err
    mean_and_se(r::BlockingResult) -> mean, err
Return the mean and standard error (as a tuple) of a time series obtained from
[`blocking_analysis`](@ref). See also [`BlockingResult`](@ref).
"""
mean_and_se(r::BlockingResult) = r.mean, r.err
function mean_and_se(v::AbstractVector; kwargs...)
    return mean_and_se(blocking_analysis(v; kwargs...))
end

"""
    blocker(v::Vector) -> new_v::Vector
Reblock the data by successively taking the mean of two adjacent data points to
form a new vector with a half of the `length(v)`. The last data point will be
discarded if `length(v)` is odd.
"""
function blocker(v::AbstractVector{T}) where {T}
    P = typeof(zero(T) / 1)
    new_v = Array{P}(undef, (length(v) ÷ 2))
    for i in 1:length(v)÷2
        @inbounds new_v[i] = (v[2i-1] + v[2i]) / 2
    end
    return new_v
end

"""
    mtest(mj::AbstractVector; α = 0.01) -> k
    mtest(table::NamedTuple; α = 0.01) -> k
Hypothesis test for decorrelation of a time series after blocking transformations
with significance level ``1-α`` after Jonson
[PRE (2018)](https://doi.org/10.1103/PhysRevE.98.043304).
`mj` or `table.mj` is expected to be a vector with relevant ``M_j`` values from a blocking analysis as
obtained from [`blocks_with_m()`](@ref).
Returns the row number `k` where the M-test is passed.
If the M-test has failed `mtest()` returns the value `-1`.
"""
function mtest(mj::AbstractVector; α=0.01)
    m = reverse(cumsum(reverse(mj)))
    k = 1
    while k <= length(m) - 1
        if m[k] < cquantile(Chisq(k), α) # compare to χ^2 distribution at quantile 1-α
            return k
        else
            k += 1
        end
    end
    return -1 # indicating the the M-test has failed
end
mtest(table::NamedTuple; kwargs...) = mtest(table.mj; kwargs...)

"""
    blocks_with_m(v; corrected = true) -> (;blocks, mean, std_err, std_err_err, p_cov, mj)
Perform the blocking algorithm from Flyvberg and Peterson
[JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480).
Returns named tuple with the results from all blocking steps. See [`mtest()`](@ref).
"""
@inline function blocks_with_m(v; corrected::Bool=true)
    T = float(eltype(v))
    R = real(T)
    n_steps = floor(Int, log2(length(v)))

    # initialise arrays to be returned
    blocks = Vector{Int}(undef, n_steps)
    mean_arr = Vector{T}(undef, n_steps)
    std_err = Vector{R}(undef, n_steps)
    std_err_err = Vector{R}(undef, n_steps)
    p_cov = Vector{T}(undef, n_steps)
    mj = Vector{R}(undef, n_steps)

    for i in 1:n_steps
        n = length(v)
        blocks[i] = n
        mean_v = mean(v)
        mean_arr[i] = mean_v
        variance = var(v; corrected, mean=mean_v) # variance
        # sample covariance ŷ(1) Eq. (6) [Jonsson]
        gamma = real(autocovariance(v, 1; corrected, mean=mean_v))
        # the M value Eq. (12) [Jonsson]
        mj[i] = n * ((n - 1) * variance / (n^2) + gamma)^2 / (variance^2)
        stderr_v = sqrt(variance / n) # standard error
        std_err[i] = stderr_v
        std_err_err[i] = stderr_v / sqrt(2 * (n - 1)) # error on standard error Eq. (28) [F&P]
        p_cov[i] = pseudo_cov(v, v; xmean=mean_v, ymean=mean_v, corrected) / n
        v = blocker(v) # re-blocking the dataset
    end
    (length(v) ≤ 0 || length(v) > 2) && @error "Something went wrong in `blocks_with_m`."
    return (; blocks, mean=mean_arr, std_err, std_err_err, p_cov, mj)
end
