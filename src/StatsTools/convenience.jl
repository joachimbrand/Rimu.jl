# convenience functions for extracting data from DataFrames

"""
    replica_fidelity(rr::Tuple; p_field = :hproj, skip = 0)
Compute the fidelity of the average coefficient vector and the projector defined in
`p_field` from the result of replica [`fci_qmc!()`](@ref) passed as  argument `rr` (a
tuple of DataFrames).
Calls [`ratio_of_means()`](@ref) to perform a blocking analysis
on a ratio of the means of separate time series and returns a
[`RatioBlockingResult`](@ref).
The first `skip` steps in the time series are skipped.

The fidelity of states `|ψ⟩` and `|ϕ⟩` is defined as
```math
F(ψ,ϕ) = \\frac{|⟨ψ|ϕ⟩|^2}{⟨ψ|ψ⟩⟨ϕ|ϕ⟩} .
```
Specifically, `replica_fidelity` computes
```math
F(\\mathbf{v},⟨\\mathbf{c}⟩) =
    \\frac{⟨(\\mathbf{c}_A⋅\\mathbf{v})(\\mathbf{v}⋅\\mathbf{c}_B)⟩}
    {⟨\\mathbf{c}_A⋅\\mathbf{c}_B⟩} ,
```
where `v` is the projector specified by `p_field`, which is assumed to be normalised to
unity with the two-norm (i.e. `v⋅v == 1`), and ``\\mathbf{c}_A`` and ``\\mathbf{c}_B``
are two replica coefficient vectors.
"""
function replica_fidelity(rr::Tuple; p_field = :hproj, skip = 0, args...)
    df1 = rr[2][1] :: DataFrame # first replica DataFrame
    df2 = rr[2][2] :: DataFrame # second replica DataFrame
    dfr = rr[1] :: DataFrame # joint results DataFrame

    # numerator for fidelity calculation as time series (array)
    fid_num = conj(getproperty(df1, p_field)) .* getproperty(df2, p_field)
    fid_num = fid_num[skip+1:end]
    # denominator
    fid_den = dfr.xdoty[skip+1:end]

    return ratio_of_means(fid_num, fid_den; args...)
end
# New way of doing replica.
function replica_fidelity(df::DataFrame; p_field = :hproj, skip = 0, args...)
    p_field_1 = Symbol(p_field, :_1)
    p_field_2 = Symbol(p_field, :_2)
    fid_num = conj(getproperty(df, p_field_1)) .* getproperty(df, p_field_2)
    fid_num = fid_num[skip+1:end]
    # denominator
    fid_den = df.c1_dot_c2[skip+1:end]

    return ratio_of_means(fid_num, fid_den; args...)
end

"""
    med_and_errs(p) -> (; med, err1_l, err1_u, err2_l, err2_u)
Convenience function for extracting plottable data from a distribution or
an uncertain object created by the packages `MonteCarloMeasurements` or `Measurements`.
Returns the median `med` and the lower `err1_l` and upper `err1_u` standard error
(for 1σ or 68% confidence inteval). `err2_l` and `err2_u` provide the lower and upper error
limits for the 2σ or 95% confidence interval.

**Example:**
```julia
julia> results = [Particles(d) for d in datasets] # Particles[]
julia> res_w_errs = med_and_errs.(results) # Vector of NamedTuple's with standard errors
julia> res_df = DataFrame(res_w_errs) # results as DataFrame with lower an upper error
1×5 DataFrame
 Row │ med      err1_l     err1_u     err2_l    err2_u
     │ Float64  Float64    Float64    Float64   Float64
─────┼────────────────────────────────────────────────────
   1 │ 1.01325  0.0173805  0.0183057  0.034042  0.0366713
```
"""
function med_and_errs(p)
    q = pquantile(p, [0.025, 0.16, 0.5, 0.84, 0.975])
    med = q[3]
    err1_l = med - q[2]
    err1_u = q[4] - med
    err2_l = med - q[1]
    err2_u = q[5] - med
    return (; med, err1_l, err1_u, err2_l, err2_u)
end
function med_and_errs(p::Measurements.Measurement)
    med = Measurements.value(p)
    err1_l = err1_u = Measurements.uncertainty(p)
    err2_l = err2_u = 2err1_l
    return (; med, err1_l, err1_u, err2_l, err2_u)
end

"""
    ratio_with_errs(r::RatioBlockingResult)
    -> (;ratio=med, err1_l, err1_u, err2_l, err2_u, f, σ_f, δ_y, k, success)
Convenience function for extracting plottable data from [`RatioBlockingResult`](@ref).
Returns `NamedTuple` with median and standard error of `r` extracted by
[`p_to_errs()`](@ref). See also [`ratio_of_means()`](@ref).

**Example:**
```julia
julia> results = [ratio_of_means(n[i], d[i]; args...) for i in datasets]
julia> res_w_errs = ratio_with_errs.(results) # Vector of NamedTuple's with standard errors
julia> res_df = DataFrame(res_w_errs) # results as DataFrame with lower an upper error
1×10 DataFrame
 Row │ ratio    err1_l     err1_u     err2_l    err2_u     f        σ_f        δ_y        k      success
     │ Float64  Float64    Float64    Float64   Float64    Float64  Float64    Float64    Int64  Bool
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 1.01325  0.0173805  0.0183057  0.034042  0.0366713  1.01361  0.0181869  0.0128806      2     true
```
"""
function ratio_with_errs(r::RatioBlockingResult)
    med, err1_l, err1_u, err2_l, err2_u = med_and_errs(r.ratio)
    return (;
        ratio=med,
        err1_l,
        err1_u,
        err2_l,
        err2_u,
        f=r.f,
        σ_f=r.σ_f,
        δ_y=r.δ_y,
        k=r.k,
        success=r.success,
    )
end

"""
    to_measurement(p::MonteCarloMeasurements.Particles) -> ::Measurements.measurement
Convert an uncertain number from `MonteCarloMeasurements` to `Measurements` format using the median as the central
point. The new `±` boundaries will include the 68% quantile around the median.
"""
function to_measurement(p::MonteCarloMeasurements.Particles)
        q = pquantile(p, [0.16, 0.5, 0.84])
        σ = max(q[3]-q[2],q[2]-q[1]) # take the larger one
        return Measurements.measurement(q[2],σ)
end

"""
    val(x)
Return the best estimate value for an uncertain `x`. Defaults to the `median`
for uncertain `x` represented by a (sampled) distribution. Supports `MonteCarloMeasurements`
and `Measurements`.

See [`errs`](@ref), [`BlockingResult`](@ref), [`RatioBlockingResult`](@ref).
"""
val(x) = x
val(m::Measurements.Measurement) = Measurements.value(m)
val(p::AbstractParticles) = pmedian(p)
val(r::BlockingResult) = r.mean
val(r::RatioBlockingResult) = pmedian(r)

import Measurements: Measurement, uncertainty, value
"""
    val_and_errs(x; n=1, p=nothing, name=:val) -> (;val, val_l, val_u)
Return the median and the lower and upper error bar for the uncertain value `x`. The
interval `[val(x)-val_l, val(x)+val_u]` represents the confidence interval at level `n*σ`,
or at probability `p`. Setting `p` overrides `n`. Supports `MonteCarloMeasurements`
and `Measurements`. The names in the `NamedTuple` can be changed with `name`.

See [`val`](@ref), [`errs`](@ref), [`BlockingResult`](@ref), [`RatioBlockingResult`](@ref).
"""
function val_and_errs(x; name=:val, kwargs...)
    return (; Symbol(name) => x, Symbol(name, :_l) => zero(x), Symbol(name, :_u) => zero(x))
end
function val_and_errs(
    m::T; p=nothing, n=1, name=:val
) where T <:Union{Measurement, AbstractParticles}
    return _errs(p, m, n, name)
end
function _errs(::Nothing, m::Measurement, n, name)
    σ = uncertainty(m)
    return (; Symbol(name) => value(m), Symbol(name, :_l) => n*σ, Symbol(name, :_u) => n*σ)
end
function _errs(p, m::Measurement, _, name)
    d = Normal(value(m), uncertainty(m))
    cp1 = (1-p)/2
    cp2 = 1-cp1
    q1, q2 = quantile(d, cp1), quantile(d, cp2)
    return (;
        Symbol(name) => value(m),
        Symbol(name, :_l) => value(m) - q1,
        Symbol(name, :_u) => q2 - value(m),
    )
end
function _errs(::Nothing, m::AbstractParticles, n, name)
    p = erf(n/√2)
    return _errs(p, m, n, name)
end
function _errs(p, m::AbstractParticles, _, name)
    cp1 = (1-p)/2
    cp2 = 1-cp1
    q1, q2, q3 = pquantile(m, (cp1, 0.5, cp2))
    return (;Symbol(name) => q2, Symbol(name, :_l) => q2 - q1, Symbol(name, :_u) => q3 - q2)
end
function val_and_errs(r::BlockingResult; kwargs...)
    return val_and_errs(measurement(r.mean, r.err); kwargs...)
end
val_and_errs(r::RatioBlockingResult; kwargs...) = val_and_errs(r.ratio; kwargs...)

"""
    errs(x; n=1, p=nothing, name=:err) -> (; err_l, err_u)
Return the lower and upper error bar for the uncertain value `x`.

See [`val_and_errs`](@ref).
"""
function errs(args...; name=:err, kwargs...)
    _, err_l, err_u = val_and_errs(args...; kwargs...)
    return (; Symbol(name, :_l) => err_l, Symbol(name, :_u) => err_u)
end
