# convenience functions for extracting data from DataFrames

"""
    to_measurement(p::MonteCarloMeasurements.Particles) -> ::Measurements.measurement
Convert an uncertain number from [`MonteCarloMeasurements`](https://github.com/baggepinnen/MonteCarloMeasurements.jl) 
to [`Measurements`](https://github.com/JuliaPhysics/Measurements.jl) format 
using the median as the central point. The new `±` boundaries will include 
the 68% quantile around the median.
"""
function to_measurement(p::MonteCarloMeasurements.Particles)
        q = pquantile(p, [0.16, 0.5, 0.84])
        σ = max(q[3]-q[2],q[2]-q[1]) # take the larger one
        return Measurements.measurement(q[2],σ)
end

"""
    val(x)
Return the best estimate value for an uncertain `x`. Defaults to the `median`
for uncertain `x` represented by a (sampled) distribution. Supports 
[`MonteCarloMeasurements`](https://github.com/baggepinnen/MonteCarloMeasurements.jl)
and [`Measurements`](https://github.com/JuliaPhysics/Measurements.jl).

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
Return the median and the lower and upper error bar for the uncertain value `x` as a
`NamedTuple`. This is useful for plotting scripts. The
interval `[val - val_l, val + val_u]` represents the confidence interval at level `n*σ`,
or at probability `p`. Setting `p` overrides `n`. Supports 
[`MonteCarloMeasurements`](https://github.com/baggepinnen/MonteCarloMeasurements.jl)
and [`Measurements`](https://github.com/JuliaPhysics/Measurements.jl). The 
names in the `NamedTuple` can be changed with `name`.

**Example:**
```jldoctest
julia> results = [blocking_analysis(i:0.1:2i+20) for i in 1:3]; # mock results

julia> v = val_and_errs.(results, name="res"); # Vector of NamedTuple's with standard errors

julia> DataFrame(v)
3×3 DataFrame
 Row │ res      res_l    res_u
     │ Float64  Float64  Float64
─────┼───────────────────────────
   1 │    11.5  1.7282   1.7282
   2 │    13.0  1.7282   1.7282
   3 │    14.5  1.78885  1.78885
```

See [`NamedTuple`](@ref), [`val`](@ref), [`errs`](@ref), [`BlockingResult`](@ref),
[`RatioBlockingResult`](@ref).
"""
function val_and_errs(x; name=:val, kwargs...)
    return (; Symbol(name) => x, Symbol(name, :_l) => zero(x), Symbol(name, :_u) => zero(x))
end
function val_and_errs(
    m::T; p=nothing, n=1, name=:val
) where T <:Union{
    Measurement,
    Complex{<:Measurement},
    AbstractParticles,
    Complex{<:AbstractParticles}
}
    return _errs(p, m, n, name)
end
function _errs(::Nothing, m::Measurement, n, name)
    σ = uncertainty(m)
    return (; Symbol(name) => value(m), Symbol(name, :_l) => n*σ, Symbol(name, :_u) => n*σ)
end
function _errs(p, m::Complex, n, name)
    tr = Tuple(_errs(p, real(m), n, name))
    ti = Tuple(_errs(p, imag(m), n, name))
    nt = (;
        Symbol(name) => tr[1] +  ti[1]*im,
        Symbol(name, :_l) => tr[2] +  ti[2]*im,
        Symbol(name, :_u) => tr[3] +  ti[3]*im,
    )
    return nt
end
function _errs(p::AbstractFloat, m::Measurement, _, name)
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
function _errs(p::T, m::AbstractParticles, _, name) where T <: AbstractFloat
    cp1 = (1-p)/2
    cp2 = 1-cp1
    q1, q2, q3 = pquantile(m, (cp1, one(T)/2, cp2))
    return (;Symbol(name) => q2, Symbol(name, :_l) => q2 - q1, Symbol(name, :_u) => q3 - q2)
end
function val_and_errs(r::BlockingResult; kwargs...)
    return val_and_errs(measurement(r); kwargs...)
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

"""
    NamedTuple(x::BlockingResult; n=1, p=nothing, name=:val)
    NamedTuple(x::RatioBlockingResult; n=1, p=nothing, name=:val)
Return a named tuple with value and error bars (see [`val_and_errs`](@ref)) as well
as additional numerical fields relevant for `x`.

**Example:**
```jldoctest
julia> results = [blocking_analysis(i:0.1:2i+20) for i in 1:3]; # mock results

julia> df = NamedTuple.(results, name=:res)|>DataFrame
3×7 DataFrame
 Row │ res      res_l    res_u    res_err_err  res_p_cov  res_k  res_blocks
     │ Float64  Float64  Float64  Float64      Float64    Int64  Int64
─────┼──────────────────────────────────────────────────────────────────────
   1 │    11.5  1.7282   1.7282      0.352767    2.98667      5          13
   2 │    13.0  1.7282   1.7282      0.352767    2.98667      5          13
   3 │    14.5  1.78885  1.78885     0.350823    3.2          5          14
```
```julia-repl
julia> rbs = ratio_of_means(1 .+sin.(1:0.1:11),2 .+sin.(2:0.1:12)); # more mock results

julia> [NamedTuple(rbs),]|>DataFrame
1×9 DataFrame
 Row │ val       val_l      val_u      val_f     val_σ_f    val_δ_y    val_k  val_blocks  val_success
     │ Float64   Float64    Float64    Float64   Float64    Float64    Int64  Int64       Bool
─────┼────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.581549  0.0925669  0.0812292  0.560532  0.0875548  0.0875548      4          12         true

```

See [`val_and_errs`](@ref), [`val`](@ref), [`errs`](@ref), [`BlockingResult`](@ref),
[`RatioBlockingResult`](@ref).
"""
function Base.NamedTuple(r::BlockingResult; name=:val, kwargs...)
    vae =  val_and_errs(r; name, kwargs...)
    br_info = (;
        Symbol(name, :_err_err) => r.err_err,
        Symbol(name, :_p_cov) => r.p_cov,
        Symbol(name, :_k) => r.k,
        Symbol(name, :_blocks) => r.blocks,
    )
    return (; vae..., br_info...)
end
function Base.NamedTuple(r::RatioBlockingResult; name=:val, kwargs...)
    vae =  val_and_errs(r; name, kwargs...)
    br_info = (;
        Symbol(name, :_f) => r.f,
        Symbol(name, :_σ_f) => r.σ_f,
        Symbol(name, :_δ_y) => r.δ_y,
        Symbol(name, :_k) => r.k,
        Symbol(name, :_blocks) => r.blocks,
        Symbol(name, :_success) => r.success,
    )
    return (; vae..., br_info...)
end
