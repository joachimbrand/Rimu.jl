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
    fid_den = df.xdoty[skip+1:end]

    return ratio_of_means(fid_num, fid_den; args...)
end

"""
    med_and_errs(p) -> (; med, err1_l, err1_u, err2_l, err2_u)
Convenience function for extracting plottable data from a distribution or
[`Particles`](@ref) object.
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
    q = quantile(p, [0.025, 0.16, 0.5, 0.84, 0.975])
    med = q[3]
    err1_l = med - q[2]
    err1_u = q[4] - med
    err2_l = med - q[1]
    err2_u = q[5] - med
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
