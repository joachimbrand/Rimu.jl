# convenience function for extracting data from DataFrames


"""
    fidelity(rr::Tuple; p_field = :hproj, steps_equilibrate = 0)
Compute the fidelity of the average coefficient vector and the projector defined in
`p_field` from the result of replica [`fci_qmc!()`](@ref) passed as  argument `rr` (a
tuple of DataFrames). Calls [`ratio_of_means()`](@ref) to perform a blocking analysis
on a ratio of the means of separate time series and returns a
[`RatioBlockingResult`](@ref).

The fidelity of states `|ψ⟩` and `|ϕ⟩` is defined as
```math
F(ψ,ϕ) = \\frac{|⟨ψ|ϕ⟩|^2}{⟨ψ|ψ⟩⟨ϕ|ϕ⟩} .
```
Specifically, `fidelity` computes
```math
F(v,⟨c⟩) = \\frac{⟨(c₁⋅v)(v⋅c₂)⟩}{⟨c₁⋅c₂⟩} ,
```
where `v` is the projector specified by `p_field`, which is assumed to be normalised to
unity with the two-norm (i.e. `v⋅v == 1`), and `c₁` and `c₂` are two replica coefficient
vectors.
"""
function fidelity(rr::Tuple; p_field = :hproj, steps_equilibrate = 0; args...)
    df1 = rr[2][1] :: DataFrame # first replica DataFrame
    df2 = rr[2][2] :: DataFrame # second replica DataFrame
    dfr = rr[1] :: DataFrame # joint results DataFrame
    
    # numerator for fidelity calculation as time series (array)
    fid_num = conj(getproperty(df1, p_field)) .* getproperty(df2, p_field)
    fid_num = fid_num[steps_equilibrate+1:end]
    # denominator
    fid_den = dfr.xdoty[steps_equilibrate+1:end]

    return ratio_of_means(fid_num, fid_den; args...)
end

"""
    med_and_errs(p) -> (; med, err_l, err_u)
Returns the median `med` and the lower `err_l` and upper `err_u` standard error.
"""
function med_and_errs(p)
    q = quantile[0.16, 0.5, 0.84]
    med = q[2]
    err_l = med - q[1]
    err_u = q[3] - med
    return (; med, err_l, err_u)
end
