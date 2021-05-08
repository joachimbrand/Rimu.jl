# reweighting functions

"""
    w_exp(shift, h, dτ; E_r = mean(shift), skip = 0)
Compute the weights for reweighting over `h` time steps with reference energy `E_r` from
the exponetial formula
```math
w_h^{(n)} = \\prod_{j=1}^h \\exp[-dτ(S^{(q+n-j)}-E_r)] ,
```
where `q = skip`.
"""
function w_exp(shift, h, dτ; E_r = mean(shift), skip = 0)
    T = eltype(shift)
    len = length(shift)-skip
    accu = ones(T, len)
    for n in 1:len
        for j in 1:h
            accu[n] *= skip+n-j > 0 ? exp(-dτ*(shift[skip+n-j] - E_r)) : T(1)
        end
    end
    return accu
end

"""
    w_lin(shift, h, dτ; E_r = mean(shift), skip = 0)
Compute the weights for reweighting over `h` time steps with reference energy `E_r` from
the linearised formula
```math
w_h^{(n)} = \\prod_{j=1}^h [1-dτ(S^{(q+n-j)}-E_r)] ,
```
where `q = skip`.
"""
function w_lin(shift, h, dτ; E_r = mean(shift), skip = 0)
    T = eltype(shift)
    len = length(shift)-skip
    accu = ones(T, len)
    for n in 1:len
        for j in 1:h
            accu[n] *= skip+n-j > 0 ? 1-dτ*(shift[skip+n-j] - E_r) : T(1)
        end
    end
    return accu
end

"""
    function growth_bias(
        shift, wn, h, dτ;
        E_r = mean(shift),
        skip = 0,
        weights = w_exp,
        kwargs...
    ) -> (; Δ::Particles, k, blocks, success)
Compute an estimator for the bias `Δ` of the growth estimator `E_r` by the reweighting
technique described in [Umirgar *et al.* (1993)](http://dx.doi.org/10.1063/1.465195).
`shift` and `wn` are equal length
vectors containing the shift and walker number time series, respectively.
Reweighting is done over `h`
time steps and `length(shift) - skip` time steps are used for the blocking analysis done
with [`ratio_of_means`](@ref). `dτ` is the time step and `weights` a function that
calulates the weights. See [`w_exp()`](@ref) and [`w_lin()`](@ref).
```math
E_r - E_0 ≈ Δ = \\frac{1}{dτ}\\ln
    \\frac{\\sum_n w_{t+1}^{(n+1)} * N_\\mathrm{w}^{(n+1)}}
        {\\sum_m w_{t}^{(m)} * N_\\mathrm{w}^{(m)}}
```
When `h` is greater than the autocorrelation time scale of the `shift`,
then `E_r - Δ` is an unbiased but approximate estimator for the ground state energy
``E_0`` with an error ``\\mathcal{O}(dτ^2)`` and potentially increased confidence intervals.
Error propagation is done with [`MonteCarloMeasurements`](@ref). If `success==true` the
blocking analysis was successful in `k-1` steps, using `blocks` uncorrelated data points.
"""
function growth_bias(
    shift, wn, h, dτ;
    E_r = mean(shift),
    skip = 0,
    weights = w_exp,
    kwargs...
)
    # W_{t+1}^{(n+1)} .* wn^{(n+1)}
    numerator = weights(shift[2:end], h+1, dτ; E_r, skip) .* wn[skip+2:end]
    # W_{t}^{(n)} .* wn^{(n)}
    denominator = weights(shift[1:end-1], h, dτ; E_r, skip) .* wn[skip+1:end-1]
    nt = ratio_of_means(numerator, denominator; kwargs...)
    Δ = log(nt.ratio)/dτ
    return (; Δ, k=nt.k, blocks = nt.blocks, success = nt.success)
end
