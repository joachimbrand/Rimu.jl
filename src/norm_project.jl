# `norm_project!()`: helper function called from `lomc!()` and `fciqmc!()`
# dispatches on `ProjectStrategy` and `StochasticStyle`
# `norm_project_threshold!`: helper function called from `norm_project!()`

"""
    norm_project!(p_strat::ProjectStrategy, w, shift, pnorm) -> walkernumber
Compute the walkernumber of `w` and update the coefficient vector `w` according to
`p_strat`.

This may include stochastic projection of the coefficients
to `s.threshold` preserving the sign depending on [`StochasticStyle(w)`](@ref)
and `p_strat`. See [`ProjectStrategy`](@ref).
"""
norm_project!(p::ProjectStrategy, w, args...) = walkernumber(w)
# default, compute 1-norm
# e.g. triggered with the `NoProjection` strategy

function norm_project!(p::NoProjectionAccumulator, w, args...)
    for (add, val) in pairs(w)
        p.accu[add] += val
    end
    return walkernumber(w)
end

norm_project!(p::NoProjectionTwoNorm, w, args...) = norm(w, 2) # MPIsync
# compute 2-norm but do not perform projection

function norm_project!(p::ThresholdProject, w, args...)
    project_threshold!(w, p.threshold) # MPIsync
    return walkernumber(w)
end


function project_threshold!(
    w::AbstractDVec{<:Any,V}, threshold
) where {K,V<:Union{Integer,Complex{<:Integer}}}
    error("Trying to scale integer based walker vector. Use float walkers!")
end

function norm_project!(p::ScaledThresholdProject, w, args...)
    f_norm = norm(w, 1) # MPIsync
    project_threshold!(w, p.threshold)
    proj_norm = walkernumber(w)
    # MPI sycncronising
    rmul!(localpart(w), f_norm/proj_norm) # scale in order to remedy projection noise
    return f_norm
end

function norm_project!(p::ComplexNoiseCancellation, w, shift::Complex, pnorm::Complex, dτ)
    f_norm = norm(w, 1)::Real # MPIsync
    im_factor = dτ*imag(shift) + p.κ*√dτ*sync_cRandn(w) # MPIsync
    # Wiener increment
    # do we need to synchronize such that we add the same noise on each MPI rank?
    # or thread ? - not thread, as threading is done inside fciqmc_step!()
    scale_factor = 1 - im_factor*imag(pnorm)/f_norm
    rmul!(localpart(w), scale_factor) # scale coefficient vector
    c_im = f_norm/real(pnorm)*imag(pnorm) + im_factor*real(pnorm)
    return complex(f_norm*scale_factor, c_im) |> T # return complex norm
end

function norm_project!(p::ComplexNoiseCancellation, args...)
    error("`ComplexNoiseCancellation` requires complex shift in `FciqmcRunStrategy`.")
end

"""
    update_dvec!(v, pnorm, dτ, step)
    update_dvec!(::StochasticStyle, v, pnorm, dτ, step)

Optional function to used to update the vector every step. Called after `fciqmc_step!`, but
before `norm_project!`.
"""
update_dvec!(v, args...) = update_dvec!(StochasticStyle(v), args...)
update_dvec!(::StochasticStyle, v, _, _, _) = v

function update_dvec!(s::IsDynamicSemistochastic{true}, v, _, _, _)
    project_threshold!(v, s.proj_threshold)
    return v
end

function update_dvec!(s::IsStochasticWithThreshold, v, _, _, _)
    project_threshold!(w, s.threshold)
    return v
end
