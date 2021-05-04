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
norm_project!(p::ProjectStrategy, w, args...) = norm_project!(StochasticStyle(w), p, w, args...)

norm_project!(::StochasticStyle, p, w, args...) = walkernumber(w) # MPIsync
# default, compute 1-norm
# e.g. triggered with the `NoProjection` strategy

norm_project!(::StochasticStyle, p::NoProjectionTwoNorm, w, args...) = norm(w, 2) # MPIsync
# compute 2-norm but do not perform projection

function norm_project!(s::S, p::ThresholdProject, w, args...) where S<:Union{IsStochasticWithThreshold}
    return norm_project_threshold!(w, p.threshold) # MPIsync
end
function norm_project!(s::IsDynamicSemistochastic{true}, _, w, args...)
    return norm_project_threshold!(w, s.proj_threshold)
end

function norm_project_threshold!(w, threshold)
    # MPIsync
    # perform projection if below threshold preserving the sign
    lw = localpart(w)
    for (add, val) in kvpairs(lw)
        pprob = abs(val)/threshold
        if pprob < 1 # projection is only necessary if abs(val) < s.threshold
            lw[add] = (pprob > cRand()) ? threshold*sign(val) : zero(val)
        end
    end
    return walkernumber(w) # MPIsync
end

function norm_project_threshold!(w::AbstractDVec{K,V}, threshold) where {K,V<:Union{Integer,Complex{Int}}}
    @error "Trying to scale integer based walker vector. Use float walkers!"
end

function norm_project!(s::S, p::ScaledThresholdProject, w, args...) where S<:Union{IsStochasticWithThreshold}
    f_norm = norm(w, 1) # MPIsync
    proj_norm = norm_project_threshold!(w, p.threshold)
    # MPI sycncronising
    rmul!(localpart(w), f_norm/proj_norm) # scale in order to remedy projection noise
    return f_norm
end

function norm_project!(s::IsStochasticWithThreshold,
                        p::ComplexNoiseCancellation, w,
                        shift::T, pnorm::T, dτ
    ) where T <: Complex
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

function norm_project!(s::StochasticStyle,
                        p::ComplexNoiseCancellation, args...
    )
    throw(ErrorException("`ComplexNoiseCancellation` requires complex shift in `FciqmcRunStrategy` and  `IsStochasticWithThreshold`."))
end
