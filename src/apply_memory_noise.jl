# `apply_memory_noise!()`: helper function called from `fciqmc_step!()`
# dispatches on `MemoryStrategy` and `StochasticStyle`
# `purge_negative_walkers!()`: helper function called from `apply_memory_noise!()`

"""
    r = apply_memory_noise!(w, v, shift, dτ, pnorm, m_strat::MemoryStrategy)
Apply memory noise to `w`, i.e. `w .+= r.*v`, computing the noise `r` according
to `m_strat`. Note that `m_strat`
needs to be compatible with `StochasticStyle(w)`. Otherwise, an
error exception is thrown. See [`MemoryStrategy`](@ref).

`w` is the walker array after fciqmc step, `v` the previous one, `pnorm` the
norm of `v`, and `r` the instantaneously applied noise.
"""
function apply_memory_noise!(w::Union{AbstractArray{T},AbstractDVec{K,T}},
         v, shift, dτ, pnorm, m
    ) where  {K,T<:Real}
    apply_memory_noise!(StochasticStyle(w), w, v, real(shift), dτ, real(pnorm), m)
end
# only use real part of the shift and norm if the coefficients are real

# otherwise, pass on complex shift in generic method
function apply_memory_noise!(w::Union{AbstractArray,AbstractDVec}, args...)
    apply_memory_noise!(StochasticStyle(w), w, args...)
end

function apply_memory_noise!(ws::NTuple{NT,W}, args...) where {NT,W}
    apply_memory_noise!(StochasticStyle(W), ws, args...)
end

function apply_memory_noise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::NoMemory)
    return 0.0 # does nothing
end

function apply_memory_noise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::MemoryStrategy)
    throw(ErrorException("MemoryStrategy `$(typeof(m))` does not work with StochasticStyle `$(typeof(s))`. Ignoring memory noise for now."))
    # @error "MemoryStrategy `$(typeof(m))` does not work with StochasticStyle `$(typeof(s))`. Ignoring memory noise for now." maxlog=2
    return 0.0 # default prints an error message
end

function apply_memory_noise!(::StochasticStyle, w, _, _, _, _, m::ConstrainedNoise)
    lw = localpart(w) # just to make sure - normally `w` would be a local DVec anyway
    # current norm of `w` per population after FCIQMC step
    return apply_constrained_noise!(lw, Norm1ProjectorPPop()⋅lw, m)
end

function apply_constrained_noise!(w::AbstractDVec{<:Any,V}, _, _) where V
    throw(ErrorException("`ConstrainedNoise` on walkers with type $V is not implemented."))
end

function apply_constrained_noise!(w::AbstractDVec{<:Any,<:Integer}, nwalkers, m)
    # define three integer counters
    n_pool = Int(nwalkers) # pool of remaining random psips
    n_m = trunc(Int, m.α/2*n_pool) # number of remaining `-1` random psips
    m.α/2*n_pool-n_m > cRand() && (n_m += 1) # stochastic rounding to Int
    n_mp = 2*n_m # number of remaining `+1` or `-1` random psips
    # making sure that we get the exact same number of `+1` random psips
    # @show n_m, n_mp, n_pool

    for (add, num) in pairs(w)
        noise = 0
        for i in 1:abs(num)
            r = cRand(1:n_pool)
            if r ≤ n_m
                noise -= 1
                n_m -= 1 # one `-1` psip is used up
                n_mp -= 1 # all counters need to be reduced
            elseif r ≤ n_mp
                noise += 1
                n_mp -= 1 # one `+1` psip is used up
            end
            n_pool -= 1 # one psip from the pool was used
        end
        w[add] = num + sign(num) * noise
    end
    return 0.0
end

function apply_constrained_noise!(w::AbstractDVec{<:Any,<:Complex{<:Integer}}, nwalkers, m)
    # define three complex-valued counters
    n_pool = nwalkers
    n_m_approx = m.α/2*n_pool
    n_m = round(n_m_approx, RoundToZero)
    real(n_m_approx) - real(n_m) > cRand() && (n_m += 1) # stochastic rounding to Int
    imag(n_m_approx) - imag(n_m) > cRand() && (n_m += im) # stochastic rounding to Int
    n_mp = 2*n_m # number of remaining `+1` or `-1` random psips
    # @show n_m, n_mp, n_pool # should all have complex non-negative integer values

    for (add, num) in pairs(w)
        noise = 0 + 0im
        for i in 1:abs(real(num))
            r = cRand(1:real(n_pool))
            if r ≤ real(n_m)
                noise -= 1
                n_m -= 1 # one `-1` psip is used up
                n_mp -= 1 # all counters need to be reduced
            elseif r ≤ real(n_mp)
                noise += 1
                n_mp -= 1 # one `+1` psip is used up
            end
            n_pool -= 1 # one psip from the pool was used
        end
        for i in 1:abs(imag(num))
            r = cRand(1:imag(n_pool))
            if r ≤ imag(n_m)
                noise -= im
                n_m -= im # one `-1` psip is used up
                n_mp -= im # all counters need to be reduced
            elseif r ≤ imag(n_mp)
                noise += im
                n_mp -= im # one `+1` psip is used up
            end
            n_pool -= im # one psip from the pool was used
        end
        w[add] = num + sign(real(num)) * real(noise) + im * sign(imag(num)) * imag(noise)
    end
    return 0.0
end

function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::DeltaMemory)
    tnorm = norm(w, 1) # MPIsync
    # current norm of `w` after FCIQMC step
    # compute memory noise
    r̃ = (pnorm - tnorm)/(dτ*pnorm) + shift
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return dτ*r
end

function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::DeltaMemory2)
    tnorm = norm(w, 1) # MPIsync
    # current norm of `w` after FCIQMC step
    # compute memory noise
    r̃ = pnorm - tnorm + shift*dτ*pnorm
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = (r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer))/(dτ*pnorm)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return dτ*r
end

function apply_memory_noise!(s::IsStochasticWithThreshold,
    w, v, shift, dτ, pnorm, m::DeltaMemory3)
tnorm = norm(w, 1) # MPIsync
# current norm of `w` after FCIQMC step
# compute memory noise
r̃ = (pnorm - tnorm)/pnorm + dτ*shift
push!(m.noiseBuffer, r̃) # add current value to buffer
# Buffer only remembers up to `Δ` values. Average over whole buffer.
r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

# apply `r` noise to current state vector
rmul!(w, 1 + m.level * r) # w = w * (1 + level*r)

return r
end

function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ShiftMemory)
    push!(m.noiseBuffer, shift) # add current value of `shift` to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = - shift + sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return dτ*r
end

function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI

    # current projection of `w` after FCIQMC step
    pp  = m.pp
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp)/pp + shift*dτ
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(r, v, w) # w .+= r .* v
    # TODO: make this work with multithreading
    m.pp = tp + r*pp # update previous projection
    return r
end

# This one works to remove the bias when projection is done with exact
# eigenvector
function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory2)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI
    # current projection of `w` after FCIQMC step

    pp  = m.projector⋅v
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp)/pp + shift*dτ
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(r, v, w) # w .+= r .* v
    # TODO: make this work with multithreading
    m.pp = tp + r*pp # update previous projection
    return r
end

# seems to not be effective
function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory3)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI
    # current projection of `w` after FCIQMC step

    pp  = m.projector⋅v
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp) + shift*dτ*pp
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)
    if true # abs(pp) > 0.01
        r = r/pp
        # apply `r` noise to current state vector
        axpy!(r, v, w) # w .+= r .* v
        # TODO: make this work with multithreading
        m.pp = tp + r*pp # update previous projection
    else
        r = 0.0
        m.pp = tp
    end
    return r
end

# this one does not work well - no bias correction achieved
function apply_memory_noise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory4)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI
    # current projection of `w` after FCIQMC step

    pp  = m.projector⋅v
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp) + shift*dτ*pp
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)
    sf = 0.2
    r = sf*tanh(r/pp/sf)
    # apply `r` noise to current state vector
    axpy!(r, v, w) # w .+= r .* v
    # TODO: make this work with multithreading
    m.pp = tp + r*pp # update previous projection
    return r
end

function apply_memory_noise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::PurgeNegatives)
    purge_negative_walkers!(w)
    return 0.0
end

function purge_negative_walkers!(w::AbstractDVec{K,V}) where {K,V <:Real}
    for (k,v) in pairs(w)
        if v < 0
            delete!(w,k)
        end
    end
    return w
end
function purge_negative_walkers!(w::AbstractDVec{K,V}) where {K,V <:Complex}
    for (k,v) in pairs(w)
        if real(v) < 0
            v = 0 + im*imag(v)
        end
        if imag(v) < 0
            v = real(v)
        end
        w[k] = convert(V,v)
    end
    return w
end
