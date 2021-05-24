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

function apply_memory_noise!(ws::NTuple, args...)
    apply_memory_noise!(StochasticStyle(first(ws)), ws, args...)
end

function apply_memory_noise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::NoMemory)
    return 0.0 # does nothing
end

function apply_memory_noise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::MemoryStrategy)
    throw(ErrorException("MemoryStrategy `$(typeof(m))` does not work with StochasticStyle `$(typeof(s))`. Ignoring memory noise for now."))
    # @error "MemoryStrategy `$(typeof(m))` does not work with StochasticStyle `$(typeof(s))`. Ignoring memory noise for now." maxlog=2
    return 0.0 # default prints an error message
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
