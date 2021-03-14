module UglyHacks

using LinearAlgebra
using DataFrames
using Rimu
using Rimu: fciqmc_step!
using Rimu.DictVectors: @delegate

export UglyHack

mutable struct UglyHack{K,V,D<:DVec2{K,V},A<:DVec2{K},P,S,T,R,PS} <: AbstractDVec{K,V}
    ca::D

    # Data needed to keep track of the cb state
    cb::D
    pnorm::Float64
    shift::Float64
    shiftMode::Bool
    params::P
    s_strat::S
    τ_strat::T
    r_strat::R
    p_strat::PS
    w::D

    alpha::Float64
    beta::Float64
    threshold::Float64
    A::A
    B::A
    Ã::A

    mul_with_ham::Bool
    accumulate_shift::Bool
    df::DataFrame
end

function UglyHack(
    ham, v::AbstractDVec{K,V};
    params, s_strat, τ_strat, r_strat, p_strat,
    decay_time=500,
    alpha=1 - 1/decay_time,
    beta=1.0,
    #norm_factor=1,
    threshold=0,
    accumulate_shift=false,
    mul_with_ham=false,
) where {K,V}
    r_strat = Rimu.refine_r_strat(r_strat, ham)
    pnorm = Float64(walkernumber(v))

    df = DataFrame(
        overlapAb=Float64[],
        overlapaB=Float64[],
        overlapAB=Float64[],
        shift=Float64[]
    )
    if accumulate_shift
        df.overlapÃB = Float64[]
    end
    if mul_with_ham
        df.overlapAHB = Float64[]
    end
    cap = ceil(Int, beta/(1 - alpha) * capacity(v))

    return UglyHack(
        v,
        copy(v),
        Float64(pnorm),
        params.shift,
        params.shiftMode,
        deepcopy(params),
        s_strat,
        τ_strat,
        r_strat,
        p_strat,
        copy(v),

        alpha,
        beta,
        threshold,
        DVec2{K,Float64}(cap),
        DVec2{K,Float64}(cap),
        DVec2{K,Float64}(cap),

        mul_with_ham,
        accumulate_shift,
        df,
    )
end

function Base.empty!(a::UglyHack)
    empty!(a.ca)
end

for f in (:length, :empty, :iterate, :pairs, :similar, :getindex, :setindex!)
    @eval Base.$f(a::UglyHack, args...; kwargs...) = $f(a.ca, args...; kwargs...)
end

Rimu.StochasticStyle(a::UglyHack) = Rimu.StochasticStyle(a.ca)
Rimu.capacity(a::UglyHack) = Rimu.capacity(a.ca)

# accumulator += beta * v + alpha * accumulator
function update_accumulator!(accumulator, v, alpha, beta, threshold)
    rmul!(accumulator, alpha)
    axpy!(beta, v, accumulator)
    if threshold > 0
        Rimu.norm_project_threshold!(accumulator, threshold)
    end
    return accumulator
end

function update!(ham, a::UglyHack{K,V,D}, dτ, shift_a, m=1.0, m_strat=NoMemory()) where {K,V,D}
    # Start by handling the accumulator
    # This makes sure both ca and cb are at the same time step.
    # death
    update_accumulator!(a.A, a.ca, a.alpha, a.beta, a.threshold)
    update_accumulator!(a.B, a.cb, a.alpha, a.beta, a.threshold)

    if a.accumulate_shift
        update_accumulator!(a.Ã, a.ca, a.alpha, a.beta * shift_a, a.threshold)
    end

    # report overlaps
    push!(a.df.overlapAb, dot(a.A, a.cb))
    push!(a.df.overlapaB, dot(a.ca, a.B))
    push!(a.df.overlapAB, dot(a.A, a.B))
    if a.accumulate_shift
        push!(a.df.overlapÃB, dot(a.Ã, a.B))
    end
    if a.mul_with_ham
        push!(a.df.overlapAHB, dot(a.A, ham, a.B))
    end
    push!(a.df.shift, a.shift)

    # Take care of the step
    v = a.cb
    w = a.w
    pnorm = a.pnorm
    r_strat = a.r_strat
    p_strat = a.p_strat
    s_strat = a.s_strat
    shiftMode = a.shiftMode
    shift = a.shift

    v, w, _, r = fciqmc_step!(
        ham, v, shift, dτ, pnorm, w, m; m_strat=m_strat
    )
    tnorm = Rimu.norm_project!(p_strat, v, shift, pnorm, dτ) |> Float64
    v_proj, h_proj = Rimu.compute_proj_observables(v, ham, r_strat)

    # update shift and mode if necessary
    shift, shiftMode, pnorm = Rimu.update_shift(
        s_strat, shift, shiftMode, tnorm, pnorm, dτ, step, nothing, v, w
    )

    a.shift = Float64(shift)
    a.shiftMode = Bool(shiftMode)
    a.pnorm = Float64(pnorm)
    a.cb = v
    a.w = w
    return nothing
end

function Rimu.fciqmc_step!(
    H, a::UglyHack, shift, dτ, pnorm, w, m::Float64=1.0; m_strat=NoMemory()
)
    update!(H, a, dτ, shift, m, m_strat)
    v, w, stats, r = fciqmc_step!(H, a.ca, shift, dτ, pnorm, w, m; m_strat)
    a.ca = v
    return a, w, stats, r
end

#function Rimu.fciqmc_step!(_, ::UglyHack, _, _, _, ::NTuple, ::Float64; m_strat=nothing)
function Rimu.fciqmc_step!(a, ::UglyHack, b, c, d, ::NTuple, ::Float64; m_strat=nothing)
    error("please don't use threading")
end

end
