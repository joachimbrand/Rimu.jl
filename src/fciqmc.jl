
"""
    fciqmc!(v, pa::FciqmcRunStrategy, [df,]
             ham, s_strat::ShiftStrategy,
             [r_strat::ReportingStrategy, τ_strat::TimeStepStrategy, w])
    -> df

Perform the FCIQMC algorithm for determining the lowest eigenvalue of `ham`.
`v` can be a single starting vector of type `:<AbstractDVec` or a vector
of such structures. In the latter case, independent replicas are constructed.
Returns a `DataFrame` `df` with statistics about the run, or a tuple of `DataFrame`s
for a replica run.
Strategies can be given for updating the shift (see [`ShiftStrategy`](@ref))
and (optionally), for reporting (see [`ReportingStrategy`](@ref)),
and for updating the time step `dτ` (see [`TimeStepStrategy`](@ref)).

A pre-allocated data structure `w` for working memory can be passed as argument,
and controls multi-threading behaviour. By default multi-threading is turned
on. To turn multi-threading off, pass `similar(localpart(v))` for w.

This function mutates `v`, the parameter struct `pa` as well as
`df`, and `w`.
"""
function fciqmc!(svec, pa::FciqmcRunStrategy,
                 ham,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 w = threaded_working_memory(svec); kwargs...)
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa
    len = length(svec) # MPIsync
    nor = norm(svec, 1) # MPIsync
    # prepare df for recording data
    df = DataFrame(steps=Int[], dτ=Float64[], shift=Float64[],
                        shiftMode=Bool[],len=Int[], norm=Float64[],
                        spawns=Int[], deaths=Int[], clones=Int[],
                        antiparticles=Int[], annihilations=Int[])
    # Note the row structure defined here (currently 11 columns)
    # When changing the structure of `df`, it has to be changed in all places
    # where data is pushed into `df`.
    @assert names(df) == [:steps, :dτ, :shift, :shiftMode, :len, :norm,
                            :spawns, :deaths, :clones, :antiparticles,
                            :annihilations
                         ] "Column names in `df` not as expected."
    # Push first row of df to show starting point
    push!(df, (step, dτ, shift, shiftMode, len, nor,
                0, 0, 0, 0, 0))
    # println("DataFrame is set up")
    # # (DD <: MPIData) && println("$(svec.s.id): arrived at barrier; before")
    # (DD <: MPIData) && MPI.Barrier(svec.s.comm)
    # # println("after barrier")
    rdf =  fciqmc!(svec, pa, df, ham, s_strat, r_strat, τ_strat, w
                    # )
                    ; kwargs...)
    # # (DD <: MPIData) && println("$(svec.s.id): arrived at barrier; after")
    # (DD <: MPIData) && MPI.Barrier(svec.s.comm)
    return rdf
end

# for continuation runs we can also pass a DataFrame
function fciqmc!(v, pa::RunTillLastStep, df::DataFrame,
                 ham,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 w = threaded_working_memory(v)
                 ; m_strat::MemoryStrategy = NoMemory(),
                 p_strat::ProjectStrategy = NoProjection()
                 )
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa

    # check `df` for consistency
    @assert names(df) == [:steps, :dτ, :shift, :shiftMode, :len,
                            :norm, :spawns, :deaths, :clones, :antiparticles,
                            :annihilations
                         ] "Column names in `df` not as expected."

    svec = v # keep around a reference to the starting data container
    pnorm = tnorm = norm(v, 1) # norm of "previous" vector
    maxlength = capacity(localpart(v))
    @assert maxlength ≤ capacity(w) "`w` needs to have at least `capacity(v)`"

    while step < laststep
        step += 1
        # println("Step: ",step)
        # perform one complete stochastic vector matrix multiplication
        v, w, step_stats = fciqmc_step!(ham, v, shift, dτ, pnorm, w;
                                        m_strat=m_strat)
        tnorm = norm_project!(v, p_strat)  # MPIsync
        # project coefficients of `w` to threshold
        # tnorm = norm(v, 1) # MPI sycncronising: total number of psips
        # tnorm = apply_memory_noise!(v, w, s_strat, pnorm, tnorm, shift, dτ)
        # update shift and mode if necessary
        shift, shiftMode, pnorm = update_shift(s_strat,
                                    shift, shiftMode,
                                    tnorm, pnorm, dτ, step, df)
        # the updated "previous" norm pnorm is returned from `update_shift()`
        # in order to allow delaying the update, e.g. with `DelayedLogUpdate`
        # pnorm = tnorm # remember norm of this step for next step (previous norm)
        dτ = update_dτ(τ_strat, dτ, tnorm) # will need to pass more information later
        # when we add different stratgies
        len = length(v) # MPI sycncronising: total number of configs
        # record results according to ReportingStrategy r_strat
        report!(df, (step, dτ, shift, shiftMode, len, tnorm,
                        step_stats...), r_strat)
        # DF ≠ Nothing && push!(df, (step, dτ, shift, shiftMode, len, tnorm,
        #                 step_stats...))
        # housekeeping: avoid overflow of dvecs
        len_local = length(localpart(v))
        len_local > 0.8*maxlength && if len_local > maxlength
            @error "`maxlength` exceeded" len_local maxlength
            break
        else
            @warn "`maxlength` nearly reached" len_local maxlength
        end
    end
    # make sure that `svec` contains the current population:
    if !(v === svec)
        copyto!(svec, v)
    end
    # pack up parameters for continuation runs
    # note that this modifes the struct pa
    @pack! pa = step, shiftMode, shift, dτ
    return  df
    # note that `svec` and `pa` are modified but not returned explicitly
end # fciqmc

# replica version
function fciqmc!(vv::Vector, pa::RunTillLastStep, ham::LinearOperator,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 wv = threaded_working_memory.(vv) # wv = similar.(localpart.(vv))
                 ; m_strat::MemoryStrategy = NoMemory(),
                 p_strat::ProjectStrategy = NoProjection())
    # τ_strat is currently ignored in the replica version
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa
    V = valtype(vv[1])
    N = length(vv) # number of replicas to propagate
    # keep references to the passed data vectors around
    vv_orig = similar(vv) # empty vector
    vv_orig .= vv # fill it with references to the coefficient DVecs

    maxlength = minimum(capacity.(vv))
    reduce(&, capacity.(wv) .≥ maxlength) || error("replica containers `wv` have insufficient capacity")

    shifts = [shift for i = 1:N] # Vector because it needs to be mutable
    vShiftModes = [shiftMode for i = 1:N] # separate for each replica
    pnorms = zeros(N) # initialise as vector
    pnorms .= norm.(vv,1) # 1-norm i.e. number of psips as Tuple (of previous step)

    # initalise df for storing results of each replica separately
    dfs = Tuple(DataFrame(steps=Int[], shift=Float64[], shiftMode=Bool[],
                         len=Int[], norm=Float64[], spawns=V[], deaths=V[],
                         clones=V[], antiparticles=V[],
                         annihilations=V[]) for i in 1:N)
    # dfs is thus an NTuple of DataFrames
    for i in 1:N
        push!(dfs[i], (step, shifts[i], vShiftModes[i], length(vv[i]),
                      pnorms[i], 0, 0, 0, 0, 0))
    end

    # prepare `DataFrame` for variational ground state estimator
    # we are assuming that N ≥ 2, otherwise this will fail
    PType = promote_type(V,eltype(ham)) # type of scalar product
    RType = promote_type(PType,Float64) # for division
    mixed_df= DataFrame(steps =Int[], xdoty =V[], xHy =PType[], aveH =RType[])
    dp = vv[1]⋅vv[2] # <v_1 | v_2>
    expval =  vv[1]⋅ham(vv[2]) # <v_1 | ham | v_2>
    push!(mixed_df,(step, dp, expval, expval/dp))

    norms = zeros(N)
    mstats = [zeros(Int,5) for i=1:N]
    while step < laststep
        step += 1
        for (i, v) in enumerate(vv) # loop over replicas
            # perform one complete stochastic vector matrix multiplication
            vv[i], wv[i], stats = fciqmc_step!(ham, v, shifts[i], dτ, pnorms[i],
                                        wv[i]; m_strat = m_strat)
            mstats[i] .= stats
            norms[i] = norm_project!(vv[i], p_strat)  # MPIsync
            shifts[i], vShiftModes[i], pnorms[i] = update_shift(
                s_strat, shifts[i], vShiftModes[i],
                norms[i], pnorms[i], dτ, step, dfs[i]
            )
        end #loop over replicas
        lengths = length.(vv)
        # update time step
        dτ = update_dτ(τ_strat, dτ) # will need to pass more information
        # later when we add different stratgies
        # record results
        for i = 1:N
            push!(dfs[i], (step, shifts[i], vShiftModes[i], lengths[i],
                  norms[i], mstats[i]...))
        end
        v1Dv2 = vv[1]⋅vv[2] # <v_1 | v_2> overlap
        v2Dhv2 =  vv[1]⋅ham(vv[2]) # <v_1 | ham | v_2>
        push!(mixed_df,(step, v1Dv2, v2Dhv2, v2Dhv2/v1Dv2))

        # prepare for next step:
        # pnorms .= norms # remember norm of this step for next step (previous norm)
        llength = maximum(lengths)
        llength > 0.8*maxlength && if llength > maxlength
            @error "`maxlength` exceeded" llength maxlength
            break
        else
            @warn "`maxlength` nearly reached" llength maxlength
        end

    end # while step
    # make sure that `svecs` contains the current population:
    for i = 1:N
        if vv[i] ≢ vv_orig[i]
            copyto!(vv_orig[i], vv[i])
        end
    end
    # pack up and parameters for continuation runs
    # note that this modifes the struct pa
    shiftMode = reduce(&,vShiftModes) # only true if all are in vShiftMode
    shift = reduce(+,shifts)/N # return average value of shift
    @pack! pa = step, shiftMode, shift, dτ

    return mixed_df, dfs # return dataframes with stats
    # note that `vv_orig` and `pa` are modified but not returned explicitly
end # fciqmc

"""
    fciqmc_step!(Ĥ, v, shift, dτ, pnorm, w;
                          m_strat::MemoryStrategy = NoMemory()) -> ṽ, w̃, stats
Perform a single matrix(/operator)-vector multiplication:
```math
\\tilde{v} = [1 - dτ(\\hat{H} - S)]⋅v ,
```
where `Ĥ == ham` and `S == shift`. Whether the operation is performed in
stochastic, semistochastic, or determistic way is controlled by the trait
`StochasticStyle(w)`. See [`StochasticStyle`](@ref). `w` is a local data
structure with the same size and type as `v` and used for working. Both `v` and
`w` are modified.

Returns the result `ṽ`, a (possibly changed) reference to working memory `w̃`,
 and the array
`stats = [spawns, deaths, clones, antiparticles, annihilations]`. Stats will
contain zeros when running in deterministic mode.
"""
function fciqmc_step!(Ĥ, v::D, shift, dτ, pnorm, w::D;
                      m_strat::MemoryStrategy = NoMemory()) where D
    # serial version
    @assert w ≢ v "`w` and `v` must not be the same object"
    stats = zeros(Int, 5) # pre-allocate array for stats
    zero!(w) # clear working memory
    for (add, num) in pairs(v)
        res = fciqmc_col!(w, Ĥ, add, num, shift, dτ)
        stats .+= res # just add all stats together
    end
    applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
    # norm_project!(w, p_strat) # project coefficients of `w` to threshold
    # thresholdProject!(w, v, shift, dτ, m_strat) # apply walker threshold if applicable
    return w, v, stats
    # stats == [spawns, deaths, clones, antiparticles, annihilations]
end # fciqmc_step!

function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, ws::NTuple{NT,W};
      m_strat::MemoryStrategy = NoMemory(),
      batchsize = max(100, min(length(dv)÷Threads.nthreads(), round(Int,sqrt(length(dv))*10)))
    ) where {NT,W}
    # println("batchsize ",batchsize)
    # multithreaded version; should also work with MPI
    @assert NT == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    statss = [zeros(Int,5) for i=1:NT]
    # [zeros(valtype(v), 5), for i=1:NT] # pre-allocate array for stats
    zero!.(ws) # clear working memory
    @sync for btr in Iterators.partition(pairs(v), batchsize)
        Threads.@spawn statss[Threads.threadid()] .+= sum(btr) do tup
            (add, num) = tup
            fciqmc_col!(ws[Threads.threadid()], Ĥ, add, num, shift, dτ)
        end
    end # all threads have returned; now running on single thread again
    applyMemoryNoise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return sort_into_targets!(dv, ws, statss) # MPI syncronizing
    # dv, w, stats
    # stats == [spawns, deaths, clones, antiparticles, annihilations]
end # fciqmc_step!

function Rimu.fciqmc_step!(Ĥ, dv::MPIData{D,S}, shift, dτ, pnorm, w::D;
                           m_strat::MemoryStrategy = NoMemory()) where {D,S}
    # MPI version, single thread
    v = localpart(dv)
    @assert w ≢ v "`w` and `v` must not be the same object"
    empty!(w)
    stats = zeros(Int, 5) # pre-allocate array for stats
    for (add, num) in pairs(v)
        res = Rimu.fciqmc_col!(w, Ĥ, add, num, shift, dτ)
        stats .+= res # just add all stats together
    end
    applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
    # thresholdProject!(w, v, shift, dτ, m_strat) # apply walker threshold if applicable
    sort_into_targets!(dv, w)
    MPI.Allreduce!(stats, +, dv.comm) # add stats of all ranks
    return dv, w, stats
    # returns the structure with the correctly distributed end
    # result `dv` and cumulative `stats` as an array on all ranks
    # stats == (spawns, deaths, clones, antiparticles, annihilations)
end # fciqmc_step!


"""
    norm_project!(w, p_strat::ProjectStrategy) -> norm
Computes the 1-norm of `w`.
Project all elements of `w` to `s.threshold` preserving the sign if
`StochasticStyle(w)` requires projection according to `p_strat`.
See [`ProjectStrategy`](@ref).
"""
norm_project!(w, p) = norm_project!(StochasticStyle(w), w, p)

norm_project!(::StochasticStyle, w, p) = norm(w, 1) # MPIsync
# default, compute 1-norm

function norm_project!(s::S, w, p::ThresholdProject) where S<:Union{IsStochasticWithThreshold}
    return norm_project_threshold!(w, p.threshold) # MPIsync
end

function norm_project_threshold!(w, threshold) # MPIsync
    # perform projection if below threshold preserving the sign
    lw = localpart(w)
    for (add, val) in kvpairs(lw)
        pprob = abs(val)/threshold
        if pprob < 1 # projection is only necessary if abs(val) < s.threshold
            lw[add] = (pprob > cRand()) ? threshold*sign(val) : zero(val)
        end
    end
    return norm(w, 1) # MPIsync
end

function norm_project!(s::S, w, p::ScaledThresholdProject) where S<:Union{IsStochasticWithThreshold}
    f_norm = norm(w, 1) # MPIsync
    proj_norm = norm_project_threshold!(w, p.threshold)
    # MPI sycncronising
    rmul!(w, f_norm/proj_norm) # scale in order to remedy projection noise
    # TODO: MPI version of rmul!()
    return f_norm
end

"""
    applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat::MemoryStrategy)
Apply memory noise to `w` according to the strategy `m_strat`. Note that the
strategy needs to be compatible with `StochasticStyle(w)`. The default is to
not add memory noise. See [`MemoryStrategy`](@ref).

`w` is the walker array after fciqmc step, `v` the previous one, `pnorm` the
norm of `v`.
"""
function applyMemoryNoise!(w::Union{AbstractArray,AbstractDVec}, args...)
    applyMemoryNoise!(StochasticStyle(w), w, args...)
end

function applyMemoryNoise!(ws::NTuple{NT,W}, args...) where {NT,W}
    applyMemoryNoise!(StochasticStyle(W), ws, args...)
end

function applyMemoryNoise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::MemoryStrategy)
    return w # default does nothing
end

function applyMemoryNoise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::DeltaMemory)
    @warn "`DeltaMemory` was selected. It does not work with `$(typeof(s))` but requires `IsStochasticWithThreshold`. Ignoring memory noise for now." maxlog=10
    return w # default does nothing
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
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

    return w
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
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

    return w
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ShiftMemory)
    push!(m.noiseBuffer, shift) # add current value of `shift` to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = - shift + sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return w
end

# to do: implement parallel version
# function fciqmc_step!(w::D, ham::LinearOperator, v::D, shift, dτ) where D<:DArray
#   check that v and w are compatible
#   for each worker
#      call fciqmc_step!()  on respective local parts
#      sort and consolidate configurations to where they belong
#      communicate via RemoteChannels
#   end
#   return statistics
# end


# struct MySSVec{T} <: AbstractVector{T}
#     v::Vector{T}
#     sssize::Int
# end
# Base.size(mv::MySSVec) = size(mv.v)
# Base.getindex(mv::MySSVec, I...) = getindex(mv.v, I...)
#
# StochasticStyle(::Type{<:MySSVec}) = IsSemistochastic()

# Here is a simple function example to demonstrate that functions can
# dispatch on the trait and that computation of the type is done at compiler
# level.
#
# ```julia
# tfun(v) = tfun(v, StochasticStyle(v))
# tfun(v, ::IsDeterministic) = 1
# tfun(v, ::IsStochastic) = 2
# tfun(v, ::IsSemistochastic) = 3
# tfun(v, ::Any) = 4
#
# b = [1, 2, 3]
# StochasticStyle(b)
# IsStochastic()

# julia> @code_llvm tfun(b)
#
# ;  @ /Users/brand/git/juliamc/scripts/fciqmc.jl:448 within `tfun'
# define i64 @julia_tfun_13561(%jl_value_t addrspace(10)* nonnull align 16 dereferenceable(40)) {
# top:
#   ret i64 2
# }
# ```

"""
    fciqmc_col!(w, ham, add, num, shift, dτ)
    fciqmc_col!(::Type{T}, args...)
    -> spawns, deaths, clones, antiparticles, annihilations
Spawning and diagonal step of FCIQMC for single column of `ham`. In essence it
computes

`w .+= (1 .+ dτ.*(shift .- ham[:,add])).*num`.

Depending on `T == `[`StochasticStyle(w)`](@ref), a stochastic or deterministic algorithm will
be chosen. The possible values for `T` are:

- [`IsDeterministic()`](@ref) deteministic algorithm
- [`IsStochastic()`](@ref) stochastic version where the changes added to `w` are purely integer, according to the FCIQMC algorithm
- [`IsStochasticNonlinear(c)`](@ref) stochastic algorithm with nonlinear diagonal
- [`IsSemistochastic()`](@ref) semistochastic version: TODO
"""
fciqmc_col!(w::Union{AbstractArray,AbstractDVec}, args...) = fciqmc_col!(StochasticStyle(w), w, args...)

# generic method for unknown trait: throw error
fciqmc_col!(::Type{T}, args...) where T = throw(TypeError(:fciqmc_col!,
    "first argument: trait not recognised",StochasticStyle,T))

function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    w .+= (1 .+ dτ.*(shift .- view(ham,:,add))).*num
    # todo: return something sensible
    return zeros(Int, 5)
end

function fciqmc_col!(::IsDeterministic, w, ham::LinearOperator, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in Hops(ham, add)
        w[nadd] += -dτ * elem * num
    end
    # diagonal death or clone
    w[add] += (1 + dτ*(shift - diagME(ham,add)))*num
    return zeros(Int, 5)
end

# fciqmc_col!(::IsStochastic,  args...) = inner_step!(args...)
# function inner_step!(w, ham::LinearOperator, add, num::Number,
#                         shift, dτ)
function fciqmc_col!(::IsStochastic, w, ham::LinearOperator, add, num::Real,
                        shift, dτ)
    # version for single population of integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(ham,add)
    pd = dτ * (dME - shift)
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    w[add] += ndiags # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return [spawns, deaths, clones, antiparticles, annihilations]
    # note that w is not returned
end # inner_step!

function fciqmc_col!(nl::IsStochasticNonlinear, w, ham::LinearOperator, add, num::Real,
                        shift, dτ)
    # version for single population of integer psips
    # Nonlinearity in diagonal death step according to Ali's suggestion
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(ham,add)
    shifteff = shift*(1 - exp(-num/nl.c))
    pd = dτ * (dME - shifteff)
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    w[add] += ndiags # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return [spawns, deaths, clones, antiparticles, annihilations]
    # note that w is not returned
end # inner_step!

function fciqmc_col!(::IsStochastic, w, ham::LinearOperator, add,
                        tup::Tuple{Real,Real},
                        shift, dτ)
    # trying out Ali's suggestion with occupation ratio of neighbours
    # off-diagonal: spawning psips
    num = tup[1] # number of psips on configuration
    occ_ratio= tup[2] # ratio of occupied vs total number of neighbours
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        wnapsips, wnaflag = w[naddress]
        if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(wnapsips),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] = (wnapsips+nspawns, wnaflag)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(ham,add)
    # modify shift locally according to occupation ratio of neighbouring configs
    mshift = occ_ratio > 0 ? shift*occ_ratio : shift
    pd = dτ * (dME - mshift) # modified
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now appropriate type
    wapsips, waflag = w[add]
    if sign(wapsips) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(wapsips),abs(ndiags))
    end
    w[add] = (wapsips + ndiags, waflag) # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return [spawns, deaths, clones, antiparticles, annihilations]
    # note that w is not returned
end # inner_step!

function fciqmc_col!(s::IsSemistochastic, w, ham::LinearOperator, add,
         val_flag_tuple::Tuple{N, F}, shift, dτ) where {N<:Number, F<:Integer}
    (val, flag) = val_flag_tuple
    deterministic = flag & one(F) # extract deterministic flag
    # diagonal death or clone
    new_val = w[add][1] + (1 + dτ*(shift - diagME(ham,add)))*val
    if deterministic
        w[add] = (new_val, flag) # new tuple
    else
        if new_val < s.threshold
            if new_val/s.threshold > cRand()
                new_val = convert(N,s.threshold)
                w[add] = (new_val, flag) # new tuple
            end
            # else # do nothing, stochastic space and rounded to zero
        else
            w[add] = (new_val, flag) # new tuple
        end
    end
    # off-diagonal: spawning psips
    if deterministic
        for (nadd, elem) in Hops(ham, add)
            wnapsips, wnaflag = w[nadd]
            if wnaflag & one(F) # new address `nadd` is also in deterministic space
                w[nadd] = (wnapsips - dτ * elem * val, wnaflag)  # new tuple
            else
                # TODO: det -> sto
                pspawn = abs(val * dτ * matelem) # non-negative Float64, pgen = 1
                nspawn = floor(pspawn) # deal with integer part separately
                cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
                # at this point, nspawn is non-negative
                # now converted to correct type and compute sign
                nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
                # - because Hamiltonian appears with - sign in iteration equation
                if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
                    annihilations += min(abs(wnapsips),abs(nspawns))
                end
                if !iszero(nspawns) # successful attempt to spawn
                    w[naddress] = (wnapsips+nspawns, wnaflag)
                    # perform spawn (if nonzero): add walkers with correct sign
                    spawns += abs(nspawns)
                end
            end
        end
    else
        # TODO: stochastic
        hops = Hops(ham, add)
        for n in 1:floor(abs(val)) # abs(val÷s.threshold) # for each psip attempt to spawn once
            naddress, pgen, matelem = generateRandHop(hops)
            pspawn = dτ * abs(matelem) /pgen # non-negative Float64
            nspawn = floor(pspawn) # deal with integer part separately
            cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
            # at this point, nspawn is non-negative
            # now converted to correct type and compute sign
            nspawns = convert(typeof(val), -nspawn * sign(val) * sign(matelem))
            # - because Hamiltonian appears with - sign in iteration equation
            wnapsips, wnaflag = w[naddress]
            if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
                annihilations += min(abs(wnapsips),abs(nspawns))
            end
            if !iszero(nspawns)
                w[naddress] = (wnapsips+nspawns, wnaflag)
                # perform spawn (if nonzero): add walkers with correct sign
                spawns += abs(nspawns)
            end
        end
        # deal with non-integer remainder
        rval =  abs(val%1) # abs(val%threshold)
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = rval * dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(val), -nspawn * sign(val) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        wnapsips, wnaflag = w[naddress]
        if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(wnapsips),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] = (wnapsips+nspawns, wnaflag)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
        # done with stochastic spawning
    end
    return [0, 0, 0, 0, 0]
end

function fciqmc_col!(s::IsStochasticWithThreshold, w, ham::LinearOperator,
        add, val::N, shift, dτ) where N <: Real

    # diagonal death or clone: deterministic fomula
    w[add] += (1 + dτ*(shift - diagME(ham,add)))*val
    # projection to threshold should be applied after all colums are evaluated

    # # apply threshold if necessary
    # if new_val < s.threshold
    #     # project stochastically to threshold
    #     w[add] = (new_val/s.threshold > cRand()) ? s.threshold : 0
    # else
    #     w[add] = new_val
    # end

    # off-diagonal: spawning psips stochastically
    # only integers are spawned!!
    hops = Hops(ham, add)
    # first deal with integer psips
    for n in 1:floor(abs(val)) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
        end
    end
    # deal with non-integer remainder: atempt to spawn
    rval =  abs(val%1) # non-integer part reduces probability for spawning
    naddress, pgen, matelem = generateRandHop(hops)
    pspawn = rval * dτ * abs(matelem) /pgen # non-negative Float64
    nspawn = floor(pspawn) # deal with integer part separately
    cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
    # at this point, nspawn is non-negative
    # now converted to correct type and compute sign
    nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
    # - because Hamiltonian appears with - sign in iteration equation
    if !iszero(nspawns)
        w[naddress] += nspawns
        # perform spawn (if nonzero): add walkers with correct sign
    end
    # done with stochastic spawning
    return [0, 0, 0, 0, 0]
end
