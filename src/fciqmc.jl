
"""
    fciqmc!(v, pa::FciqmcRunStrategy, [df,]
             ham, s_strat::ShiftStrategy,
             [r_strat::ReportingStrategy, τ_strat::TimeStepStrategy, w])
    -> df

Perform the FCIQMC algorithm for determining the lowest eigenvalue of `ham`.
`v` can be a single starting vector of type `:<AbstractDVec` or a tuple
of such vectors. In the latter case, independent replicas are constructed.
Returns a `DataFrame` `df` with statistics about the run, or a tuple of `DataFrame`s
for a replica run.
Strategies can be given for updating the shift (see [`ShiftStrategy`](@ref))
and (optionally), for reporting (see [`ReportingStrategy`](@ref)),
and for updating the time step `dτ` (see [`TimeStepStrategy`](@ref)).
A pre-allocated data structure `w` for working memory can be passed as argument.
This function mutates `v`, the parameter struct `pa` as well as
`df`, and `w`.
"""
function fciqmc!(svec::DD, pa::FciqmcRunStrategy,
                 ham,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 w::D = similar(localpart(svec))) where {DD, D<:AbstractDVec}
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
    rdf =  fciqmc!(svec, pa, df, ham, s_strat, r_strat, τ_strat, w)
    # # (DD <: MPIData) && println("$(svec.s.id): arrived at barrier; after")
    # (DD <: MPIData) && MPI.Barrier(svec.s.comm)
    return rdf
end

# for continuation runs we can also pass a DataFrame
function fciqmc!(v, pa::RunTillLastStep, df::DF,
                 ham,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy,
                 τ_strat::TimeStepStrategy,
                 w::D) where {D<:AbstractDVec, DF<:Union{DataFrame, Nothing}}
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
        v, w, step_stats = fciqmc_step!(ham, v, shift, dτ, w)
        tnorm = norm(v, 1) # MPI sycncronising: total number of psips
        # update shift and mode if necessary
        shift, shiftMode, pnorm = update_shift(s_strat,
                                    shift, shiftMode,
                                    tnorm, pnorm, dτ, step, df)
        # the updated "previous" norm pnorm is returned from `update_shift()`
        # in order to allow delaying the update, e.g. with `DelayedLogUpdate`
        # pnorm = tnorm # remember norm of this step for next step (previous norm)
        dτ = update_dτ(τ_strat, dτ) # will need to pass more information later
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
function fciqmc!(svecs::T, ham::LinearOperator, pa::RunTillLastStep,
                 s_strat::ShiftStrategy,
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 vsNew::T = similar.(svecs)) where {N, K, V,
                                                T<:NTuple{N,AbstractDVec{K,V}}}
                 # N is number of replica, V is eltype(svecs[1])
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa

    maxlength = minimum(capacity.(svecs))
    reduce(&, capacity.(vsNew) .≥ maxlength) || error("replica containers `vsNew` have insufficient capacity")
    vsOld = svecs # keep reference of the starting vectors
    zero!.(vsNew) # reset the vectors without allocating new memory

    shifts = [shift for i = 1:N] # Vector because it needs to be mutable
    vShiftModes = [shiftMode for i = 1:N] # separate for each replica
    pnorms = zeros(N) # initialise as vector
    pnorms .= norm.(vsOld,1) # 1-norm i.e. number of psips as Tuple (of previous step)

    # initalise df for storing results of each replica separately
    dfs = Tuple(DataFrame(steps=Int[], shift=Float64[], shiftMode=Bool[],
                         len=Int[], norm=Float64[], spawns=V[], deaths=V[],
                         clones=V[], antiparticles=V[],
                         annihilations=V[]) for i in 1:N)
    # dfs is thus an NTuple of DataFrames
    for i in 1:N
        push!(dfs[i], (step, shifts[i], vShiftModes[i], length(vsOld[i]),
                      pnorms[i], 0, 0, 0, 0, 0))
    end

    # prepare `DataFrame` for variational ground state estimator
    # we are assuming that N ≥ 2, otherwise this will fail
    PType = promote_type(V,eltype(ham)) # type of scalar product
    RType = promote_type(PType,Float64) # for division
    mixed_df= DataFrame(steps =Int[], xdoty =V[], xHy =PType[], aveH =RType[])
    dp = vsOld[1]⋅vsOld[2] # <v_1 | v_2>
    expval =  vsOld[1]⋅ham(vsOld[2]) # <v_1 | ham | v_2>
    push!(mixed_df,(step, dp, expval, expval/dp))

    norms = zeros(N)
    mstats = [zeros(V,5) for i=1:N]
    while step < laststep
        step += 1
        @sync for (i, vOld) in enumerate(vsOld) # loop over replicas
            # perform one complete stochastic vector matrix multiplication
            @async begin
                vNew = vsNew[i]
                a, b, stats = fciqmc_step!(ham, vOld, shifts[i], dτ, vNew)
                mstats[i] .= stats
                norms[i] = norm(vNew,1) # total number of psips
                shifts[i], vShiftModes[i], pnorms[i] = update_shift(
                    s_strat, shifts[i], vShiftModes[i],
                    norms[i], pnorms[i], dτ, step, dfs[i]
                )
            end
        end #loop over replicas
        lengths = length.(vsNew)
        # update time step
        dτ = update_dτ(τ_strat, dτ) # will need to pass more information
        # later when we add different stratgies
        # record results
        for i = 1:N
            push!(dfs[i], (step, shifts[i], vShiftModes[i], lengths[i],
                  norms[i], mstats[i]...))
        end
        v1Dv2 = vsNew[1]⋅vsNew[2] # <v_1 | v_2> overlap
        v2Dhv2 =  vsNew[1]⋅ham(vsNew[2]) # <v_1 | ham | v_2>
        push!(mixed_df,(step, v1Dv2, v2Dhv2, v2Dhv2/v1Dv2))

        # prepare for next step:
        # pnorms .= norms # remember norm of this step for next step (previous norm)
        dummy = vsOld # keep reference to old vector
        vsOld = vsNew # new will be old
        vsNew = dummy # new new is former old
        zero!.(vsNew) # reset the vectors without allocating new memory
        llength = maximum(lengths)
        llength > 0.8*maxlength && if llength > maxlength
            @error "`maxlength` exceeded" llength maxlength
            break
        else
            @warn "`maxlength` nearly reached" llength maxlength
        end

    end # while step
    # make sure that `svecs` contains the current population:
    if !(vsOld === svecs)
        for i = 1:N
            copyto!(svecs[i], vsOld[i])
        end
    end
    # pack up and parameters for continuation runs
    # note that this modifes the struct pa
    shiftMode = reduce(&,vShiftModes) # only true if all are in vShiftMode
    shift = reduce(+,shifts)/N # return average value of shift
    @pack! pa = step, shiftMode, shift, dτ

    return mixed_df, dfs # return dataframes with stats
    # note that `svecs` and `pa` are modified but not returned explicitly
end # fciqmc

"""
    fciqmc_step!(Ĥ, v, shift, dτ, w) -> ṽ, w̃, stats
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
function fciqmc_step!(Ĥ, v::D, shift, dτ, w::D) where D
    # serial version
    @assert w ≢ v "`w` and `v` must not be the same object"
    stats = zeros(valtype(v), 5) # pre-allocate array for stats
    zero!(w) # clear working memory
    for (add, num) in pairs(v)
        res = fciqmc_col!(w, Ĥ, add, num, shift, dτ)
        ismissing(res) || (stats .+= res) # just add all stats together
    end
    return w, v, stats
    # stats == [spawns, deaths, clones, antiparticles, annihilations]
end # fciqmc_step!

function Rimu.fciqmc_step!(Ĥ, dv::MPIData{D,S}, shift, dτ, w::D) where {D,S}
    # MPI version
    v = localpart(dv)
    @assert w ≢ v "`w` and `v` must not be the same object"
    empty!(w)
    stats = zeros(valtype(v), 5) # pre-allocate array for stats
    for (add, num) in pairs(v)
        res = Rimu.fciqmc_col!(w, Ĥ, add, num, shift, dτ)
        ismissing(res) || (stats .+= res) # just add all stats together
    end
    sort_into_targets!(dv, w)
    MPI.Allreduce!(stats, +, dv.comm) # add stats of all ranks
    return dv, w, stats
    # returns the structure with the correctly distributed end
    # result `dv` and cumulative `stats` as an array on all ranks
    # stats == (spawns, deaths, clones, antiparticles, annihilations)
end # fciqmc_step!

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

# let's decide whether a simulation is deterministic, stochastic, or
# semistochastic upon a trait on the vector type

"""
    StochasticStyle(V)
    StochasticStyle(typeof(V))
`StochasticStyle` specifies the native style of the generalised vector `V` that
determines how simulations are to proceed. This can be fully stochastic (with
`IsStochastic`), fully deterministic (with `IsDeterministic`), or semistochastic
(with `IsSemistochastic`).
"""
abstract type StochasticStyle end

struct IsStochastic <: StochasticStyle end

struct IsDeterministic <: StochasticStyle end

struct IsSemistochastic <: StochasticStyle end

# some sensible defaults
StochasticStyle(A::Union{AbstractArray,AbstractDVec}) = StochasticStyle(typeof(A))
StochasticStyle(::Type{<:Array}) = IsDeterministic()
StochasticStyle(::Type{Vector{Int}}) = IsStochastic()
# the following works for dispatch, i.e. the function is evaluated at compile time
function StochasticStyle(T::Type{<:AbstractDVec})
    ifelse(eltype(T) <: Integer, IsStochastic(), IsDeterministic())
end

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
    fciqmc_col!(T:Type, args...)
    -> spawns, deaths, clones, antiparticles, annihilations
Spawning and diagonal step of FCIQMC for single column of `ham`. In essence it
computes

`w .+= (1 .+ dτ.*(shift .- ham[:,add])).*num`.

Depending on `StochasticStyle(w)`, a stochastic or deterministic algorithm will
be chosen.

- `T == IsDeterministic()` deteministic algorithm
- `T == IsStochastic()` stochastic version where the changes added to `w` are purely integer, according to the FCIQMC algorithm.
- `T == IsSemistochastic()` semistochastic version: TODO
"""
fciqmc_col!(w::Union{AbstractArray,AbstractDVec}, args...) = fciqmc_col!(StochasticStyle(w), w, args...)

# generic method for unknown trait: throw error
fciqmc_col!(::Type{T}, args...) where T = throw(TypeError(:fciqmc_col!,
    "first argument: trait not recognised",StochasticStyle,T))

function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    w .+= (1 .+ dτ.*(shift .- view(ham,:,add))).*num
    # todo: return something sensible
    return missing
end

function fciqmc_col!(::IsDeterministic, w, ham::LinearOperator, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in Hops(ham, add)
        w[nadd] += -dτ * elem * num
    end
    # diagonal death or clone
    w[add] += (1 + dτ*(shift - diagME(ham,add)))*num
    return missing
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
    return (spawns, deaths, clones, antiparticles, annihilations)
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
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end # inner_step!
