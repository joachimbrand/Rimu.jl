
@with_kw mutable struct FCIQMCParams
    step::Int = 0 # number of current/starting timestep
    laststep::Int = 50 # number of final timestep
    targetwalkers::Int = 1_000 # when to switch to variable shift mode
    variableShiftMode::Bool = false # whether to adjust shift
    shift::Float64 = 0.0 # starting/current value of shift
    dτ::Float64 = 0.01 # time step
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    fciqmc!(svec, h::LinearOperator, pa::FCIQMCParams [, svec2])

Perform the FCIQMC algorithm for determining the lowest eigenvalue of `h`.
`svec` can be a single starting vector of type `:<AbstractDVec` or a tuple
of such vectors. In the latter case, independent replicas are constructed.
Returns a `DataFrame` with statistics about the run, or a tuple of `DataFrame`s
for a replica run. This function mutates `svec` and the parameter struct `pa.`
Optionally a pre-allocated `svec2` can be passed as arguments.
"""
function fciqmc!(svec::D, h::LinearOperator, pa::FCIQMCParams,
                 svec2::D = similar(svec)) where D<:AbstractDVec
    # unpack the parameters:
    @unpack step, laststep, targetwalkers, variableShiftMode, shift, dτ, ζ = pa

    maxlength = capacity(svec)
    maxlength ≤ capacity(svec2) || error("`svec2` needs to have at least `capacity(svec)`")
    vOld = svec # the starting vector
    vNew = zero!(svec2) # clear second vector
    pnorm = tnorm = norm(vOld, 1) # norm of "previous" vector
    # prepare df for recording data
    df = DataFrame(steps=Int[], shift=Float64[], shiftMode=Bool[], len=Int[],
                        norm=Float64[], spawns=Int[], deaths=[], clones=Int[],
                        antiparticles=Int[], annihilations=Int[])
    # first row of df to show starting point
    push!(df, (step, shift, variableShiftMode, length(svec), pnorm, 0, 0, 0, 0, 0))

    while step < laststep
        step += 1
        # perform one complete stochastic vector matrix multiplication
        ss, ds, cs, aps, ans = fciqmc_step!(vNew, h, vOld, shift, dτ)
        tnorm = norm(vNew, 1) # total number of psips
        # update shift and mode if necessary
        if variableShiftMode
            # shift -= 0.5/dτ * (tnorm/targetwalkers-1)
            shift -= ζ/dτ * log(tnorm/pnorm)
            # adjusts the shift
        elseif tnorm > targetwalkers
            # shift = -105.0
            variableShiftMode = true
            # turn variableShiftMode on if norm > targetwalkers
        # elseif tnorm > 0.7*targetwalkers
        #     shift = -80.0
        end
        pnorm = tnorm # remember norm of this step for next step (previous norm)
        len = length(vNew)
        # record results
        push!(df, (step, shift, variableShiftMode, len, tnorm,
                        ss, ds, cs, aps, ans))
        # prepare for next step:
        dvec = vOld # keep reference to old vector
        vOld = vNew # new will be old
        vNew = empty!(dvec) # clean out the old vector and assign to vNew reference
        len > 0.8*maxlength && if len > maxlength
            @error "`maxlength` exceeded" len maxlength
            break
        else
            @warn "`maxlength` nearly reached" len maxlength
        end
    end
    # make sure that `svec` contains the current population:
    if !(vOld === svec)
        copyto!(svec, vOld)
    end
    # pack up and parameters for continuation runs
    # note that this modifes the struct pa
    @pack! pa = step, variableShiftMode, shift, dτ, ζ
    return df
    # note that `svec` and `pa` are modified but not returned explicitly
end # fciqmc

# replica version
function fciqmc!(svecs::T, h::LinearOperator, pa::FCIQMCParams,
                 vsNew::T = similar.(svecs)) where {N, K, V,
                                                T<:NTuple{N,AbstractDVec{K,V}}}
                 # N is number of replica, V is eltype(svecs[1])
    # unpack the parameters:
    @unpack step, laststep, targetwalkers, variableShiftMode, shift, dτ, ζ = pa

    maxlength = minimum(capacity.(svecs))
    reduce(&, capacity.(vsNew) .≥ maxlength) || error("replica containers `vsNew` have insufficient capacity")
    vsOld = svecs # keep reference of the starting vectors
    zero!.(vsNew) # reset the vectors without allocating new memory

    shifts = [shift for i = 1:N] # Vector because it needs to be mutable
    vShiftModes = [variableShiftMode for i = 1:N] # separate for each replica
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
    PType = promote_type(V,eltype(h)) # type of scalar product
    RType = promote_type(PType,Float64) # for division
    mixed_df= DataFrame(steps =Int[], xdoty =V[], xHy =PType[], aveH =RType[])
    dp = vsOld[1]⋅vsOld[2] # <v_1 | v_2>
    expval =  vsOld[1]⋅h(vsOld[2]) # <v_1 | h | v_2>
    push!(mixed_df,(step, dp, expval, expval/dp))

    norms = zeros(N)
    mstats = [zeros(V,5) for i=1:N]
    while step < laststep
        step += 1
        @sync for (i, vOld) in enumerate(vsOld) # loop over replicas
            # perform one complete stochastic vector matrix multiplication
            @async begin
                vNew = vsNew[i]
                mstats[i] .= fciqmc_step!(vNew, h, vOld, shifts[i], dτ)
                norms[i] = norm(vNew,1) # total number of psips
                if vShiftModes[i]
                    shifts[i] -= ζ/dτ * log(norms[i]/pnorms[i])
                elseif norms[i] > targetwalkers
                    vShiftModes[i] = true
                end
            end
        end #loop over replicas
        lengths = length.(vsNew)
        # record results
        for i = 1:N
            push!(dfs[i], (step, shifts[i], vShiftModes[i], lengths[i],
                  norms[i], mstats[i]...))
        end
        v1Dv2 = vsNew[1]⋅vsNew[2] # <v_1 | v_2> overlap
        v2Dhv2 =  vsNew[1]⋅h(vsNew[2]) # <v_1 | h | v_2>
        push!(mixed_df,(step, v1Dv2, v2Dhv2, v2Dhv2/v1Dv2))

        # prepare for next step:
        pnorms .= norms # remember norm of this step for next step (previous norm)
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
    variableShiftMode = reduce(&,vShiftModes) # only true if all are in vShiftMode
    shift = reduce(+,shifts)/N # return average value of shift
    @pack! pa = step, variableShiftMode, shift, dτ, ζ

    return mixed_df, dfs # return dataframes with stats
    # note that `svecs` and `pa` are modified but not returned explicitly
end # fciqmc


function fciqmc_step!(w, h::LinearOperator, v, shift, dτ)
    w === v && error("`w` and `v` must not be the same object")
    spawns = deaths = clones = antiparticles = annihilations = zero(eltype(v))
    for (add, num) in v
        res = fciqmc_col!(w, h, add, num, shift, dτ)
        if !ismissing(res)
            spawns += res[1]
            deaths += res[2]
            clones += res[3]
            antiparticles += res[4]
            annihilations += res[5]
        end
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end # fciqmc_step!

# to do: implement parallel version
# function fciqmc_step!(w::D, h::LinearOperator, v::D, shift, dτ) where D<:DArray
#   check that v and w are compatible
#   for each worker
#      call fciqmc_step!()  on respective local parts
#      sort and consolidate configurations to where they belong
#      commuicate via RemoteChannels
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

struct MySSVec{T} <: AbstractVector{T}
    v::Vector{T}
    sssize::Int
end
Base.size(mv::MySSVec) = size(mv.v)
Base.getindex(mv::MySSVec, I...) = getindex(mv.v, I...)

StochasticStyle(::Type{<:MySSVec}) = IsSemistochastic()

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
    fciqmc_col!(w, h, add, num, shift, dτ)
    fciqmc_col!(T:Type, args...)
    -> spawns, deaths, clones, antiparticles, annihilations
Spawning and diagonal step of FCIQMC for single column of `h`. In essence it
computes

`w .+= (1 .+ dτ*(shift - h[:,add])).*num`.

Depending on `StochasticStyle(w)`, a stochastic or deterministic algorithm will
be chosen.

- `T == IsDeterministic()` deteministic algorithm
- `T == IsStochastic()` stochastic version where the changes added to `w` are
purely integer, according to the FCIQMC algorithm.
- `T == IsSemistochastic()` semistochastic version: TODO
"""
fciqmc_col!(w, args...) = fciqmc_col!(StochasticStyle(w), w, args...)

fciqmc_col!(T::Type, args...) = throw(TypeError(:fciqmc_col!,
    "first argument: trait not recognised",StochasticStyle,T))

function fciqmc_col!(::IsDeterministic, w, h::AbstractMatrix, add, num, shift, dτ)
    w .+= (1 .+ dτ.*(shift .- view(h,:,add))).*num
    # todo: return something sensible
    return missing
end

function fciqmc_col!(::IsDeterministic, w, h::LinearOperator, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in Hops(h, add)
        w[nadd] += -dτ * elem * num
    end
    # diagonal death or clone
    w[add] += (1 + dτ*(shift - diagME(h,add)))*num
    return missing
end

# fciqmc_col!(::IsStochastic,  args...) = inner_step!(args...)
# function inner_step!(w, h::LinearOperator, add, num::Number,
#                         shift, dτ)
function fciqmc_col!(::IsStochastic, w, h::LinearOperator, add, num::Number,
                        shift, dτ)
    # version for single population of integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(h,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) ≠ sign(nspawns) # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(h,add)
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

"""
    nearUniform(ham)
Create bitstring address with near uniform distribution of particles
across modes for the Hamiltonian `ham`.
"""
function nearUniform(h::BosonicHamiltonian)
    fillingfactor, extras = divrem(h.n, h.m)
    startonr = fill(fillingfactor,h.m)
    startonr[1:extras] += ones(Int, extras)
    return bitaddr(startonr, h.AT)
end
