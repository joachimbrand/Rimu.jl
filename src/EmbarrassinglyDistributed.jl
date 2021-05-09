"""
    Rimu.EmbarrassinglyDistributed
Module that provides an embarrassingly parallel option for breaking up long
time-series with `lomc!()` into chunks performed in parallel using the
`Distributed` package.

### Exports:
* [`d_lomc!()`](@ref) - run [`lomc!()`](@ref) in embarrassingly parallel mode
* [`combine_dfs()`](@ref) - combine the resulting `DataFrame`s to a single one
* [`setup_workers()`](@ref) - set up workers for distributed computing
* [`seedCRNGs_workers!()`](@ref) - seed random number generators for distributed computing
"""
module EmbarrassinglyDistributed

using Random, Parameters, DataFrames
using Rimu, Rimu.ConsistentRNG
using Distributed

export d_lomc!, setup_workers, seedCRNGs_workers!, combine_dfs

"""
    seedCRNGs_workers!([seed])
Seed the random number generators `CRNG` from [`ConsistentRNG`](@ref) on all
available processes in a `Distributed` environment deterministically from `seed`
but such that their pseudo-random number sequences are statistically
independent.

If no `seed` is given, obtain one from system entropy
(with [`Random.RandomDevice()`](@ref)).
"""
function seedCRNGs_workers!(seed=rand(Random.RandomDevice(),UInt))
    @everywhere seedCRNG!($seed + hash(myid()))
end

"""
    d_lomc!(ham, v; eqsteps, kwargs...)
    -> (; dfs = DataFrame[], eqsteps)
Perform linear operator Monte Carlo with [`lomc!()`](@ref) in embarrassingly parallel
mode using `Distributed` computing. Returns all dataframes.

### Keyword arguments in addition to those of [`lomc!()`](@ref):
* `eqsteps` - Number of time steps used for equilibration. Each worker will run an independent simulation with `eqstep + (laststep - step) ÷ nworkers()` time steps.

### Example:
```julia
using Rimu, Rimu.EmbarrassinglyDistributed

setup_workers(4) # set up to run on 4 workers
seedCRNGs_workers!(127) # seed random number generators for deterministic evolution

add = BoseFS((1,1,0,1))
ham = BoseHubbardReal1D(add, u=4.0)
v = DVec(add => 2, capacity = 200)
# run `lomc!()` for 20_100 time steps by
# performing 4 parallel runs of `lomc!()` with 5_100 time steps each and
# stiching the results together into a single dataframe:
df, eqsteps = d_lomc!(ham, v; eqsteps = 100, laststep = 20_100) |> combine_dfs
# or
energies = d_lomc!(ham, v; eqsteps = 100, laststep = 20_100) |> combine_dfs |> autoblock
```

### See also:
* [`setup_workers()`](@ref)
* [`seedCRNGs_workers!()`](@ref)
* [`combine_dfs()`](@ref)
"""
function d_lomc!(ham, v;
    eqsteps,
    laststep = nothing,
    params::Rimu.FciqmcRunStrategy = Rimu.RunTillLastStep(),
    kwargs...
    )

    nw = nworkers()
    nw < 2 && @warn "Not enough workers available for parallel execution" nw

    if !isnothing(laststep)
        params.laststep = laststep
    end
    # unpack the parameters:
    @unpack step, laststep = params
    @assert laststep - step - eqsteps > nw "not enough time steps to run for"
    psteps = laststep - step - eqsteps # steps to be run after equilibration
    stepseach = psteps ÷ nw # to be run on each worker after equilibration

    # @everywhere do_it() = Rimu.lomc!($ham, sizehint!($v,($vc*3)>>1); $kwargs..., laststep = $step+$stepseach, params = $params, threading = false).df
    # start shorter jobs in parallel
    # futures = [@spawnat(p, Main.do_it()) for p in workers()]
    futures = [@spawnat(p, Rimu.lomc!(ham, v; kwargs..., laststep = step+eqsteps+stepseach, params = params, threading = false).df) for p in workers()]
    # futures = [@spawnat(p, Rimu.lomc!(ham, v; kwargs..., laststep = step+stepseach, params = params, threading = false).df) for p in workers()]
    dfs = fetch.(futures) # will now be an array of dataframes
    return (; dfs, eqsteps) # returns NamedTuple
end

"""
    combine_dfs(dfs::AbstractVector, eqsteps)
    combine_dfs((dfs, eqsteps))
Return a dataframe with compounded data from
the parallel runs assuming that after `eqsteps` time steps the runs are
equilibrated and uncorrelated.
"""
combine_dfs(tuple) = combine_dfs(tuple...) # for easy chaining

function combine_dfs(dfs::AbstractVector, eqsteps)
    # add worker id to dfs
    for i in eachindex(dfs)
        df = dfs[i]
        df.workerid = [workers()[i] for n in 1:size(df)[1]]
        df.stepsorg = copy(df.steps) # remember original time step
    end
    # check that the dataframes are all the same size:
    @assert mapreduce(isequal(size(dfs[1])), &, size.(dfs))

    # last time step recorded in the first dataframe
    s = dfs[1].steps[end]

    # create views of all the remaining dataframes starting after the eqsteps
    lelem = size(dfs[1])[1]
    df_adds = [view(dfs[i], (eqsteps+2):lelem, :) for i in 2:length(dfs)]
    # now number the time steps in the views consecutively
    for df in df_adds
        df[:,:steps] .= [s+i for i in 1:length(df.steps)]
        s = df.steps[end]
    end
    dfm = vcat(dfs[1],df_adds...)
    return (; df=dfm, eqsteps)
end # d_lomc!()

"""
    setup_workers([nw])
Set up and prepare `nw` workers for distributed computing. If `nw` is not given,
but multiple threads are available then as many workers will be prepared. The
`Rimu` package code will be made available on all workers.
"""
function setup_workers(nw = nothing)
    # organise the right number of workers
    if isnothing(nw)
        # if `nw` is not specified, use at least as many workers are threads are
        # available
        while Threads.nthreads() > nworkers()
            addprocs(1)
        end
        nw = nworkers()
    else
        while nw > nworkers()
            addprocs(1)
        end
    end
    @assert nw ≥ 1 "not enough workers to run in parallel"
    while nworkers() > nw
        # reduce the number of workers if it is larger than `nw`
        rmprocs(workers()[end])
    end
    # just to check that we have the right number of workers now:
    @assert nw == nworkers()

    @eval @everywhere using Rimu # load code on all walkers

    return nw
end

# function __init__()
#     if myid() == 1
#         nw = setup_workers()
#         @info "`REmbarrassinglyDistributed`: $nw workers set up and ready to go"
#     end
# end

end # module
