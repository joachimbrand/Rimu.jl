# default root and comm for this package
"Default MPI root for `RMPI`."
const mpi_root = zero(Int32)

"Default MPI communicator for `RMPI`."
mpi_comm() = MPI.COMM_WORLD

"""
    mpi_size(comm = mpi_comm())
Size of MPI communicator.
"""
mpi_size(comm = mpi_comm()) = MPI.Comm_size(comm)

"""
    next_mpiID()
Produce a new ID number for MPI distributed objects. Uses an internal counter.
"""
next_mpiID

let mpiID::Int = 0
    global next_mpiID
    next_mpiID() = (mpiID +=1; mpiID)
end

"""
    mpi_rank(comm = mpi_comm())
Return the current MPI rank.
"""
mpi_rank(comm = mpi_comm()) = MPI.Comm_rank(comm)

"""
    is_mpi_root(root = mpi_root)
Returns `true` if called from the root rank
"""
is_mpi_root(root = mpi_root) = mpi_rank() == root

"""
    @mpi_root expr
Evaluate expression only on the root rank.
Extra care needs to be taken as `expr` *must not* contain any code that involves
syncronising MPI operations, i.e. actions that would require syncronous action
of all MPI ranks.

Example:
```julia
wn = walkernumber(dv)   # an MPI syncronising function call that gathers
                        # information from all MPI ranks
@mpi_root @info "The current walker number is" wn # print info message on root only
```
"""
macro mpi_root(args...)
    :(is_mpi_root() && $(esc(args...)))
end

"""
    mpi_barrier(comm = mpi_comm())
The MPI barrier with optional argument. MPI syncronizing.
"""
mpi_barrier(comm = mpi_comm()) = MPI.Barrier(comm)

"""
    targetrank(key, np, hash = hash(key))
Compute the rank where the `key` belongs.
"""
targetrank(key, np, hash = hash(key)) = hash%np

"""
    sort_into_targets!(target::MPIData, source, stats)
Distribute the entries of `source` to the `target` data structure such that all
entries in the `target` dictionaries are on the process with the correct rank
as controlled by [`targetrank()`](@ref). Combine `stats` if appropriate.
MPI syncronizing.
"""
function Rimu.sort_into_targets!(dtarget::MPIData, source::AbstractDVec)
    ltarget = localpart(dtarget)
    empty!(ltarget) # start with empty slate
    strategy = dtarget.s
    P = pairtype(source) # compute pairtype
    # println("pairtype = ",P)
    sort_into_targets!(ltarget, source, P, strategy)
end

# three-argument version
function Rimu.sort_into_targets!(dtarget::MPIData, ws::NTuple{NT,W}, statss) where {NT,W}
    # multi-threaded MPI version
    # should only ever run on thread 1
    @assert Threads.threadid() == 1 "`sort_into_targets!()` is running on `threadid()` == $(Threads.threadid()) instead of 1!"
    lwm = ws[1]
    for i in 2:NT # combine new walkers generated from different threads
        add!(lwm, ws[i])
    end
    sort_into_targets!(dtarget,lwm) # combine walkers from different MPI ranks
    stats = sum(statss) # combine stats from all threads
    MPI.Allreduce!(stats, +, dtarget.comm) # add stats from all MPI ranks
    return dtarget, ws, stats
end
function Rimu.sort_into_targets!(dtarget::MPIData, w::AbstractDVec, stats)
    # single threaded MPI version
    sort_into_targets!(dtarget,w) # combine walkers from different MPI ranks
    vstats = convert(Vector,stats) # necessary for MPI.Allreduce
    MPI.Allreduce!(vstats, +, dtarget.comm) # add stats from all MPI ranks
    return dtarget, w, vstats
end

function Rimu.sort_into_targets!(target, source, ::Type{P}, s::DistributeStrategy) where P
    # now target is just a local data structure, e.g. DVec
    # allocate local buffers for sorting
    bufs = [Vector{P}(undef,length(source)) for i in 1:(s.np-1)] # type-stable
    lens = zeros(Int,(s.np-1))
    # sort source into send buffers
    @inbounds for (key,val) in pairs(source)
        tr = targetrank(key, s.np)
        if tr < s.id
            lens[tr+1] +=1
            bufs[tr+1][lens[tr+1]] = key => val
        elseif tr > s.id
            lens[tr] +=1
            bufs[tr][lens[tr]] = key => val
        else # tr == s.id
            target[key] += val # local: just copy to target
        end
    end
    # call strategy-specific method (with 5 arguments):
    return sort_into_targets!(target, bufs, lens, P, s)
end

"""
    sync_cRandn(md::MPIData)
Generate one random number with [`cRandn()`](@ref) in a synchronous way such
that all MPI ranks have the same random number.
The argument is ignored unless it is of type `MPIData`, in which case a random
number from the root rank is broadcasted to all MPI ranks. MPI syncronizing.
"""
function ConsistentRNG.sync_cRandn(md::MPIData)
    MPI.bcast(cRandn(), md.root, md.comm)
end

"""
    ConsistentRNGs.check_crng_independence(dv::MPIData)
Does a sanity check to detect dependence of random number generators across
all MPI ranks. Returns the size of the combined RNG state,
i.e. `mpi_size()*Threads.nthreads()*fieldcount(ConsistentRNG.CRNG)`.
MPI syncronizing.
"""
ConsistentRNG.check_crng_independence(dv::MPIData) = _check_crng_independence(dv.comm)

function _check_crng_independence(comm::MPI.Comm) # MPI syncronizing
    # get vector of threaded RNGs on this rank
    crngs = ConsistentRNG.CRNGs[]
    # extract the numbers that make up the state of the RNGs and package into
    # an MPI-suitable buffer
    statebuffer = [getfield(rng,i) for rng in crngs for i in 1:fieldcount(typeof(rng))]
    # gather from all ranks
    combined_statebuffer = MPI.Allgather(statebuffer, comm)  # MPI syncronizing
    # check independence
    @mpi_root @assert union(combined_statebuffer) == combined_statebuffer "Dependency in parallel rngs detected"

    @assert length(ConsistentRNG.CRNGs[]) == Threads.nthreads() "Number of CNRGs should be equal to nthreads()"
    return length(combined_statebuffer)
end

"""
    mpi_seed_CRNGs!(seed = rand(Random.RandomDevice(), UInt))
Re-seed the random number generators in an MPI-safe way. If seed is provided,
the random numbers from [`cRand()`](@ref) will follow a deterministic sequence.

Independence of the random number generators on different MPI ranks is achieved
buy adding `hash(mpi_rank())` to `seed`.
"""
function mpi_seed_CRNGs!(seed = rand(Random.RandomDevice(), UInt))
    rngs = seedCRNG!(seed + hash(mpi_rank()))
    _check_crng_independence(mpi_comm())
    return rngs
end
