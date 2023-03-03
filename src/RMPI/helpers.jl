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

macro mpi_on(p, args...)
    :(mpi_rank() == $p && $(esc(args...)))
end

"""
    mpi_barrier(comm = mpi_comm())
The MPI barrier with optional argument. MPI syncronizing.
"""
mpi_barrier(comm = mpi_comm()) = MPI.Barrier(comm)

"""
    targetrank(key, np)
Compute the rank where the `key` belongs.
"""
targetrank(key::Union{Integer,AbstractFockAddress}, np) = hash(key, hash(1)) % np
targetrank(pair::Pair, np) = targetrank(pair[1], np)

"""
    mpi_combine_walkers!(target, source, [strategy])
Distribute the entries of `source` to the `target` data structure such that all
entries in the `target` are on the process with the correct mpi rank
as controlled by [`targetrank()`](@ref).
MPI syncronizing.

Note: the [`storage`](@ref) of the `source` is communicated rather than the `source` itself.
"""
function mpi_combine_walkers!(dtarget::MPIData, source::AbstractDVec)
    ltarget = localpart(dtarget)
    empty!(ltarget) # start with empty slate
    strategy = dtarget.s
    mpi_combine_walkers!(ltarget, storage(source), strategy)
end

function sort_into_targets!(dtarget::MPIData, w::AbstractDVec, stats)
    # single threaded MPI version
    mpi_combine_walkers!(dtarget, w) # combine walkers from different MPI ranks
    @static if Sys.ARCH ∈ (:aarch64, :ppc64le, :powerpc64le) ||
            startswith(lowercase(String(Sys.ARCH)), "arm")
        # Reductions of a custom type (`MultiScalar`) are not possible with MPI.jl on
        # non-Intel architectures at the moment
        # see https://github.com/JuliaParallel/MPI.jl/issues/404
        res_stats = (MPI.Allreduce(stat, +, dtarget.comm) for stat in stats)
    else
        # this should be more efficient if it is allowed
        res_stats = MPI.Allreduce(Rimu.MultiScalar(stats), +, dtarget.comm)
    end
    return dtarget, w, res_stats
end

"""
    mpi_seed!(seed = rand(Random.RandomDevice(), UInt))
Re-seed the random number generators in an MPI-safe way. If seed is provided,
the random numbers from `rand` will follow a deterministic sequence.

Independence of the random number generators on different MPI ranks is achieved
by adding `hash(mpi_rank())` to `seed`.
"""
function mpi_seed!(seed = rand(Random.RandomDevice(), UInt))
    rngs = Random.seed!(seed + hash(mpi_rank()))
    return rngs
end

"""
    mpi_allprintln(args...)
Print a message to `stdout` from each rank separately, in order. MPI synchronizing.
"""
function mpi_allprintln(args...)
    mpi_barrier()
    for i in 0:(mpi_size() - 1)
        if mpi_rank() == i
            println("[ rank ", lpad(i, length(string(mpi_size() - 1))), ": ", args...)
            flush(stdout)
        end
        mpi_barrier()
    end
end
