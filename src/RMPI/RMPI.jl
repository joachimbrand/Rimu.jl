"""
Module for providing MPI helper functions for `Rimu`. This module is unexported. To use it,
run

```
using Rimu.RMPI
```
"""
module RMPI

# The following is actually set by MPI.jl itself. See:
# https://juliaparallel.org/MPI.jl/stable/knownissues/#Multi-threading-and-signal-handling
# Despite that, in some cases it needs to be set manually _before_ loading MPI.
ENV["UCX_ERROR_SIGNALS"] = "SIGILL,SIGBUS,SIGFPE"
import MPI

using Random: Random
using Rimu: Rimu

import ..DictVectors: mpi_rank, mpi_comm, mpi_size

export mpi_rank, is_mpi_root, @mpi_root, mpi_barrier
export mpi_comm, mpi_root, mpi_size, mpi_seed!, mpi_allprintln

function __init__()
    # Initialise the MPI library once at runtime.
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
end

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

# Make MPI reduction of a `MultiScalar` work on non-Intel processors.
# The `MultiScalar` is converted into a vector before sending through MPI.Allreduce.
# Testing shows that this is about the same speed or even a bit faster on Intel processors
# than reducing the MultiScalar directly via a custom reduction operator.
# Defining the method in RMPI is strictly type piracy as MultiScalar belongs to Rimu and
# not to RMPI. Might clean this up later.
function MPI.Allreduce(ms::Rimu.MultiScalar{T}, op, comm::MPI.Comm) where {T<:Tuple}
    result_vector = MPI.Allreduce([ms...], op, comm)
    return Rimu.MultiScalar(T(result_vector))
end

end
