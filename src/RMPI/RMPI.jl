"""
Module for providing MPI functionality for `Rimu`. This module is unexported. To use it, run

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

using LinearAlgebra: LinearAlgebra, I, dot, ⋅
using Random: Random
using StaticArrays: StaticArrays, SVector
using VectorInterface: VectorInterface, add, zerovector

using Rimu: Rimu, AbstractDVec, AbstractFockAddress, DictVectors, IsDiagonal,
    LOStructure, StochasticStyle, deposit!, diagonal_element, freeze,
    localpart, offdiagonals, storage, val, walkernumber

import Rimu: sort_into_targets!
import ..DictVectors: mpi_rank, mpi_comm, mpi_size

export MPIData
export mpi_rank, is_mpi_root, @mpi_root, mpi_barrier
export mpi_comm, mpi_root, mpi_size, mpi_seed!, mpi_allprintln

function __init__()
    # Initialise the MPI library once at runtime.
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
end

const mpi_registry = Dict{Int,Any}()
# The registry keeps references to distributed structures that must not be
# garbage collected at random
abstract type DistributeStrategy end

include("mpidata.jl")
include("multiscalar.jl")
include("helpers.jl")
include("noexchange.jl")
include("pointtopoint.jl")
include("onesided.jl")
include("alltoall.jl")

end # module RMPI
