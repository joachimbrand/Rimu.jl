"""
Module for providing MPI functionality for `Rimu`. This module is unexported. To use it, run

```
using Rimu.RMPI
```
"""
module RMPI

import MPI
using Rimu
using LinearAlgebra
using Random
using StaticArrays
using VectorInterface

import ..Interfaces: sort_into_targets!

export MPIData
export mpi_rank, is_mpi_root, @mpi_root, mpi_barrier
export mpi_comm, mpi_root, mpi_size, mpi_seed!, mpi_allprintln

function __init__()
    # Initialise the MPI library once at runtime.
    MPI.Initialized() || MPI.Init()
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
