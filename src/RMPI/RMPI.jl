"""
Module for providing MPI functionality for `Rimu`.
"""
module RMPI

import MPI
using Rimu, LinearAlgebra, Rimu.ConsistentRNG, Random

import Rimu: sort_into_targets!

export MPIData
export mpi_rank, is_mpi_root, @mpi_root, mpi_barrier
export mpi_comm, mpi_root, mpi_size, mpi_seed_CRNGs!

function __init__()
    if MPI.MPI_LIBRARY ≠ 0
        # Initialise the MPI library once at runtime.
        MPI.Initialized() || MPI.Init()
        # make sure that MPI ranks have independent random numbers
        mpi_seed_CRNGs!()
    else
        @warn "No known MPI library detected."
    end
end

const mpi_registry = Dict{Int,Any}()
# The registry keeps references to distributed structures that must not be
# garbage collected at random
abstract type DistributeStrategy end

include("mpidata.jl")
include("helpers.jl")
include("noexchange.jl")
include("pointtopoint.jl")
include("onesided.jl")
include("alltoall.jl")

end # module RMPI
