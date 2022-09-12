"""
    mpi_no_exchange(data, comm = mpi_comm(), root = mpi_root)
Declare `data` as mpi-distributed and set communication strategy to
`MPINoWalkerExchange`. Sets up the [`MPIData`](@ref) structure with
[`MPINoWalkerExchange`](@ref) strategy.
"""
function mpi_no_exchange(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPINoWalkerExchange(np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
    MPINoWalkerExchange(nprocs, my_rank, comm)
Strategy for not exchanging walkers between ranks. Consequently there
will be no cross-rank annihilations.
"""
struct MPINoWalkerExchange <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm
end

function mpi_combine_walkers!(target, source, ::MPINoWalkerExchange)
    # specific for `MPINoWalkerExchange`: copy without communicating with
    # other ranks.
    return copyto!(target, source)
end
