"""
    mpi_default(data, comm = mpi_comm(), root = mpi_root)
Declare `data` as mpi-distributed and set communication strategy to default.
Sets up the [`MPIData`](@ref) structure with
[`MPIDefault`](@ref) strategy.
"""
function mpi_default(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIDefault(np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
    MPIDefault(nprocs, my_rank, comm)
Strategy for point-to-point MPI communication.
"""
struct MPIDefault <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm
end

function Rimu.sort_into_targets!(target, bufs::Vector{Vector{P}}, lens, ::Type{P}, s::MPIDefault) where P
    # use standard MPI message passing communication
    # use ring structure for sending around data with blocking communications:
    # first receive from lower ranks, then send, then recieve from higher ranks
    # receiving from lower ranks
    for r = 0:(s.id-1)
        status = MPI.Probe(r, 0, s.comm)
        count = MPI.Get_count(status, P) # how many pairs are ready
        rbuf = Vector{P}(undef,count) # allocate buffer of correct size
        MPI.Recv!(rbuf, r, 0, s.comm)
        for (key, val) in rbuf # sort into target dict right away
            target[key] += val
        end
    end
    # send all buffer entries moving to higher ranks
    for tr in (s.id + 1) : (s.np - 1)
        # s.id == 0 && println("sending to rank $tr")
        sstat = MPI.Send(view(bufs[tr],1:lens[tr]), tr, 0, s.comm)
        # println("sent from $(s.id) to $tr with status $sstat")
    end
    for tr in 0 : (s.id - 1)
        # s.id == 0 && println("sending to rank $tr")
        sstat = MPI.Send(view(bufs[tr+1],1:lens[tr+1]), tr, 0, s.comm)
        # println("sent from $(s.id) to $tr with status $sstat")
    end
    # receiving from higher ranks
    for r = (s.id+1):(s.np-1)
        status = MPI.Probe(r, 0, s.comm)
        count = MPI.Get_count(status, P) # how many pairs are ready
        rbuf = Vector{P}(undef,count) # allocate buffer of correct size
        MPI.Recv!(rbuf, r, 0, s.comm)
        for (key, val) in rbuf # sort into target dict right away
            target[key] += val
        end
    end

    # s.id == 0 && println("receiving done")
    MPI.Barrier(s.comm)
    return target
end # sort_into_targets! MPIDefault
