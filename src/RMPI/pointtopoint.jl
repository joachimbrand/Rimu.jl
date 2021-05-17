"""
    mpi_point_to_point(data, comm = mpi_comm(), root = mpi_root)

Declare `data` as mpi-distributed and set communication strategy to point-to-point.

Sets up the [`MPIData`](@ref) structure with [`MPIPointToPoint`](@ref) strategy.
"""
function mpi_point_to_point(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIPointToPoint(eltype(storage(data)), np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
    MPIPointToPoint{N,A}

Point-to-point communication strategy. Uses circular communication using `MPI.Send` and
`MPI.Recv!`.

# Constructor

* `MPIPointToPoint(::Type{P}, np, id, comm)`: Construct an instance with pair type `P` on
  `np` processes with current rank `id`.

"""
struct MPIPointToPoint{P,N} <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm
    datatype::MPI.Datatype
    buffers::NTuple{N,Vector{P}}
end
function MPIPointToPoint(::Type{Pair{K,V}}, np, id, comm) where {K,V}
    P = Pair{K,V}
    datatype = MPI.Datatype(P)
    return MPIPointToPoint{P,np}(np, id, comm, datatype, ntuple(_ -> P[], np))
end
"""
    recvbuff(s::MPIPointToPoint)

Get the receive buffer.
"""
function recvbuff(s::MPIPointToPoint)
    return s.buffers[s.id + 1]
end
"""
    sendbuff(s::MPIPointToPoint, id)

Get the send buffer associated with `id`.
"""
function sendbuff(s::MPIPointToPoint, id)
    return s.buffers[id + 1]
end
"""
    receive!(target, s::MPIPointToPoint, id)

Recieve from rank with `id` and move recieved values to `target`.
"""
function receive!(target, s::MPIPointToPoint{P}, id) where P
    status = MPI.Probe(id, 0, s.comm)
    count = MPI.Get_count(status, s.datatype)
    resize!(recvbuff(s), count)
    rb = recvbuff(s)
    MPI.Recv!(MPI.Buffer(rb, length(rb), s.datatype), id, 0, s.comm)
    dict = storage(target)
    for (key, val) in recvbuff(s)
        prev_val = get(dict, key, zero(valtype(dict)))
        dict[key] = prev_val + val
    end
    return target
end
"""
    send!(s::MPIPointToPoint{P})

Send the contents of the send buffers to all other ranks.
"""
function send!(s::MPIPointToPoint{<:Any,N}) where {N}
    for id in 0:N-1
        id == s.id && continue
        sb = sendbuff(s, id)
        MPI.Send(MPI.Buffer(sb, length(sb), s.datatype), id, 0, s.comm)
    end
end

# TODO: the type is unused, but without it, everything breaks.
function Rimu.sort_into_targets!(target, source, s::MPIPointToPoint{<:Any,N}) where {N}
    foreach(empty!, s.buffers)

    # sort source into send buffers, put appropriate values into target.
    for (key, val) in pairs(source)
        tr = targetrank(key, s.np)
        if tr == s.id
            target[key] += val
        else
            push!(sendbuff(s, tr), key => val)
        end
    end
    # Receive from lower ranks.
    for id in 0:(s.id - 1)
        receive!(target, s, id)
    end
    # Send to all ranks.
    send!(s)
    # Receive from higher ranks.
    for id in (s.id + 1):(N - 1)
        receive!(target, s, id)
    end
    MPI.Barrier(s.comm)
    return target
end
