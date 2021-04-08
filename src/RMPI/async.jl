function mpi_async(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIAsynchronous(pairtype(localpart(data)), np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
    MPIAsynchronous{N,A}

Asynchronous point to point communication strategy.
"""
struct MPIAsynchronous{P,N,D} <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm
    datatype::D
    buffers::NTuple{N,Vector{P}}
end

function MPIAsynchronous(::Type{Pair{K,V}}, np, id, comm) where {K,V}
    P = Pair{K,V}
    datatype = MPI.Datatype(P)
    return MPIAsynchronous{P,np,typeof(datatype)}(np, id, comm, datatype, ntuple(_ -> P[], np))
end
function other_ranks(s::MPIAsynchronous{<:Any,N}) where {N}
    ntuple(Val(N - 1)) do i
        mod(s.id + i, s.np)
    end
end

function sendbuff(s::MPIAsynchronous, id)
    return s.buffers[id + 1]
end
function recvbuff(s::MPIAsynchronous)
    return s.buffers[s.id + 1]
end

function send!(s::MPIAsynchronous{<:Any,N}) where {N}
    for id in other_ranks(s)
        id == s.id && error()
        buffer = sendbuff(s, id)
        MPI.Isend(MPI.Buffer(buffer, length(buffer), s.datatype), id, 0, s.comm)
    end
end

function receive!(target, s::MPIAsynchronous{<:Any,N}) where {N}
    for id in reverse(other_ranks(s))
        id == s.id && error()
        status = MPI.Probe(id, 0, s.comm)
        count = MPI.Get_count(status, s.datatype)
        buffer = recvbuff(s)
        resize!(buffer, count)
        MPI.Recv!(MPI.Buffer(buffer, length(buffer), s.datatype), id, 0, s.comm)
        for (key, val) in buffer
            target[key] += val
        end
    end
end

function Rimu.sort_into_targets!(
    target, source, ::Type{P}, s::MPIAsynchronous{P,N,D}
) where {P,N,D}
    foreach(empty!, s.buffers)

    # Sort source into send buffers, put appropriate values into target.
    for (key, val) in pairs(source)
        tr = targetrank(key, s.np)
        if tr == s.id
            target[key] += val
        else
            push!(sendbuff(s, tr), key => val)
        end
    end
    send!(s)
    receive!(target, s)

    return target
end
