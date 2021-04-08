function mpi_all_to_all(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIAllToAll(data, np, id, comm)
    return MPIData(data, comm, root, s)
end

struct MPIAllToAll{P} <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm

    lenbuffer::MPI.UBuffer{Vector{Cint}} # for communicating numbers of elements
    sendbuffer::MPI.VBuffer{Vector{P}}   # for sending chunks
    recvbuffer::MPI.VBuffer{Vector{P}}   # for receiving chunks
end

function MPIAllToAll(data::AbstractDVec{K,V}, np, id, comm) where {K,V}
    P = Pair{K,V}
    datatype = MPI.Datatype(P)
    lenbuf = MPI.UBuffer(zeros(Cint, np), 1, np, MPI.Datatype(Cint))
    sendbuf = MPI.VBuffer(P[], zeros(Cint, np), zeros(Cint, np), datatype)
    recvbuf = MPI.VBuffer(P[], zeros(Cint, np), zeros(Cint, np), datatype)

    return MPIAllToAll{P}(np, id, comm, lenbuf, sendbuf, recvbuf)
end

function prepare_send!(s::MPIAllToAll, source)
    tr = Base.Fix2(targetrank, s.np) # partially applied targetrank for convenience
    buffer = s.sendbuffer.data
    counts = s.sendbuffer.counts
    displs = s.sendbuffer.displs

    # Copy pairs to send buffer and sort by target rank
    len = length(source)
    resize!(buffer, len)
    for (i, p) in enumerate(pairs(source))
        buffer[i] = p
    end
    sort!(buffer, by=tr)

    # Prepare send buffer counts and displs
    counts .= zero(Cint)
    displs .= zero(Cint)
    i = 1
    for r in 0:s.np - 1
        c = 0
        displs[r + 1] = i - 1
        while i ≤ len && tr(buffer[i]) == r
            c += 1
            i += 1
        end
        counts[r + 1] = c
    end
    copyto!(s.lenbuffer.data, counts)
    return s
end

function prepare_recv!(s::MPIAllToAll)
    counts = s.lenbuffer.data
    resize!(s.recvbuffer.data, sum(counts))
    copyto!(s.recvbuffer.counts, counts)

    displ = 0
    for (r, c) in enumerate(counts)
        s.recvbuffer.displs[r] = displ
        displ += c
    end
    return s
end

function Rimu.sort_into_targets!(target, source, ::Type{P}, s::MPIAllToAll{P}) where {P}
    prepare_send!(s, source)
    MPI.Alltoall!(MPI.IN_PLACE, s.lenbuffer, s.comm)
    prepare_recv!(s)
    MPI.Alltoallv!(s.sendbuffer, s.recvbuffer, s.comm)

    for (k, v) in s.recvbuffer.data
        target[k] += v
    end
    return target
end
