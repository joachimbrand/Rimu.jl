"""
    mpi_all_to_all(data, comm = mpi_comm(), root = mpi_root)

Declare `data` as mpi-distributed and set communication strategy to all-to-all.

Sets up the [`MPIData`](@ref) structure with [`MPIAllToAll`](@ref) strategy.
"""
function mpi_all_to_all(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIAllToAll(pairtype(data), np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
     MPIAllToAll

All-to-all communication strategy. The communication works in two steps: first
`MPI.Alltoall!` is used to communicate the number of walkers each rank wants to send to
other ranks, then `MPI.Alltoallv!` is used to send the walkers around.

# Constructor

* `MPIAllToAll(Type{P}, np, id, comm)`: Construct an instance with pair type `P` on
  `np` processes with current rank `id`.

"""
struct MPIAllToAll{P} <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm

    targets::Vector{UInt}
    lenbuffer::MPI.UBuffer{Vector{Cint}} # for communicating numbers of elements
    sendbuffer::MPI.VBuffer{Vector{P}}   # for sending chunks
    recvbuffer::MPI.VBuffer{Vector{P}}   # for receiving chunks
end

function MPIAllToAll(::Type{Pair{K,V}}, np, id, comm) where {K,V}
    P = Pair{K,V}
    datatype = MPI.Datatype(P)
    lenbuf = MPI.UBuffer(zeros(Cint, np), 1, np, MPI.Datatype(Cint))
    sendbuf = MPI.VBuffer(P[], zeros(Cint, np), zeros(Cint, np), datatype)
    recvbuf = MPI.VBuffer(P[], zeros(Cint, np), zeros(Cint, np), datatype)

    return MPIAllToAll{P}(np, id, comm, UInt[], lenbuf, sendbuf, recvbuf)
end

"""
    @swap! arr i j

Swap the `i`-th and `j`-th indices in `arr`.
"""
macro swap!(arr, i, j)
    return quote
        tmp = $(esc(arr))[$(esc(j))]
        $(esc(arr))[$(esc(j))] = $(esc(arr))[$(esc(i))]
        $(esc(arr))[$(esc(i))] = tmp
        $(esc(arr))
    end
end

"""
    sort_by_rank!(arr, s)

In-place sort a-la insertion sort, but for a small number of unique elements. Much faster than
`sort!`.

TODO: this should also construct `counts` and `displs`. targetranks should be precomputed.
"""
function sort_by_rank!(arr, s)
    # Precompute targets for efficiency.
    targets = s.targets
    resize!(targets, length(arr))
    targets .= targetrank.(arr, s.np)

    i = 1
    len = length(arr)
    @inbounds for r in 0:(s.np - 1)
        while true
            # Find the first non-`r` index.
            while i ≤ len && targets[i] == r
                i += 1
            end
            j = i + 1
            # Find the first `r` to swap into `i`
            while j ≤ len && targets[j] ≠ r
                j += 1
            end
            j > len && break
            @swap! arr i j
            @swap! targets i j
            i += 1
            j += 1
        end
    end
    return arr
end

function prepare_send!(s::MPIAllToAll, source)
    buffer = s.sendbuffer.data
    counts = s.sendbuffer.counts
    displs = s.sendbuffer.displs

    # Copy pairs to send buffer and sort by target rank
    len = length(source)
    resize!(buffer, len)
    for (i, p) in enumerate(pairs(source))
        buffer[i] = p
    end
    sort_by_rank!(buffer, s)

    # Prepare send buffer counts and displs
    counts .= zero(Cint)
    displs .= zero(Cint)
    i = 1
    for r in 0:s.np - 1
        c = 0
        displs[r + 1] = i - 1
        while i ≤ len && s.targets[i] == r
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
    MPI.Barrier(s.comm)
    MPI.Alltoall!(MPI.IN_PLACE, s.lenbuffer, s.comm)
    prepare_recv!(s)
    MPI.Alltoallv!(s.sendbuffer, s.recvbuffer, s.comm)

    for (k, v) in s.recvbuffer.data
        target[k] += v
    end
    return target
end
