"""
    mpi_all_to_all(data, comm = mpi_comm(), root = mpi_root)

Declare `data` as mpi-distributed and set communication strategy to all-to-all.

Sets up the [`MPIData`](@ref) structure with [`MPIAllToAll`](@ref) strategy.
"""
function mpi_all_to_all(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIAllToAll(eltype(data), np, id, comm)
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
    sort_and_count!(counts, displs, vec, order, (lo, hi), i_start, j_start)

Sort new spawns by target rank. While this is done, also count the values and calculate the
offsets needed for `MPI.Alltoallv!`.

`counts`, `displs`, `vec`, and `order` are modified in-place. `order` should contain the
values you want to sort vec by. (i.e. `targetrank.(vec, s.np)`)

    sort_and_count!(s::MPIAllToAll)

As above, but operating on the internal buffers of `s`. Note that `s.targets` is expected to
contain the correct values to sort by.
"""
function sort_and_count!(
    counts, displs, vec, ord, (lo, hi), i_start=firstindex(ord), j_start=lastindex(ord)
)
    if i_start == j_start
        # Only one value left. We can stop recursing.
        counts[ord[i_start] + 1] = 1
        displs[ord[i_start] + 1] = i_start - 1
    elseif i_start < j_start
        if lo == hi
            # At this point, the recursion is finished and all values have the same
            # target rank.
            counts[lo + 1] = j_start - i_start + 1
            displs[lo + 1] = i_start - 1
        else
            # Idea:
            # Move from left (i) and right (j). Swap values so that the left part
            # will contain the values that are ≤ mid, while the right part will contain
            # those that are greater.
            # This is essentially equivalent to quicksort, but in this case, we know how
            # to pick a good pivot.
            mid = fld(lo + hi, 2)
            i = i_start
            j = j_start
            @inbounds while i < j
                if ord[i] ≤ mid
                    i += 1
                elseif ord[j] > mid
                    j -= 1
                else
                    @swap! ord i j
                    @swap! vec i j
                end
            end
            # This correction is needed if all values ≤ mid
            if ord[j] ≤ mid
                i += 1
            end
            # Recursively sort subarrays.
            sort_and_count!(counts, displs, vec, ord, (mid + 1, hi), j, j_start)
            sort_and_count!(counts, displs, vec, ord, (lo, mid), i_start, i - 1)
        end
    end
    return nothing
end

function sort_and_count!(s::MPIAllToAll)
    sb = s.sendbuffer
    sb.counts .= 0
    sb.displs .= 0
    sort_and_count!(sb.counts, sb.displs, sb.data, s.targets, (0, s.np - 1))
end

function prepare_send!(s::MPIAllToAll, source)
    buffer = s.sendbuffer.data
    targets = s.targets

    # Copy pairs to send buffer and sort by target rank
    len = length(source)
    if len > 0
        resize!(buffer, len)
        resize!(targets, len)
        @inbounds for (i, p) in enumerate(pairs(source))
            buffer[i] = p
            targets[i] = targetrank(p, s.np)
        end
        sort_and_count!(s)
    end
    copyto!(s.lenbuffer.data, s.sendbuffer.counts)
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

function Rimu.sort_into_targets!(target, source, s::MPIAllToAll)
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
