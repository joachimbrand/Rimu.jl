import Rimu: mpi_rank, mpi_size, mpi_comm

struct CommunicatorError <: Exception
    msg::String
end
CommunicatorError(args...) = CommunicatorError(string(args...))

function Base.showerror(io::IO, ex::CommunicatorError)
    print(io, "CommunicatorError: ", ex.msg)
end

"""
    abstract type Communicator

Communicators are used to handle MPI communication when using [`PDVec`](@ref)s. Currently,
three implementations are provided, [`NotDistributed`](@ref), [`AllToAll`](@ref) and
[`PointToPoint`](@ref). The communicator is picked automatically according to the number of
MPI ranks available.

When implementing a communicator, use [`local_segments`](@ref) and
[`remote_segments`](@ref).

# Interface

* [`synchronize_remote!`](@ref)
* [`mpi_rank`](@ref)
* [`mpi_size`](@ref)
* [`mpi_comm`](@ref)

# Optional interface

* [`is_distributed`](@ref): defaults to returning `true`.
* [`merge_remote_reductions`](@ref): defaults to using `MPI.Allreduce`.
* [`total_num_segments`](@ref): defaults to `n * mpi_size`.
* [`target_segment`](@ref): defaults to selecting using
  [fastrange](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/)  to pick the segment.

See also: [`PDVec`](@ref), [`PDWorkingMemory`](@ref).
"""
abstract type Communicator end

"""
    is_distributed(::Communicator)

Return `true` if [`Communicator`](@ref) operates over MPI.
"""
is_distributed(::Communicator) = true

"""
    merge_remote_reductions(c::Communicator, op, x)

Merge the results of reductions over MPI. By default, it uses `MPI.Allreduce`.

See also: [`Communicator`](@ref).
"""
merge_remote_reductions(c::Communicator, op, x) = MPI.Allreduce!(Ref(x), op, mpi_comm(c))[]

"""
    total_num_segments(c::Communicator, n) -> Int

Return the total number of segments, including the remote ones, where `n` is number of
local segments.

See also: [`PDVec`](@ref), [`Communicator`](@ref).
"""
total_num_segments(c::Communicator, n) = n * mpi_size(c)

"""
    target_segment(c::Communicator, k, num_segments) -> target, is_local

This function is used to determine where in the [`PDVec`](@ref) a key should be stored.
If the key is local (stored on the same MPI rank), return its segment index and `true`. If
the key is non-local, return any value and `false`.

See also: [`PDVec`](@ref), [`Communicator`](@ref).
"""
function target_segment(c::Communicator, k, num_segments)
    total_segments = num_segments * mpi_size(c)
    result = fastrange_hash(k, total_segments) - mpi_rank(c) * num_segments
    return result, 1 ≤ result ≤ num_segments
end

"""
    mpi_rank(::Communicator) -> Int

Return the MPI rank of the [`Communicator`](@ref).
"""
mpi_rank

"""
    mpi_size(::Communicator) -> Int

Return the total number of MPI ranks in the [`Communicator`](@ref).
"""
mpi_size

"""
    mpi_comm(::Communicator) -> MPI.Comm

Return the `MPI.Comm` that the [`Communicator`](@ref) operates on.
"""
mpi_comm

"""
    copy_to_local!([::Communicator,] w::PDWorkingMemory, t::PDVec) -> PDVec

Copy pairs in `t` from all ranks and return them as a (possibly) new [`PDVec`](@ref),
possibly using the [`PDWorkingMemory`](@ref) as temporary storage.

See also: [`PDVec`](@ref), [`PDWorkingMemory`](@ref), [`Communicator`](@ref).
"""
copy_to_local!

"""
    NotDistributed <: Communicator

This [`Communicator`](@ref) is used when MPI is not available.
"""
struct NotDistributed <: Communicator end

is_distributed(::NotDistributed) = false

mpi_rank(::NotDistributed) = 0
mpi_size(::NotDistributed) = 1

synchronize_remote!(::NotDistributed, _) = (), ()

function copy_to_local!(::NotDistributed, w, t)
    return t
end

merge_remote_reductions(::NotDistributed, _, x) = x

total_num_segments(::NotDistributed, n) = n

target_segment(::NotDistributed, k, num_segments) = fastrange_hash(k, num_segments), true

"""
    LocalPart <: Communicator

When [`localpart`](@ref) is used, the vector's [`Communicator`](@ref) is replaced with this.
This allows iteration and local reductions.
"""
struct LocalPart{C} <: Communicator
    communicator::C
end

is_distributed(::LocalPart) = false
mpi_rank(lp::LocalPart) = mpi_rank(lp.communicator)
mpi_size(lp::LocalPart) = mpi_size(lp.communicator)

function synchronize_remote!(::LocalPart, _)
    throw(CommunicatorError("attemted to synchronize localpart"))
end

merge_remote_reductions(::LocalPart, _, x) = x

function target_segment(c::LocalPart, k, num_segments)
    total_segments = num_segments * mpi_size(c)
    result = fastrange_hash(k, total_segments) - mpi_rank(c) * num_segments
    if 1 ≤ result ≤ num_segments
        return result, true
    else
        throw(CommunicatorError("attempted to access non-local key $k"))
    end
end

const SubVector{T} = SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}

"""
    SegmentedBuffer{T}() <: AbstractVector{AbstractVector{T}}

Behaves like a vector of vectors, but is stored in a single buffer. It can be sent/received
over MPI keeping its structure intact. Used in the [`PointToPoint`](@ref) communication
strategy.

# Supported operations

* [`replace_collections!`](@ref): insert data into the buffers
* [`mpi_send`](@ref): send the contents of a buffer to a given rank
* [`mpi_recv_any!`](@ref): receive a message sent by [`mpi_send`](@ref) from any rank,
  storing the contents in this buffer

See also: [`NestedSegmentedBuffer`](@ref).
"""
struct SegmentedBuffer{T} <: AbstractVector{SubVector{T}}
    offsets::Vector{Int}
    buffer::Vector{T}
end
function SegmentedBuffer{T}() where {T}
    return SegmentedBuffer(Int[], T[])
end

Base.size(buf::SegmentedBuffer) = size(buf.offsets)

function Base.getindex(buf::SegmentedBuffer, i)
    start_index = get(buf.offsets, i-1, 0) + 1
    end_index = buf.offsets[i]
    return view(buf.buffer, start_index:end_index)
end

"""
    replace_collections!(buf::SegmentedBuffer, iters)

Insert collections in `iters` into a [`SegmentedBuffer`](@ref).

```julia
julia> using Rimu.DictVectors: SegmentedBuffer

julia> buf = SegmentedBuffer{Int}()
0-element SegmentedBuffer{Int64}

julia> Rimu.DictVectors.replace_collections!(buf, [[1,2,3], [4,5]])
2-element SegmentedBuffer{Int64}:
 [1, 2, 3]
 [4, 5]

julia> Rimu.DictVectors.replace_collections!(buf, [[1], [2,3], [4]])
3-element SegmentedBuffer{Int64}:
 [1]
 [2, 3]
 [4]
```
"""
function replace_collections!(buf::SegmentedBuffer, iters)
    resize!(buf.offsets, length(iters))
    resize!(buf.buffer, sum(length, iters))

    # Compute offsets
    curr = 0
    for (i, col) in enumerate(iters)
        curr += length(col)
        buf.offsets[i] = curr
    end

    # Copy over the data
    Folds.foreach(buf, iters) do dst, src
        for (i, v) in enumerate(src)
            dst[i] = v
        end
    end
    return buf
end

"""
    mpi_send(buf::SegmentedBuffer, dest, comm::MPI.Comm)

Send the buffer to rank with id `dest`.
"""
function mpi_send(buf::SegmentedBuffer, dest, comm)
    @assert MPI.Is_thread_main()
    MPI.Isend(buf.offsets, comm; dest, tag=0)
    MPI.Isend(buf.buffer, comm; dest, tag=1)
    return buf
end

"""
    mpi_recv_any!(buf::SegmentedBuffer, comm::MPI_Comm) -> Int

Find a source that is ready to send a buffer and receive from it. Return the rank ID of the
sender.
"""
function mpi_recv_any!(buf::SegmentedBuffer, comm)
    @assert MPI.Is_thread_main()
    status = offset_status = MPI.Probe(MPI.ANY_SOURCE, 0, comm)
    source = status.source
    resize!(buf.offsets, MPI.Get_count(offset_status, Int))
    MPI.Recv!(buf.offsets, comm; source, tag=0)

    resize!(buf.buffer, last(buf.offsets))
    MPI.Recv!(buf.buffer, comm; source, tag=1)
    return source
end

"""
    PointToPoint{K,V}(; mpi_comm, report) <: Communicator

MPI [`Communicator`](@ref) that uses circular communication using `MPI.Isend` and
`MPI.Recv!`.

# Keyword arguments

* `mpi_comm=MPI.COMM_WORLD`: the MPI communicator to use.
*  `report=false`: if set to true, report MPI communication times during a projector Monte
  Carlo run.
"""
struct PointToPoint{K,V} <: Communicator
    send_buffers::Vector{SegmentedBuffer{Pair{K,V}}}
    recv_buffer::SegmentedBuffer{Pair{K,V}}
    mpi_comm::MPI.Comm
    mpi_rank::Int
    mpi_size::Int
    report::Bool
end
function PointToPoint{K,V}(
    ;
    mpi_comm=MPI.COMM_WORLD,
    report=false,
) where {K,V}
    mpi_rank=MPI.Comm_rank(mpi_comm)
    mpi_size=MPI.Comm_size(mpi_comm)
    return PointToPoint(
        [SegmentedBuffer{Pair{K,V}}() for _ in 1:mpi_size-1],
        SegmentedBuffer{Pair{K,V}}(),
        mpi_comm,
        mpi_rank,
        mpi_size,
        report,
    )
end

function Base.show(io::IO, ata::PointToPoint{K,V}) where {K,V}
    print(io, "PointToPoint{$K,$V}(mpi_comm=$(ata.mpi_comm), report=$(ata.report))")
end

mpi_rank(ptp::PointToPoint) = ptp.mpi_rank
mpi_size(ptp::PointToPoint) = ptp.mpi_size
mpi_comm(ptp::PointToPoint) = ptp.mpi_comm

function synchronize_remote!(ptp::PointToPoint, w)
    comm_time = @elapsed begin
        # Asynchronously send all buffers.
        for offset in 1:ptp.mpi_size - 1
            dst_rank = mod(ptp.mpi_rank + offset, ptp.mpi_size)
            send_buffer = ptp.send_buffers[offset]
            replace_collections!(send_buffer, remote_segments(w, dst_rank))
            mpi_send(send_buffer, dst_rank, ptp.mpi_comm)
        end

        # Receive and insert from each rank. The order is first come first serve.
        for _ in 1:ptp.mpi_size - 1
            mpi_recv_any!(ptp.recv_buffer, ptp.mpi_comm)
            Folds.foreach(dict_add!, local_segments(w), ptp.recv_buffer)
        end
    end
    if ptp.report
        return (:total_comm_time,), (comm_time,)
    else
        return (), ()
    end
end

function copy_to_local!(ptp::PointToPoint, w, t)
    # Same data sent to all ranks, so we can reuse the buffer.
    send_buffer = first(ptp.send_buffers)
    replace_collections!(send_buffer, t.segments)

    for offset in 1:ptp.mpi_size - 1
        dst_rank = mod(ptp.mpi_rank + offset, ptp.mpi_size)
        mpi_send(send_buffer, dst_rank, ptp.mpi_comm)
    end

    # We need all the data, including local in w.
    Folds.foreach(copy!, local_segments(w), t.segments)

    # Receive and insert from each rank. The order is first come first serve.
    for _ in 1:ptp.mpi_size - 1
        src_rank = mpi_recv_any!(ptp.recv_buffer, ptp.mpi_comm)
        Folds.foreach(remote_segments(w, src_rank), ptp.recv_buffer) do dst, src
            empty!(dst)
            for (k, v) in src
                dst[k] = valtype(dst)(v)
            end
        end
    end

    return first_column(w)
end

"""
    NestedSegmentedBuffer{T}(nrows) <: AbstractMatrix{AbstractVector{T}}

Matrix of vectors stored in a single buffer with collective MPI communication support. The
number of rows in the matrix is fixed to `nrows`.

Used in the [`AllToAll`](@ref) communication strategy, where each column corresponds to an
MPI rank and each row corresponds to a segment in the [`PDVec`](@ref).

# Supported operations

* [`append_collections!`](@ref): add a column to the matrix.
* [`append_empty_column!`](@ref): add an empty column to the matrix.
* [`mpi_exchange_alltoall!`](@ref): each rank sends the `i`-th column of the matrix to the
  `(i-1)`-st rank.
* [`mpi_exchange_allgather!`](@ref): each rank sends the `1`-st column of the matrix to all
  ranks.

See also: [`SegmentedBuffer`](@ref).
"""
struct NestedSegmentedBuffer{T} <: AbstractMatrix{SubVector{T}}
    nrows::Int
    counts::Vector{Int}
    offsets::Vector{Int}
    buffer::Vector{T}
end
function NestedSegmentedBuffer{T}(nrows) where {T}
    return NestedSegmentedBuffer{T}(nrows, Int[], Int[], T[])
end

Base.size(buf::NestedSegmentedBuffer) = (buf.nrows, length(buf.offsets) ÷ buf.nrows)

function Base.getindex(buf::NestedSegmentedBuffer, i, j)
    nrows = buf.nrows
    ncols = length(buf.counts)
    index = (j - 1) * nrows + i

    offset = sum(view(buf.counts, 1:(j-1)))

    start_index = (i == 1 ? 0 : buf.offsets[index - 1]) + offset + 1
    end_index = buf.offsets[index] + offset
    return view(buf.buffer, start_index:end_index)
end

"""
    append_collections!(buf::NestedSegmentedBuffer, iters)

Add a column to `buf`. The length of `iters` should match `buf.nrows`.

See also: [`NestedSegmentedBuffer`](@ref), [`append_empty_column!`](@ref).
"""
function append_collections!(buf::NestedSegmentedBuffer, iters)
    if length(iters) ≠ buf.nrows
        throw(ArgumentError("Expected $(buf.nrows) iterators, got $(length(iters))"))
    end

    count = sum(length, iters)
    buf_start = length(buf.buffer)
    offset_start = length(buf.offsets)

    push!(buf.counts, count)
    resize!(buf.offsets, length(iters) + length(buf.offsets))
    resize!(buf.buffer, count + length(buf.buffer))

    curr = 0
    for (i, it) in enumerate(iters)
        curr += length(it)
        buf.offsets[offset_start + i] = curr
    end

    for (i, v) in enumerate(Iterators.flatten(iters))
        buf.buffer[buf_start + i] = v
    end
    return buf
end

"""
    append_empty_column!(buf::NestedSegmentedBuffer)

Like [`append_collections!`](@ref), but adds an empty column.

See also: [`NestedSegmentedBuffer`](@ref), [`append_collections!`](@ref).
"""
function append_empty_column!(buf::NestedSegmentedBuffer)
    push!(buf.counts, 0)
    for i in 1:buf.nrows
        push!(buf.offsets, 0)
    end
    return buf
end

function Base.empty!(buf::NestedSegmentedBuffer)
    empty!(buf.counts)
    empty!(buf.offsets)
    empty!(buf.buffer)
end

"""
    mpi_exchange_alltoall!(src::NestedSegmentedBuffer, dst::NestedSegmentedBuffer, comm)

The `n`-th column from `src` will be sent to rank `n-1`. The data sent from rank `r` will be
stored in the `(r+1)`-st column of `dst`.

See also: [`NestedSegmentedBuffer`](@ref), [`mpi_exchange_allgather!`](@ref).
"""
function mpi_exchange_alltoall!(
    src::NestedSegmentedBuffer, dst::NestedSegmentedBuffer, comm
)
    @assert MPI.Is_thread_main()
    nrows = src.nrows
    if dst.nrows ≠ nrows
        throw(ArgumentError("mismatch in number of columns ($nrows and $(dst.nrows))."))
    end

    resize!(dst.offsets, length(src.offsets))
    MPI.Alltoall!(MPI.UBuffer(src.offsets, nrows), MPI.UBuffer(dst.offsets, nrows), comm)

    resize!(dst.counts, length(src.counts))
    for i in eachindex(dst.counts)
        dst.counts[i] = dst.offsets[i * nrows]
    end

    resize!(dst.buffer, sum(dst.counts))
    send_vbuff = MPI.VBuffer(src.buffer, src.counts)
    recv_vbuff = MPI.VBuffer(dst.buffer, dst.counts)

    MPI.Alltoallv!(send_vbuff, recv_vbuff, comm)
    return dst
end

"""
    mpi_exchange_allgather!(src::NestedSegmentedBuffer, dst::NestedSegmentedBuffer, comm)

The first and only column in `src` will be sent to all ranks. The data from all ranks will
be gethered in `dst`. After this operation, `dst` will contain the same data on all ranks.

See also [`NestedSegmentedBuffer`](@ref), [`mpi_exchange_alltoall!`](@ref).
"""
function mpi_exchange_allgather!(
    src::NestedSegmentedBuffer, dst::NestedSegmentedBuffer, comm
)
    @assert MPI.Is_thread_main()
    # only sending from first column.
    @assert size(src, 2) == 1
    nrows = src.nrows

    resize!(dst.counts, MPI.Comm_size(comm))
    MPI.Allgather!(MPI.Buffer(src.counts), MPI.UBuffer(dst.counts, 1), comm)

    resize!(dst.offsets, MPI.Comm_size(comm) * nrows)
    MPI.Allgather!(MPI.Buffer(src.offsets), MPI.UBuffer(dst.offsets, nrows), comm)

    resize!(dst.buffer, sum(dst.counts))
    send_buff = MPI.Buffer(src.buffer)
    recv_vbuff = MPI.VBuffer(dst.buffer, dst.counts)

    MPI.Allgatherv!(send_buff, recv_vbuff, comm)

    return dst
end

"""
    AllToAll{K,V}(; mpi_comm, n_segments, report) <: Communicator

[`Communicator`](@ref) that uses collective communication using `MPI.Alltoall[v]!`.

# Keyword arguments

* `mpi_comm=MPI.COMM_WORLD`: the MPI communicator to use.
* `n_segments=Threads.nthreads()`: the number of segments per rank to use. Should match the
  [`PDVec`](@ref) the communicator is used with.
*  `report=false`: if set to true, report MPI communication times during a projector Monte
  Carlo run.

See also: [`Communicator`](@ref).
"""
struct AllToAll{K,V} <: Communicator
    send_buffer::NestedSegmentedBuffer{Pair{K,V}}
    recv_buffer::NestedSegmentedBuffer{Pair{K,V}}
    mpi_comm::MPI.Comm
    mpi_rank::Int
    mpi_size::Int
    report::Bool
end
function AllToAll{K,V}(
    ; mpi_comm=MPI.COMM_WORLD, n_segments=Threads.nthreads(), report=false,
) where {K,V}
    mpi_rank=MPI.Comm_rank(mpi_comm)
    mpi_size=MPI.Comm_size(mpi_comm)

    return AllToAll(
        NestedSegmentedBuffer{Pair{K,V}}(n_segments),
        NestedSegmentedBuffer{Pair{K,V}}(n_segments),
        mpi_comm,
        mpi_rank,
        mpi_size,
        report,
    )
end

function Base.show(io::IO, ata::AllToAll{K,V}) where {K,V}
    print(io, "AllToAll{$K,$V}(mpi_comm=$(ata.mpi_comm), report=$(ata.report))")
end

mpi_rank(ata::AllToAll) = ata.mpi_rank
mpi_size(ata::AllToAll) = ata.mpi_size
mpi_comm(ata::AllToAll) = ata.mpi_comm

function synchronize_remote!(ata::AllToAll, w)
    # Fill the buffer
    comm_time = @elapsed begin
        empty!(ata.send_buffer)

        for i in 0:(ata.mpi_size - 1)
            if i == ata.mpi_rank
                # these are not remote, but need to be added anyway
                append_empty_column!(ata.send_buffer)
            else
                append_collections!(ata.send_buffer, remote_segments(w, i))
            end
        end
        mpi_time = @elapsed mpi_exchange_alltoall!(
            ata.send_buffer, ata.recv_buffer, ata.mpi_comm
        )
        for i in 1:ata.mpi_size
            Folds.foreach(dict_add!, local_segments(w), view(ata.recv_buffer, :, i))
        end
    end
    if ata.report
        return (
            (:mpi_comm_time, :total_comm_time),
            (mpi_time, comm_time),
        )
    else
        return (), ()
    end
end

function copy_to_local!(ata::AllToAll, w, p)
    empty!(ata.send_buffer)
    append_collections!(ata.send_buffer, p.segments)
    mpi_exchange_allgather!(ata.send_buffer, ata.recv_buffer, ata.mpi_comm)

    for i in 1:ata.mpi_size
        Folds.foreach(
            remote_segments(w, i-1), view(ata.recv_buffer, :, i)
        ) do seg, buff
            empty!(seg)
            dict_add!(seg, buff)
        end
    end

    return first_column(w)
end
