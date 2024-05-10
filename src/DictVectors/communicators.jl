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
two implementations are provided, [`NotDistributed`](@ref), and [`PointToPoint`](@ref). The
communicator is picked automatically according to the number of MPI ranks available.

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

"""
abstract type Communicator end

"""
    is_distributed(::Communicator)

Return `true` if communicator operates over MPI.
"""
is_distributed(::Communicator) = true

"""
    merge_remote_reductions(c::Communicator, op, x)

Merge the results of reductions over MPI. By default, it uses `MPI.Allreduce`.
"""
merge_remote_reductions(c::Communicator, op, x) = only(MPI.Allreduce!([x], op, mpi_comm(c)))

"""
    total_num_segments(c::Communicator, n) -> Int

Return the total number of segments, including the remote ones, where `n` is number of
local segments.
"""
total_num_segments(c::Communicator, n) = n * mpi_size(c)

"""
    target_segment(c::Communicator, k, num_segments) -> target, is_local

This function is used to determine where in the [`PDVec`](@ref) a key should be stored.

If the key is local (stored on the same MPI rank), return its segment index and `true`. If the key is non-local, return any value and `false`.
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

Copy pairs in `t` from all ranks and return them as (possibly) new [`PDVec`](@ref), possibly
using the [`PDWorkingMemory`](@ref) as temporary storage.
"""
copy_to_local!

"""
    synchronize_remote!([::Communicator,] ::PDWorkingMemory)

Copy pairs from remote ranks to the local part of the [`PDWorkingMemory`](@ref).
"""
synchronize_remote!


"""
    NotDistributed <: Communicator

This [`Communicator`](@ref) is used when MPI is not available.
"""
struct NotDistributed <: Communicator end

is_distributed(::NotDistributed) = false

mpi_rank(::NotDistributed) = 0
mpi_size(::NotDistributed) = 1

synchronize_remote!(::NotDistributed, w) = w

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

function synchronize_remote!(::LocalPart, w)
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

"""
    SegmentedBuffer

Multiple vectors stored in a simple buffer with MPI communication.

See [`replace_collections!`](@ref), [`mpi_send`](@ref), [`mpi_recv_any!`](@ref).
"""
struct SegmentedBuffer{T} <: AbstractVector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}
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

Insert collections in `iters` into buffers.
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
    mpi_send(buf::SegmentedBuffer, dest, comm)

Send the buffers to `dest`.
"""
function mpi_send(buf::SegmentedBuffer, dest, comm)
    MPI.Isend(buf.offsets, comm; dest, tag=0)
    MPI.Isend(buf.buffer, comm; dest, tag=1)
    return buf
end

"""
    mpi_recv_any!(buf::SegmentedBuffer, comm) -> Int

Find a source that is ready to send a buffer and receive from it. Return the rank ID of the
sender.
"""
function mpi_recv_any!(buf::SegmentedBuffer, comm)
    status = offset_status = MPI.Probe(MPI.ANY_SOURCE, 0, comm)
    source = status.source
    resize!(buf.offsets, MPI.Get_count(offset_status, Int))
    MPI.Recv!(buf.offsets, comm; source, tag=0)

    resize!(buf.buffer, last(buf.offsets))
    MPI.Recv!(buf.buffer, comm; source, tag=1)
    return source
end

"""
    PointToPoint <: Communicator

[`Communicator`](@ref) that uses circular communication using `MPI.Isend` and `MPI.Recv!`.
"""
struct PointToPoint{K,V} <: Communicator
    send_buffers::Vector{SegmentedBuffer{Pair{K,V}}}
    recv_buffer::SegmentedBuffer{Pair{K,V}}
    mpi_comm::MPI.Comm
    mpi_rank::Int
    mpi_size::Int
end
function PointToPoint{K,V}(
    ;
    mpi_comm=MPI.COMM_WORLD,
    mpi_rank=MPI.Comm_rank(mpi_comm),
    mpi_size=MPI.Comm_size(mpi_comm),
) where {K,V}
    return PointToPoint(
        [SegmentedBuffer{Pair{K,V}}() for _ in 1:mpi_size-1],
        SegmentedBuffer{Pair{K,V}}(),
        mpi_comm,
        mpi_rank,
        mpi_size,
    )
end

mpi_rank(ptp::PointToPoint) = ptp.mpi_rank
mpi_size(ptp::PointToPoint) = ptp.mpi_size
mpi_comm(ptp::PointToPoint) = ptp.mpi_comm

function synchronize_remote!(ptp::PointToPoint, w)
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

    # Pack the segments into a PDVec and return it.
    return main_column(w)
end
