struct CommunicatorError <: Exception
    msg::String
end
CommunicatorError(args...) = CommunicatorError(string(args...))

function Base.showerror(io::IO, ex::CommunicatorError)
    print(io, "CommunicatorError: ", ex.msg)
end

"""
    abstract type Communicator

When implementing a communicator, use [`local_segments`](@ref) and
[`remote_segments`](@ref).

# Interface

* [`synchronize_remote!`](@ref)
* [`mpi_rank`](@ref)
* [`mpi_size`](@ref)
* [`mpi_comm`](@ref)

# Optional interface

* [`is_distributed`](@ref): defaults to returning `true`.
* [`reduce_remote`](@ref): defaults to using `MPI.Allreduce`.
* [`total_num_segments`](@ref): defaults to `n * mpi_size`.
* [`target_segment`](@ref): defaults to selecting using fastrange to pick the segment.

"""
abstract type Communicator end

"""
    is_distributed(::Communicator)

Return `true` if communicator operates over MPI.
"""
is_distributed(::Communicator) = true

"""
    reduce_remote(c::Communicator, op, x)

Perform a reduction over MPI, by using `MPI.Allreduce`.
"""
reduce_remote(c::Communicator, op, x) = MPI.Allreduce(x, op, mpi_comm(c))

"""
    total_num_segments(c::Communicator, n) -> Int

Return the total number of segments, including the remote ones, where `n` is number of
local segments.
"""
total_num_segments(c::Communicator, n) = n * mpi_size(c)

"""
    target_segment(c::Communicator, k, num_segments) -> target, is_local

Return the target segment for a key if it's local key and whether it's local or not.
"""
function target_segment(c::Communicator, k, num_segments)
    total_segments = num_segments * mpi_size(c)
    result = fastrange_hash(k, total_segments) - mpi_rank(c) * num_segments
    return result, 1 ≤ result ≤ num_segments
end

"""
    mpi_rank(::Communicator) -> Int

Return the MPI rank of the communicator.
"""
mpi_rank

"""
    mpi_size(::Communicator) -> Int

Return the total number of MPI ranks.
"""
mpi_size

"""
    mpi_comm(::Communicator) -> MPI.Comm

Return the `MPI.Comm` that the communicator operates on.
"""
mpi_comm

"""
    copy_to_local!([::Communicator,] w::WorkingMemory, t::TVec) -> TVec

Copy pairs in `t` from all ranks and return them as (possibly) new [`TVec`](@ref), possibly
using the [`WorkingMemory`](@ref) as temporary storage.
"""
copy_to_local!

"""
    synchronize_remote!([::Communicator,] ::WorkingMemory)

Copy pairs from remote ranks to the local part of the [`WorkingMemory`](@ref).
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

reduce_remote(::NotDistributed, _, x) = x

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

function synchronize_remote!(::LocalPart, w)
    throw(CommunicatorError("attemted to synchronize localpart"))
end

reduce_remote(::LocalPart, _, x) = x

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

See [`insert_collections!`](@ref), [`mpi_send`](@ref), [`mpi_recv!`](@ref).
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
    insert_collections!(buf::SegmentedBuffer, iters, ex=ThreadedEx())

Insert collections in `iters` into buffers.
"""
function insert_collections!(buf::SegmentedBuffer, iters, ex=ThreadedEx())
    resize!(buf.offsets, length(iters))
    resize!(buf.buffer, sum(length, iters))

    # Compute offsets
    curr = 0
    for (i, col) in enumerate(iters)
        curr += length(col)
        buf.offsets[i] = curr
    end

    # Copy over the data
    Folds.foreach(buf, iters, ex) do dst, src
        for (i, x) in enumerate(src)
            dst[i] = x
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
    mpi_recv!(buf::SegmentedBuffer, dest, comm)

Receive the buffers from `source`.
"""
function mpi_recv!(buf::SegmentedBuffer, source, comm)
    offset_status = MPI.Probe(source, 0, comm)
    resize!(buf.offsets, MPI.Get_count(offset_status, Int))
    MPI.Recv!(buf.offsets, comm; source, tag=0)

    # Done for checking only
    buffer_status = MPI.Probe(source, 1, comm)
    buffer_length = MPI.Get_count(buffer_status, eltype(eltype(buf)))

    resize!(buf.buffer, buffer_length)
    MPI.Recv!(buf.buffer, comm; source, tag=1)

    # TODO: reproduce, fix and remove
    if buffer_length != last(buf.offsets)
        error(
            "Something went wrong.\n       ",
            "buffer_length: ", buffer_length, "\n       ",
            "last(buf.offsets): ", last(buf.offsets), "\n       ",
            buf.buffer,
        )
    end

    return buf
end

"""
    PointToPoint <: Communicator

[`Communicator`](@ref) that uses circular communication using `MPI.Isend` and `MPI.Recv!`.
"""
struct PointToPoint{K,V} <: Communicator
    send_buffer::SegmentedBuffer{Pair{K,V}}
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
        SegmentedBuffer{Pair{K,V}}(),
        SegmentedBuffer{Pair{K,V}}(),
        mpi_comm,
        MPI.Comm_rank(mpi_comm),
        MPI.Comm_size(mpi_comm),
    )
end

mpi_rank(ptp::PointToPoint) = ptp.mpi_rank
mpi_size(ptp::PointToPoint) = ptp.mpi_size
mpi_comm(ptp::PointToPoint) = ptp.mpi_comm

function synchronize_remote!(ptp::PointToPoint, w)
    for offset in 1:ptp.mpi_size - 1
        dst_rank = mod(ptp.mpi_rank + offset, ptp.mpi_size)
        src_rank = mod(ptp.mpi_rank - offset, ptp.mpi_size)

        insert_collections!(ptp.send_buffer, remote_segments(w, dst_rank), w.executor)
        mpi_send(ptp.send_buffer, dst_rank, ptp.mpi_comm)
        mpi_recv!(ptp.recv_buffer, src_rank, ptp.mpi_comm)

        Folds.foreach(add!, local_segments(w), ptp.recv_buffer, w.executor)
    end
end

function copy_to_local!(ptp::PointToPoint, w, t)
    insert_collections!(ptp.send_buffer, t.segments, w.executor)
    Folds.foreach(copy!, local_segments(w, mpi_rank(ptp)), t.segments)

    for offset in 1:ptp.mpi_size - 1
        dst_rank = mod(ptp.mpi_rank + offset, ptp.mpi_size)
        src_rank = mod(ptp.mpi_rank - offset, ptp.mpi_size)

        mpi_send(ptp.send_buffer, dst_rank, ptp.mpi_comm)
        mpi_recv!(ptp.recv_buffer, src_rank, ptp.mpi_comm)

        Folds.foreach(copy!, remote_segments(w, src_rank), ptp.recv_buffer, w.executor)
    end
    return main_column(w)
end
