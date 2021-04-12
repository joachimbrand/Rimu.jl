"""
    mpi_one_sided(data, comm = mpi_comm(), root = mpi_root)

Declare `data` as mpi-distributed and set communication strategy to one-sided with remote
memory access (RMA).

Sets up the [`MPIData`](@ref) structure with [`MPIOneSided`](@ref) strategy.
"""
function mpi_one_sided(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    P = pairtype(data)

    # compute the required capacity for the communication buffer as a
    # fraction of the capacity of `data`
    cap = capacity(data) ÷ np + 1
    # id == root && println("on rank $id, capacity = ",cap)
    s = MPIOneSided(np, id, comm, P, Int32(cap))
    return MPIData(data, comm, root, s)
end

"""
    MPIOneSided(nprocs, myrank, comm, ::Type{T}, capacity)

Communication buffer for use with MPI one-sided communication (remote memory access). Up to
`capacity` elements of type `T` can be exchanged between MPI ranks via [`put`](@ref). It is
important that `isbitstype(T) == true`. Objects of type `MPIOneSided` have to be
freed manually with a (blocking) call to [`free()`](@ref).
"""
mutable struct MPIOneSided{T}  <: DistributeStrategy
    mpiid::Int # unique ID for MPI-distributed objects
    np::Int32 # number of MPI processes
    id::Int32 # number of this rank
    comm::MPI.Comm # MPI communicator
    b_win::MPI.Win # window for data buffer
    l_win::MPI.Win # window for length of buffered data
    capacity::Int32 # capacity of buffer
    buf::Vector{T} # local array for MPI window b_win
    n_elem::Vector{Int32} # local array for MPI window l_win

    function MPIOneSided(nprocs, myrank, comm, ::Type{T}, capacity) where T
        @assert isbitstype(T) "Buffer type for MPIOneSided needs to be isbitstype; found $T"
        buf = Vector{T}(undef, capacity)
        n_elem = zeros(Int32,1)
        b_win = MPI.Win_create(buf, comm) # separate windows for buffer and length
        l_win = MPI.Win_create(n_elem, comm)
        mpiid = next_mpiID()
        obj = new{T}(mpiid, nprocs, myrank, comm, b_win, l_win, capacity, buf, n_elem)
        # MPI.refcount_inc() # ref counting was removed in MPI.jl v0.16.0
        mpi_registry[mpiid] = Ref(obj) # register the object to avoid
        # arbitrary garbage collection
        # ccall(:jl_, Cvoid, (Any,), "installing finalizer on MPIOneSided")
        finalizer(myclose, obj) # install finalizer
        return obj
    end
end
Base.eltype(::MPIOneSided{T}) where T = T

function Base.show(io::IO, s::MPIOneSided{T}) where T
    println(io, "MPIOneSided{$T}")
    println(io, "  mpiid: ",s.mpiid)
    println(io, "  np: ",s.np)
    println(io, "  id: ",s.id)
    println(io, "  comm: ",s.comm)
    println(io, "  MPI.Win and buf::Vector{$T} with capacity ",s.capacity)
    println(io, "  MPI.Win for number of elements in buffer")
end

function myclose(obj::MPIOneSided)
    MPI.free(obj.b_win)
    MPI.free(obj.l_win)
    return nothing
end

"""
    free(obj::MPIOneSided)
De-reference the object, call finalizer and the garbage collector immediately.
This is a syncronizing MPI call. Make sure that the object is not used later.
MPI syncronizing.
"""
function free(obj::MPIOneSided)
    global mpi_registry
    MPI.Barrier(obj.comm)
    delete!(mpi_registry, obj.mpiid)
    finalize(obj)
    GC.gc()
end
function free(d::MPIData{D,S}) where {D,S<:MPIOneSided}
    free(d.s)
    finalize(d)
    GC.gc()
end
function free(d::MPIData) # default
    finalize(d)
    GC.gc()
end

function fence(s::MPIOneSided, assert = 0)
    MPI.Win_fence(assert, s.b_win)
    MPI.Win_fence(assert, s.l_win) # joint fences
end

"""
    put(buf::Vector{T}, [len,] targetrank, s::MPIOneSided{T})
    put(obj::T, targetrank, s::MPIOneSided{T})

Deposit a single `obj` or vector `buf` into the MPI window `s` on rank `targetrank`. If
`len` is given, only the first `len` elements are transmitted.
"""
@inline function put(buf::Vector{T}, len, targetrank, s::MPIOneSided{T}) where T
    @boundscheck len ≤ length(buf) && len ≤ s.capacity ||
        error("Not enough space left in buffer")
    MPI.Put(buf, len, targetrank, 0, s.b_win)
    MPI.Put(Ref(len), 1, targetrank, 0, s.l_win)
end
@inline function put(buf::Vector{T}, targetrank, s::MPIOneSided{T}) where T
    len = length(buf)
    put(buf, len, targetrank, s)
end
@inline function put(obj::T, targetrank, s::MPIOneSided{T}) where T
    @boundscheck s.capacity ≥ 1 || error("Not enough space left in buffer")
    MPI.Put(Ref(obj), 1, targetrank, 0, s.b_win)
    MPI.Put(Ref(1), 1, targetrank, 0, s.l_win)
end

function sbuffer!(s::MPIOneSided) # unsafe version - avoids allocations
    # println("from sbuffer $(s.buf), $(s.n_elem), $(s.n_elem[1])")
    # return s.buf[1:s.n_elem[1]]
    return view(s.buf, 1 : s.n_elem[1])
end
function sbuffer(s::MPIOneSided) # safe version for reading shared buffer - returns local array
    MPI.Win_lock(MPI.LOCK_SHARED, s.id, 0, s.b_win)
    MPI.Win_lock(MPI.LOCK_SHARED, s.id, 0, s.l_win)
    res = s.buf[1 : s.n_elem[1]]
    MPI.Win_unlock(s.id,s.l_win)
    MPI.Win_unlock(s.id,s.b_win)
    return res
end

function Rimu.sort_into_targets!(target, bufs::Vector{Vector{P}}, lens, ::Type{P}, s::MPIOneSided) where P
    # send data with RMA communications to higher rank by `offset`
    for offset in 1 : s.np-1
        # println("$(s.id) before first fence")
        fence(s)
        # println("jumped first fence on $(s.id)")

        tr = (s.id + offset) % s.np # compute target rank for sending
        bufindex = tr > s.id ? tr : tr + 1 # and buffer index
        # println("Sending $(lens[bufindex]) pairs from $(s.id) to $tr")

        put(bufs[bufindex], lens[bufindex], tr, s)
        # println("sending done on $(s.id)")

        fence(s)
        # println("jumped the fence on $(s.id)")
        for (key, val) in sbuffer(s)
            target[key] += val
        end
    end
    fence(s)
    return target
end
