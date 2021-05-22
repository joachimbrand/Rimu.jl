"""
    mpi_one_sided(data, comm = mpi_comm(), root = mpi_root; capacity)

Declare `data` as mpi-distributed and set communication strategy to one-sided with remote
memory access (RMA). `capacity` sets the capacity of the RMA windows.

Sets up the [`MPIData`](@ref) structure with [`MPIOneSided`](@ref) strategy.
"""
function mpi_one_sided(data, comm = mpi_comm(), root = mpi_root; capacity)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    P = eltype(storage(data))

    # compute the required capacity for the communication buffer as a
    # fraction of the capacity of `data`
    # id == root && println("on rank $id, capacity = ",cap)
    s = MPIOneSided(np, id, comm, P, Int32(capacity))
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
    DT_b::MPI.Datatype # datatype for data
    b_win::MPI.Win # window for data buffer
    DT_l::MPI.Datatype # datatype for lengths of buffered data
    l_win::MPI.Win # window for length of buffered data
    capacity::Int32 # capacity of buffer
    b_vec::Vector{T} # local array for MPI window b_win
    l_vec::Vector{Int32} # local array for MPI window l_win

    function MPIOneSided(nprocs, myrank, comm, ::Type{T}, capacity) where T
        @assert isbitstype(T) "Buffer type for MPIOneSided needs to be isbitstype; found $T"
        b_vec = Vector{T}(undef, capacity)
        l_vec = zeros(Int32,1)
        DT_b = MPI.Datatype(T)
        b_win = MPI.Win_create(b_vec, comm) # separate windows for buffer and length
        DT_l = MPI.Datatype(Int32)
        l_win = MPI.Win_create(l_vec, comm)
        mpiid = next_mpiID()
        obj = new{T}(mpiid,nprocs,myrank,comm, DT_b,b_win, DT_l,l_win,capacity,b_vec,l_vec)
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
    println(io, "  MPI.Win and b_vec::Vector{$T} with capacity ",s.capacity)
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

Deposit a vector `buf` into the MPI window `s` on rank `targetrank`. If
`len` is given, only the first `len` elements are transmitted.
"""
@inline function put(buf::Vector{T}, len, targetrank, s::MPIOneSided{T}) where T
    @boundscheck len ≤ length(buf) && len ≤ s.capacity ||
        error("Not enough space left in buffer")
    # TODO: using Buffers onlt works on master MPI.jl
    #b_buffer = MPI.Buffer(buf, len, s.DT_b)
    MPI.Put(buf, targetrank, s.b_win)
    #l_buffer = MPI.Buffer([len,], 1, s.DT_l)
    MPI.Put([len], targetrank, s.l_win)
end
@inline function put(buf::Vector{T}, targetrank, s::MPIOneSided{T}) where T
    len = length(buf)
    put(buf, len, targetrank, s)
end

function sbuffer!(s::MPIOneSided) # unsafe version - avoids allocations
    # println("from sbuffer $(s.b_vec), $(s.l_vec), $(s.l_vec[1])")
    # return s.b_vec[1:s.l_vec[1]]
    return view(s.b_vec, 1 : s.l_vec[1])
end
function sbuffer(s::MPIOneSided) # safe version for reading shared buffer - returns local array
    MPI.Win_lock(MPI.LOCK_SHARED, s.id, 0, s.b_win)
    MPI.Win_lock(MPI.LOCK_SHARED, s.id, 0, s.l_win)
    res = s.b_vec[1 : s.l_vec[1]]
    MPI.Win_unlock(s.id,s.l_win)
    MPI.Win_unlock(s.id,s.b_win)
    return res
end

function mpi_combine_walkers!(target, source, s::MPIOneSided{P}) where P
    # now target is just a local data structure, e.g. DVec
    # allocate local buffers for sorting
    bufs = [Vector{P}(undef,length(source)) for i in 1:(s.np-1)] # type-stable
    lens = zeros(Int,(s.np-1))
    # sort source into send buffers
    @inbounds for (key,val) in pairs(source)
        tr = targetrank(key, s.np)
        if tr < s.id
            lens[tr+1] +=1
            bufs[tr+1][lens[tr+1]] = key => val
        elseif tr > s.id
            lens[tr] +=1
            bufs[tr][lens[tr]] = key => val
        else # tr == s.id
            deposit!(target, key, val, nothing)
        end
    end
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
            deposit!(target, key, val, nothing)
        end
    end
    fence(s)
    return target
end
