# import MPI, LinearAlgebra

const mpi_registry = Dict{Int,Any}()
# The registry keeps references to distributed structures that must not be
# garbage collected at random

"""
    next_mpiID()
Produce a new ID number for MPI distributed objects. Uses an internal counter.
"""
next_mpiID

let mpiID::Int = 0
    global next_mpiID
    next_mpiID() = (mpiID +=1; mpiID)
end


abstract type DistributeStrategy end

"""
Simple wrapper used for signaling that this data is part of a distributed
data structure and communication should happen with MPI.
"""
struct MPIData{D,S}
    data::D # local data, e.g. a DVec
    comm::MPI.Comm
    root::Int32 # rank of root process
    isroot::Bool # true if running on root process
    s::S # type (struct) with further details needed for communication

    function MPIData(data::D, comm, root, s::S) where {D, S<:DistributeStrategy}
        return new{D,S}(data, comm, root, s.id==root, s)
    end
end

localpart(dv) = dv # default for local data
localpart(md::MPIData) = md.data

"""
    MPIDefault(nprocs, my_rank, comm)
Strategy for point-to-point MPI communication.
"""
struct MPIDefault <: DistributeStrategy
    np::Int32
    id::Int32
    comm::MPI.Comm
end


"""
    MPIOSWin(nprocs, myrank, comm, ::Type{T}, capacity)
Communication buffer for use with MPI one-sided communication (remote memory
access). Up to `capacity` elements of type `T` can be exchanged between MPI
ranks via [`put`](@ref). It is important that `isbitstype(T) == true`.
Objects of type `MPIOSWin` have to be freed manually with a (blocking) call
to [`free()`](@ref).
"""
mutable struct MPIOSWin{T}  <: DistributeStrategy
    mpiid::Int # unique ID for MPI-distributed objects
    np::Int32 # number of MPI processes
    id::Int32 # number of this rank
    comm::MPI.Comm # MPI communicator
    b_win::MPI.Win # window for data buffer
    l_win::MPI.Win # window for length of buffered data
    capacity::Int32 # capacity of buffer
    buf::Vector{T} # local array for MPI window b_win
    n_elem::Vector{Int32} # local array for MPI window l_win

    function MPIOSWin(nprocs, myrank, comm, ::Type{T}, capacity) where T
        @assert isbitstype(T) "Buffer type for MPIOSWin needs to be isbitstype; found $T"
        buf = Vector{T}(undef, capacity)
        n_elem = zeros(Int32,1)
        b_win = MPI.Win_create(buf, comm) # separate windows for buffer and length
        l_win = MPI.Win_create(n_elem, comm)
        mpiid = next_mpiID()
        obj = new{T}(mpiid, nprocs, myrank, comm, b_win, l_win, capacity, buf, n_elem)
        MPI.refcount_inc() # required according to MPI.jl docs
        mpi_registry[mpiid] = Ref(obj) # register the object to avoid
        # arbitrary garbage collection
        ccall(:jl_, Cvoid, (Any,), "installing finalizer on MPIOSWin")
        # @async println("Installing finaliser for MPIOSWin on rank ", obj.id)
        finalizer(myclose, obj) # install finalizer
        return obj
    end
end
Base.eltype(::MPIOSWin{T}) where T = T
"""
    length(md::MPIData)
Compute the length of the distributed data on every MPI rank with
`MPI.Allreduce`.
"""
Base.length(md::MPIData) = MPI.Allreduce(length(md.data), +, md.comm)
"""
    norm(md::MPIData, p=2)
Compute the norm of the distributed data on every MPI rank with `MPI.Allreduce`.
"""
function LinearAlgebra.norm(md::MPIData, p::Real=2)
    if p === 2
        return sqrt(MPI.Allreduce(Rimu.DictVectors.norm_sqr(md.data), +, md.comm))
    elseif p === 1
        return MPI.Allreduce(Rimu.DictVectors.norm1(md.data), +, md.comm)
    elseif p === Inf
        return MPI.Allreduce(Rimu.DictVectors.normInf(md.data), max, md.comm)
    else
        @error "$p-norm of MPIData is not implemented."
    end
end


"""
    mpi_default(data, comm = MPI.COMM_WORLD, root = 0)
Declare `data` as mpi-distributed and set communication strategy to default.
"""
function mpi_default(data, comm = MPI.COMM_WORLD, root = 0)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIDefault(np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
    mpi_one_sided(data, comm = MPI.COMM_WORLD, root = 0)
Declare `data` as mpi-distributed and set communication strategy to one-sided
with remote memory access (RMA).
"""
function mpi_one_sided(data, comm = MPI.COMM_WORLD, root = 0)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    P = pairtype(data)

    # compute the required capacity for the communication buffer as a
    # fraction of the capacity of `data`
    cap = capacity(data) ÷ np + 1
    id == root && println("on rank $id, capacity = ",cap)
    s = MPIOSWin(np, id, comm, P, Int32(cap))
    return MPIData(data, comm, root, s)
end

function myclose(obj::MPIOSWin)
    ccall(:jl_, Cvoid, (Any,), "running finalizer on MPIOSWin")
    # @async println("Finalising MPIOSWin on rank ", obj.id)
    MPI.free(obj.b_win)
    MPI.free(obj.l_win)
    MPI.refcount_dec()
    return nothing
end

"""
    free(obj::MPIOSWin)
De-reference the object, call finalizer and the garbage collector immediately.
This is a syncronizing MPI call. Make sure that the object is not used later.
"""
function free(obj::MPIOSWin)
    global mpi_registry
    MPI.Barrier(obj.comm)
    delete!(mpi_registry, obj.mpiid)
    finalize(obj)
    GC.gc()
end
function free(d::MPIData{D,S}) where {D, S<:MPIOSWin}
    free(d.s)
    finalize(d)
    GC.gc()
end

function fence(s::MPIOSWin, assert = 0)
    MPI.Win_fence(assert, s.b_win)
    MPI.Win_fence(assert, s.l_win) # joint fences
end
"""
    put(buf::Vector{T}, [len,] targetrank, s::MPIOSWin{T})
    put(obj::T, targetrank, s::MPIOSWin{T})
Deposit a single `obj` or vector `buf` into the MPI window `s` on
rank `targetrank`. If `len` is given, only the first `len` elements are
transmitted.
"""
@inline function put(buf::Vector{T}, len::Integer, targetrank::Integer,
                    s::MPIOSWin{T}) where T
    @boundscheck len ≤ length(buf) && len ≤ s.capacity
    MPI.Put(buf, len, targetrank, 0, s.b_win)
    MPI.Put(Ref(len), 1, targetrank, 0, s.l_win)
end
@inline function put(buf::Vector{T}, targetrank::Integer, s::MPIOSWin{T}) where T
    len = length(buf)
    put(buf, len, targetrank, s)
end
@inline function put(obj::T, targetrank::Integer, s::MPIOSWin{T}) where T
    len = 1
    @boundscheck len ≤ s.capacity # length is 1
    MPI.Put(Ref(obj), len, targetrank, 0, s.b_win)
    MPI.Put(Ref(len), 1, targetrank, 0, s.l_win)
end


function sbuffer!(s::MPIOSWin) # unsafe version - avoids allocations
    # println("from sbuffer $(s.buf), $(s.n_elem), $(s.n_elem[1])")
    # return s.buf[1:s.n_elem[1]]
    return view(s.buf, 1 : s.n_elem[1])
end

function sbuffer(s::MPIOSWin) # safe version for reading shared buffer - returns local array
    MPI.Win_lock(MPI.LOCK_SHARED, s.id, 0, s.b_win)
    MPI.Win_lock(MPI.LOCK_SHARED, s.id, 0, s.l_win)
    res = s.buf[1 : s.n_elem[1]]
    MPI.Win_unlock(s.id,s.l_win)
    MPI.Win_unlock(s.id,s.b_win)
    return res
end

"""
    targetrank(key, np, hash = hash(key))
Compute the rank where the `key` belongs.
"""
targetrank(key, np, hash = hash(key)) = hash%np

"""
    sort_into_targets!(target::MPIData, source)
Distribute the entries of `source` to the `target` data structure such that all
entries in the `target` dictionaries are on the process with the correct rank
as controlled by [`targetrank()`](@ref).
"""
function sort_into_targets!(dtarget::MPIData, source::AbstractDVec)
    ltarget = localpart(dtarget)
    empty!(ltarget) # start with empty slate
    strategy = dtarget.s
    P = pairtype(source) # compute pairtype
    # println("pairtype = ",P)
    sort_into_targets!(ltarget, source, P, strategy)
end

function sort_into_targets!(target, source, ::Type{P}, s::DistributeStrategy) where P
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
            target[key] += val # local: just copy to target
        end
    end
    # call strategy-specific method:
    return sort_into_targets!(target, bufs, lens, P, s)
end
function sort_into_targets!(target, bufs::Vector{Vector{P}}, lens, ::Type{P}, s::MPIDefault) where P
    # use standard MPI message passing communication
    # use ring structure for sending around data with blocking communications:
    # first receive from lower ranks, then send, then recieve from higher ranks
    # receiving from lower ranks
    for r = 0:(s.id-1)
        status = MPI.Probe(r, 0, s.comm)
        count = MPI.Get_count(status, P) # how many pairs are ready
        rbuf = Vector{P}(undef,count) # allocate buffer of correct size
        MPI.Recv!(rbuf, r, 0, s.comm)
        for (key, val) in rbuf # sort into target dict right away
            target[key] += val
        end
    end
    # send all buffer entries moving to higher ranks
    for tr in (s.id + 1) : (s.np - 1)
        # s.id == 0 && println("sending to rank $tr")
        sstat = MPI.Send(view(bufs[tr],1:lens[tr]), tr, 0, s.comm)
        # println("sent from $(s.id) to $tr with status $sstat")
    end
    for tr in 0 : (s.id - 1)
        # s.id == 0 && println("sending to rank $tr")
        sstat = MPI.Send(view(bufs[tr+1],1:lens[tr+1]), tr, 0, s.comm)
        # println("sent from $(s.id) to $tr with status $sstat")
    end
    # receiving from higher ranks
    for r = (s.id+1):(s.np-1)
        status = MPI.Probe(r, 0, s.comm)
        count = MPI.Get_count(status, P) # how many pairs are ready
        rbuf = Vector{P}(undef,count) # allocate buffer of correct size
        MPI.Recv!(rbuf, r, 0, s.comm)
        for (key, val) in rbuf # sort into target dict right away
            target[key] += val
        end
    end

    # s.id == 0 && println("receiving done")
    MPI.Barrier(s.comm)
    return target
end # sort_into_targets! MPIDefault

function sort_into_targets!(target, bufs::Vector{Vector{P}}, lens, ::Type{P}, s::MPIOSWin) where P
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
end # MPIOSWin
