"""
Module for providing MPI functionality for `Rimu`.
"""
module RMPI

import MPI
using Rimu, LinearAlgebra, Rimu.ConsistentRNG, Random

# extending methods for:
import Base: show, length, eltype
import LinearAlgebra: dot, norm
import Rimu.ConsistentRNG: sync_cRandn, check_crng_independence
import Rimu: walkernumber, sort_into_targets!, localpart, StochasticStyle

export MPIData
export mpi_rank, is_mpi_root, @mpi_root, mpi_barrier
export mpi_comm, mpi_root, mpi_size

function __init__()
    # Initialise the MPI library once at runtime.
    MPI.Initialized() || MPI.Init()
    # make sure that MPI ranks have independent random numbers
    seedCRNG!(rand(Random.RandomDevice(),UInt) + hash(mpi_rank()))
end

const mpi_registry = Dict{Int,Any}()
# The registry keeps references to distributed structures that must not be
# garbage collected at random

# default root and comm for this package
"Default MPI root for `RMPI`."
const mpi_root = zero(Int32)

"Default MPI communicator for `RMPI`."
mpi_comm() = MPI.COMM_WORLD

"""
    mpi_size(comm = mpi_comm())
Size of MPI communicator.
"""
mpi_size(comm = mpi_comm()) = MPI.Comm_size(comm)

"""
    next_mpiID()
Produce a new ID number for MPI distributed objects. Uses an internal counter.
"""
next_mpiID

let mpiID::Int = 0
    global next_mpiID
    next_mpiID() = (mpiID +=1; mpiID)
end

"""
    mpi_rank(comm = mpi_comm())
Return the current MPI rank.
"""
mpi_rank(comm = mpi_comm()) = MPI.Comm_rank(comm)

"""
    is_mpi_root(root = mpi_root)
Returns `true` if called from the root rank
"""
is_mpi_root(root = mpi_root) = mpi_rank() == root

"""
    @mpi_root expr
Evaluate expression if on the root rank.
"""
macro mpi_root(args...)
    :(is_mpi_root() && $(esc(args...)))
end

"""
    mpi_barrier(comm = mpi_comm())
The MPI barrier with optional argument. MPI syncronizing.
"""
mpi_barrier(comm = mpi_comm()) = MPI.Barrier(comm)

abstract type DistributeStrategy end

"""
    MPIData(data; kwargs...)
Wrapper used for signaling that this data is part of a distributed
data structure and communication should happen with MPI.

Keyword arguments:
* `setup = mpi_one_sided` - controls the communication stratgy
  * [`mpi_one_sided`](@ref) uses one-sided communication with remote memory access (RMA), sets [`MPIOSWin`](@ref) strategy.
  * [`mpi_default`](@ref) uses [`MPIDefault`](@ref) strategy.
  * [`mpi_no_exchange`](@ref) sets [`MPINoWalkerExchange`](@ref) strategy. Experimental. Use with caution!
* `comm = mpi_comm()`
* `root = mpi_root`
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
# convenient constructor with setup function
function MPIData(data;
                    setup = mpi_one_sided,
                    comm = mpi_comm(),
                    root = mpi_root
                    )
    return setup(data, comm, root)
end

Base.valtype(md::MPIData{D,S}) where {D,S} = valtype(D)
Rimu.localpart(md::MPIData) = md.data
Rimu.StochasticStyle(d::MPIData) = Rimu.StochasticStyle(d.data)

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
    MPINoWalkerExchange(nprocs, my_rank, comm)
Strategy for for not exchanging walkers between ranks. Consequently there
will be no cross-rank annihilations.
"""
struct MPINoWalkerExchange <: DistributeStrategy
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
        # MPI.refcount_inc() # ref counting was removed in MPI.jl v0.16.0
        mpi_registry[mpiid] = Ref(obj) # register the object to avoid
        # arbitrary garbage collection
        # ccall(:jl_, Cvoid, (Any,), "installing finalizer on MPIOSWin")
        finalizer(myclose, obj) # install finalizer
        return obj
    end
end
Base.eltype(::MPIOSWin{T}) where T = T

"""
    length(md::MPIData)
Compute the length of the distributed data on every MPI rank with
`MPI.Allreduce`. MPI syncronizing.
"""
Base.length(md::MPIData) = MPI.Allreduce(length(md.data), +, md.comm)

"""
    norm(md::MPIData, p=2)
Compute the norm of the distributed data on every MPI rank with `MPI.Allreduce`.
MPI syncronizing.
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
    walkernumber(md::MPIData)
Compute the walkernumber of the distributed data on every MPI rank with `MPI.Allreduce`.
MPI syncronizing.
"""
function Rimu.walkernumber(md::MPIData)
    return MPI.Allreduce(walkernumber(md.data), +, md.comm)
end

###############################
# setup strategies for MPIData()

"""
    mpi_default(data, comm = mpi_comm(), root = mpi_root)
Declare `data` as mpi-distributed and set communication strategy to default.
Sets up the [`MPIData`](@ref) structure with
[`MPIDefault`](@ref) strategy.
"""
function mpi_default(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPIDefault(np, id, comm)
    return MPIData(data, comm, root, s)
end
"""
    mpi_no_exchange(data, comm = mpi_comm(), root = mpi_root)
Declare `data` as mpi-distributed and set communication strategy to
`MPINoWalkerExchange`. Sets up the [`MPIData`](@ref) structure with
[`MPINoWalkerExchange`](@ref) strategy.
"""
function mpi_no_exchange(data, comm = mpi_comm(), root = mpi_root)
    MPI.Initialized() || error("MPI needs to be initialised first.")
    np = MPI.Comm_size(comm)
    id = MPI.Comm_rank(comm)
    s = MPINoWalkerExchange(np, id, comm)
    return MPIData(data, comm, root, s)
end

"""
    mpi_one_sided(data, comm = mpi_comm(), root = mpi_root)
Declare `data` as mpi-distributed and set communication strategy to one-sided
with remote memory access (RMA).
Sets up the [`MPIData`](@ref) structure with
[`MPIOSWin`](@ref) strategy.
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
    s = MPIOSWin(np, id, comm, P, Int32(cap))
    return MPIData(data, comm, root, s)
end

function myclose(obj::MPIOSWin)
    # ccall(:jl_, Cvoid, (Any,), "running finalizer on MPIOSWin")
    MPI.free(obj.b_win)
    MPI.free(obj.l_win)
    # MPI.refcount_dec() # ref counting was removed in MPI.jl v0.16.0
    return nothing
end

"""
    free(obj::MPIOSWin)
De-reference the object, call finalizer and the garbage collector immediately.
This is a syncronizing MPI call. Make sure that the object is not used later.
MPI syncronizing.
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
function free(d::MPIData) # default
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
    ConsistentRNGs.check_crng_independence(dv::MPIData)
Does a sanity check to detect dependence of random number generators across
all MPI ranks. MPI syncronizing. Returns the size of the combined RNG state,
i.e. `mpi_size()*Threads.nthreads()*fieldcount(ConsistentRNG.CRNG)`.
"""
function ConsistentRNG.check_crng_independence(dv::MPIData) # MPI syncronizing
    # get vector of threaded RNGs on this rank
    crngs = ConsistentRNG.CRNGs[]
    # extract the numbers that make up the state of the RNGs and package into
    # an MPI-suitable buffer
    statebuffer = [getfield(rng,i) for rng in crngs for i in 1:fieldcount(typeof(rng))]
    # gather from all ranks
    combined_statebuffer = MPI.Gather(statebuffer, dv.root, dv.comm)
    # check independence
    @mpi_root @assert union(combined_statebuffer) == combined_statebuffer "Dependency in parallel rngs detected"

    @assert length(ConsistentRNG.CRNGs[]) == Threads.nthreads() "Number of CNRGs should be equal to nthreads()"
    return length(combined_statebuffer)
end


"""
    sort_into_targets!(target::MPIData, source, stats)
Distribute the entries of `source` to the `target` data structure such that all
entries in the `target` dictionaries are on the process with the correct rank
as controlled by [`targetrank()`](@ref). Combine `stats` if appropriate.
MPI syncronizing.
"""
function Rimu.sort_into_targets!(dtarget::MPIData, source::AbstractDVec)
    ltarget = localpart(dtarget)
    empty!(ltarget) # start with empty slate
    strategy = dtarget.s
    P = pairtype(source) # compute pairtype
    # println("pairtype = ",P)
    sort_into_targets!(ltarget, source, P, strategy)
end

# three-argument version
function Rimu.sort_into_targets!(dtarget::MPIData, ws::NTuple{NT,W}, statss) where {NT,W}
    # multi-threaded MPI version
    # should only ever run on thread 1
    @assert Threads.threadid() == 1 "`sort_into_targets!()` is running on `threadid()` == $(Threads.threadid()) instead of 1!"
    lwm = ws[1]
    for i in 2:NT # combine new walkers generated from different threads
        add!(lwm, ws[i])
    end
    sort_into_targets!(dtarget,lwm) # combine walkers from different MPI ranks
    stats = sum(statss) # combine stats from all threads
    MPI.Allreduce!(stats, +, dtarget.comm) # add stats from all MPI ranks
    return dtarget, ws, stats
end
function Rimu.sort_into_targets!(dtarget::MPIData, w::AbstractDVec, stats)
    # single threaded MPI version
    sort_into_targets!(dtarget,w) # combine walkers from different MPI ranks
    vstats = convert(Vector,stats) # necessary for MPI.Allreduce
    MPI.Allreduce!(vstats, +, dtarget.comm) # add stats from all MPI ranks
    return dtarget, w, vstats
end

# four-argument version
function Rimu.sort_into_targets!(target, source, ::Type{P}, s::MPINoWalkerExchange) where P
    # specific for `MPINoWalkerExchange`: copy without communicating with
    # other ranks.
    return copyto!(target, source)
end

function Rimu.sort_into_targets!(target, source, ::Type{P}, s::DistributeStrategy) where P
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
    # call strategy-specific method (with 5 arguments):
    return sort_into_targets!(target, bufs, lens, P, s)
end

# five-argument version
function Rimu.sort_into_targets!(target, bufs::Vector{Vector{P}}, lens, ::Type{P}, s::MPIDefault) where P
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

function Rimu.sort_into_targets!(target, bufs::Vector{Vector{P}}, lens, ::Type{P}, s::MPIOSWin) where P
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

function Base.show(io::IO, s::MPIOSWin{T}) where T
    println(io, "MPIOSWin{$T}")
    println(io, "  mpiid: ",s.mpiid)
    println(io, "  np: ",s.np)
    println(io, "  id: ",s.id)
    println(io, "  comm: ",s.comm)
    println(io, "  MPI.Win and buf::Vector{$T} with capacity ",s.capacity)
    println(io, "  MPI.Win for number of elements in buffer")
end

function LinearAlgebra.dot(x, md::MPIData)
    return MPI.Allreduce(x⋅localpart(md), +, md.comm)
end

function LinearAlgebra.dot(x, lop, md::MPIData)
    return MPI.Allreduce(dot(x, lop, localpart(md)), +, md.comm)
end

"""
    sync_cRandn(md::MPIData)
Generate one random number with [`cRandn()`](@ref) in a synchronous way such
that all MPI ranks have the same random number.
The argument is ignored unless it is of type `MPIData`, in which case a random
number from the root rank is broadcasted to all MPI ranks. MPI syncronizing.
"""
function Rimu.ConsistentRNG.sync_cRandn(md::MPIData)
    MPI.bcast(cRandn(), md.root, md.comm)
end

end # module RMPI
