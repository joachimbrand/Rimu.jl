"""
    MPIData(data; kwargs...)

Wrapper used for signaling that this data is part of a distributed
data structure and communication should happen with MPI.

Keyword arguments:
* `setup = mpi_point_to_point` - controls the communication stratgy
  * [`mpi_one_sided`](@ref) uses one-sided communication with remote memory access (RMA), sets [`MPIOneSided`](@ref) strategy.
  * [`mpi_point_to_point`](@ref) uses [`MPIPointTOPoint`](@ref) strategy.
  * [`mpi_all_to_all`](@ref) uses [`MPIAllToAll`](@ref) strategy.
  * [`mpi_no_exchange`](@ref) sets [`MPINoWalkerExchange`](@ref) strategy. Experimental. Use with caution!
* `comm = mpi_comm()`
* `root = mpi_root`
* The rest of the keyword arguments are passed to `setup`.
"""
struct MPIData{D,S}
    data::D # local data, e.g. a DVec
    comm::MPI.Comm
    root::Int32 # rank of root process
    isroot::Bool # true if running on root process
    s::S # type (struct) with further details needed for communication

    function MPIData(data::D, comm, root, s::S) where {D, S<:DistributeStrategy}
        return new{D,S}(data, comm, root, s.id == root, s)
    end
end
# convenient constructor with setup function
function MPIData(data; setup=mpi_point_to_point, comm=mpi_comm(), root=mpi_root, kwargs...)
    return setup(data, comm, root; kwargs...)
end

Base.eltype(md::MPIData) = eltype(md.data)
Base.valtype(md::MPIData) = valtype(md.data)
Base.keytype(md::MPIData) = keytype(md.data)
Rimu.localpart(md::MPIData) = md.data
Rimu.StochasticStyle(d::MPIData) = Rimu.StochasticStyle(d.data)

function Base.summary(io::IO, md::MPIData)
    data = nameof(typeof(md.data))
    len = length(md)
    style = StochasticStyle(md)
    strat = nameof(typeof(md.s))
    print(io, "MPIData($data) with $len entries, style = $style, strategy = $strat")
end
function Base.show(io::IO, md::MPIData)
    summary(io, md)
    limit, _ = displaysize()
    for (i, p) in enumerate(pairs(localpart(md)))
        if length(md) > i > limit - 4
            print(io, "\n  ⋮   => ⋮")
            break
        else
            print(io, "\n  ", p)
        end
    end
end

###
### Iterators
###
"""
    MPIDataIterator{I,M<:MPIData}

Iterator over `keys`, `values`, or `pairs` of a `dv::MPIData`. Unlike its name would
suggest, it does not actually support iteration. To perform computations with it, use
`mapreduce`, or its derivatives (`sum`, `prod`, `reduce`...), which will perform the
reduction accross MPI ranks.
"""
struct MPIDataIterator{I,M<:MPIData}
    iter::I
    data::M
end

function Base.iterate(it::MPIDataIterator, args...)
    error(
        "iterating over `::MPIData` is not supported. ",
        "Use `localpart` to iterate over the local part of the vector or `mapreduce` to ",
        "perform a reduction accross ranks",
    )
end

function Base.mapreduce(f, op, it::MPIDataIterator; kwargs...)
    res = mapreduce(f, op, it.iter; kwargs...)
    return MPI.Allreduce(res, op, it.data.comm)
end

Base.pairs(data::MPIData) = MPIDataIterator(pairs(localpart(data)), data)
Base.keys(data::MPIData) = MPIDataIterator(keys(localpart(data)), data)
Base.values(data::MPIData) = MPIDataIterator(values(localpart(data)), data)

Rimu.localpart(it::MPIDataIterator) = it.iter

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
        return sqrt(sum(abs2, values(md)))
    elseif p === 1
        return float(sum(abs, values(md)))
    elseif p === Inf
        return float(mapreduce(abs, max, values(md); init=real(zero(valtype(md)))))
    else
        error("$p-norm of MPIData is not implemented.")
    end
end

"""
    walkernumber(md::MPIData)

Compute the walkernumber of the distributed data on every MPI rank with `MPI.Allreduce`.
MPI syncronizing.
"""
function Rimu.DictVectors.walkernumber(md::MPIData)
    return MPI.Allreduce(walkernumber(md.data), +, md.comm)
end

function Base.:*(lop, md::MPIData)
    T = promote_type(eltype(lop),valtype(md))
    P = Pair{keytype(md),T}
    buffers = Vector{P}[P[] for _ in 1:mpi_size()]
    datatype = MPI.Datatype(P)
    myrank = mpi_rank()
    recbuf = buffers[myrank + 1]

    result = similar(localpart(md), T)

    # Do the multiplication into the buffers.
    for (key,val) in pairs(localpart(md))
        result[key] += diagonal_element(lop, key)*val
        for (add, elem) in offdiagonals(lop, key)
            tr = targetrank(add, mpi_size())
            if tr == myrank
                result[add] += elem * val
            else
                push!(buffers[tr + 1], add => elem * val)
            end
        end
    end

    # Receive from lower ranks.
    for id in 0:(myrank - 1)
        resize!(recbuf, MPI.Get_count(MPI.Probe(id, 0, md.comm), datatype))
        MPI.Recv!(recbuf, id, 0, md.comm)
        for (add, value) in recbuf
            result[add] += value
        end
    end
    # Perform sends.
    for id in 0:(mpi_size() - 1)
        id == myrank && continue
        MPI.Send(buffers[id + 1], id, 0, md.comm)
    end
    # Receive from lower ranks.
    for id in (myrank + 1):(mpi_size() - 1)
        resize!(recbuf, MPI.Get_count(MPI.Probe(id, 0, md.comm), datatype))
        MPI.Recv!(recbuf, id, 0, md.comm)
        for (add, value) in recbuf
            result[add] += value
        end
    end
    return result
end

function LinearAlgebra.dot(x, md::MPIData)
    return MPI.Allreduce(localpart(x)⋅localpart(md), +, md.comm)
end
function LinearAlgebra.dot(x, lop, md::MPIData)
    return MPI.Allreduce(dot(x, lop, localpart(md)), +, md.comm)
end
function LinearAlgebra.dot(md_left::MPIData, lop, md_right::MPIData)
    # Arguments are swapped since lop * md_right is not an MPIData and MPIData should be on
    # the right hand side.
    return conj(dot(lop * md_right, md_left))
end
