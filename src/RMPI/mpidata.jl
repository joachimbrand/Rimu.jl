"""
    MPIData(data; kwargs...)

Wrapper used for signaling that this data is part of a distributed data structure and
communication should happen with MPI. `MPIData` can generally be used where an
[`AbstractDVec`](@ref) would be used otherwise. Unlike [`AbstractDVec`](@ref)s, `MPIData`
does not support indexing, or iteration over `keys`, `values`, and `pairs`.

Keyword arguments:
* `setup = mpi_point_to_point` - controls the communication stratgy
  * [`mpi_one_sided`](@ref) uses one-sided communication with remote memory access (RMA), sets [`MPIOneSided`](@ref) strategy.
  * [`mpi_point_to_point`](@ref) uses [`MPIPointToPoint`](@ref) strategy.
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
    entries = length(md) == 1 ? "entry" : "entries"
    print(io, "MPIData($data) with $len $entries, style = $style, strategy = $strat")
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

function Base.similar(md::MPIData)
    return MPIData(similar(md.data), md.comm, md.root, md.s)
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

Base.IteratorSize(::MPIDataIterator) = Base.SizeUnknown()
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

"""
    mpi_synchronize!(md::MPIData)

Synchronize `md`, ensuring its contents are distributed among ranks correctly.
"""
function mpi_synchronize!(md::MPIData)
    P = eltype(md)
    myrank = mpi_rank(md.comm)
    buffers = Vector{P}[P[] for _ in 1:mpi_size(md.comm)]
    for (add, val) in pairs(localpart(md))
        tr = targetrank(add, mpi_size(md.comm))
        if tr ≠ myrank
            push!(buffers[tr + 1], add => val)
            localpart(md)[add] = zero(valtype(md))
        end
    end
    mpi_communicate_buffers!(localpart(md), buffers, md.comm)
    return md
end

"""
    mpi_communicate_buffers!(target::AbstractDVec{K,V}, buffers::Vector{<:Vector{V}})

Use MPI to communicate the contents of `buffers` and sort them into `target`. The length
of `buffers` should be equal to [`mpi_size`](@ref).
"""
function mpi_communicate_buffers!(target, buffers, comm)
    myrank = mpi_rank(comm)
    recbuf = buffers[myrank + 1]
    datatype = MPI.Datatype(eltype(target))
    # Receive from lower ranks.
    for id in 0:(myrank - 1)
        resize!(recbuf, MPI.Get_count(MPI.Probe(id, 0, comm), datatype))
        MPI.Recv!(recbuf, id, 0, comm)
        for (add, value) in recbuf
            target[add] += value
        end
    end
    # Perform sends.
    for id in 0:(mpi_size(comm) - 1)
        id == myrank && continue
        MPI.Send(buffers[id + 1], id, 0, comm)
    end
    # Receive from higher ranks.
    for id in (myrank + 1):(mpi_size(comm) - 1)
        resize!(recbuf, MPI.Get_count(MPI.Probe(id, 0, comm), datatype))
        MPI.Recv!(recbuf, id, 0, comm)
        for (add, value) in recbuf
            target[add] += value
        end
    end
    return target
end

"""
    *(lop::AbstractHamiltonian, md::MPIData)
Allocating "Matrix"-"vector" multiplication with MPI-distributed "vector" `md`. The result is similar to
[`localpart(md)`](@ref) with all content having been communicated to the correct [`targetrank`](@ref).
MPI communicating.

See [`MPIData`](@ref).
"""
function Base.:*(lop, md::MPIData)
    T = promote_type(eltype(lop),valtype(md))
    P = Pair{keytype(md),T}
    buffers = Vector{P}[P[] for _ in 1:mpi_size(md.comm)]
    myrank = mpi_rank()

    result = similar(localpart(md), T)

    # Sort values into buffers and communicate.
    for (key, val) in pairs(localpart(md))
        result[key] += diagonal_element(lop, key)*val
        for (add, elem) in offdiagonals(lop, key)
            tr = targetrank(add, mpi_size(md.comm))
            if tr == myrank
                result[add] += elem * val
            else
                push!(buffers[tr + 1], add => elem * val)
            end
        end
    end
    mpi_communicate_buffers!(result, buffers, md.comm)

    return result
end

# Note: the following methods assume MPIDatas are distributed correctly.
function LinearAlgebra.dot(x, md::MPIData)
    return MPI.Allreduce(localpart(x)⋅localpart(md), +, md.comm)
end
function LinearAlgebra.dot(md::MPIData, x)
    return MPI.Allreduce(localpart(md)⋅localpart(x), +, md.comm)
end
function LinearAlgebra.dot(md_left::MPIData, md_right::MPIData)
    return MPI.Allreduce(localpart(md_left)⋅localpart(md_right), +, md_left.comm)
end

# Note: the following two methods work with x::DVec and assume `x` is the same on all ranks.
function LinearAlgebra.dot(x, lop, md::MPIData)
    return MPI.Allreduce(dot(x, lop, localpart(md)), +, md.comm)
end
function LinearAlgebra.dot(md::MPIData, lop, x)
    return MPI.Allreduce(dot(localpart(md), lop, x), +, md.comm)
end

"""
    copy_to_local(md::MPIData)

Collect all pairs in `md` from all ranks and store them in a local `AbstractDVec`.
"""
copy_to_local(md::MPIData) = copy_to_local!(similar(localpart(md)), md)

"""
    copy_to_local!(target, md::MPIData)

Collect all pairs in `md` from all ranks and store them in `target`.
In-place version of [`copy_to_local`](@ref).
"""
function copy_to_local!(target, md::MPIData)
    copy!(target, localpart(md))
    comm = md.comm
    myrank = mpi_rank(comm)
    datatype = MPI.Datatype(eltype(md))

    # Store all pairs to a buffer.
    sendbuf = collect(pairs(localpart(md)))
    recbuf = eltype(md)[]

    # Receive from lower ranks.
    for id in 0:(myrank - 1)
        resize!(recbuf, MPI.Get_count(MPI.Probe(id, 0, comm), datatype))
        MPI.Recv!(recbuf, id, 0, comm)
        for (add, value) in recbuf
            target[add] += value
        end
    end
    # Perform sends.
    for id in 0:(mpi_size(comm) - 1)
        id == myrank && continue
        MPI.Send(sendbuf, id, 0, comm)
    end
    # Receive from higher ranks.
    for id in (myrank + 1):(mpi_size(comm) - 1)
        resize!(recbuf, MPI.Get_count(MPI.Probe(id, 0, comm), datatype))
        MPI.Recv!(recbuf, id, 0, comm)
        for (add, value) in recbuf
            target[add] += value
        end
    end
    MPI.Barrier(comm)
    return target
end

function LinearAlgebra.dot(md_left::MPIData, lop, md_right::MPIData)
    if LOStructure(lop) isa IsDiagonal
        # If operator is diagonal, the operation can be performed on the local parts.
        comm = md_left.comm
        return MPI.Allreduce(dot(localpart(md_left), lop, localpart(md_right)), +, comm)
    else
        # Idea: lop * md_right can be huge. It might be better to just collect the full left
        # vector and do the multiplication locally.
        left = copy_to_local(md_left)
        return dot(left, lop, md_right)
    end
end

function Rimu.freeze(md::MPIData)
    mpi_synchronize!(md)
    return freeze(localpart(md))
end

function get_overlaps_diagonal!(names, values, operators, vecs::NTuple{N}, ::Val{B}) where {N,B}
    for i in 1:N, j in i+1:N
        if B
            push!(names, "c$(i)_dot_c$(j)")
            push!(values, dot(vecs[i], vecs[j]))
        end
        for (k, op) in enumerate(operators)
            push!(names, "c$(i)_Op$(k)_c$(j)")
            push!(values, dot(vecs[i], op, vecs[j]))
        end
    end
end
function get_overlaps_nondiagonal!(names, values, operators, vecs::NTuple{N}, ::Val{B}) where {N,B}
    local_vec_i = similar(localpart(vecs[1]))
    for i in 1:N, j in i+1:N
        if B
            push!(names, "c$(i)_dot_c$(j)")
            push!(values, dot(vecs[i], vecs[j]))
        end
        copy_to_local!(local_vec_i, vecs[i])
        for (k, op) in enumerate(operators)
            push!(names, "c$(i)_Op$(k)_c$(j)")
            push!(values, dot(local_vec_i, op, vecs[j]))
        end
    end
end

function Rimu.all_overlaps(operators::Tuple, vecs::NTuple{N,MPIData}, ::Val{B}) where {N,B}
    T = promote_type((valtype(v) for v in vecs)..., eltype.(operators)...)
    names = String[]
    values = T[]
    if all(op -> LOStructure(op) isa IsDiagonal, operators)
        get_overlaps_diagonal!(names, values, operators, vecs, Val(B))
    else
        get_overlaps_nondiagonal!(names, values, operators, vecs, Val(B))
    end

    num_reports = (N * (N - 1) ÷ 2) * (B + length(operators))
    return Tuple(SVector{num_reports,String}(names)), Tuple(SVector{num_reports,T}(values))
end
