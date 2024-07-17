"""
    PDWorkingMemoryColumn

A column in [`PDWorkingMemory`](@ref). Supports `getindex`, [`deposit!`](@ref) and
[`StochasticStyle`](@ref) and acts as a target for spawning. Can be used as a target in a
three-way dot-product.
"""
struct PDWorkingMemoryColumn{K,V,W<:AbstractInitiatorValue{V},I<:InitiatorRule,S,N}
    segments::NTuple{N,Dict{K,W}}
    initiator::I
    style::S
end
function PDWorkingMemoryColumn(t::PDVec{K,V}, style=t.style) where {K,V}
    n = total_num_segments(t.communicator, num_segments(t))
    W = initiator_valtype(t.initiator, V)

    segments = ntuple(_ -> Dict{K,W}(), n)
    return PDWorkingMemoryColumn(segments, t.initiator, style)
end

function deposit!(c::PDWorkingMemoryColumn{K,V,W}, k::K, val, parent) where {K,V,W}
    segment_id = fastrange_hash(k, num_segments(c))
    segment = c.segments[segment_id]
    new_val = get(segment, k, zero(W)) + to_initiator_value(c.initiator, k, V(val), parent)
    if iszero(new_val)
        delete!(segment, k)
    else
        segment[k] = new_val
    end
    return nothing
end

function Base.getindex(c::PDWorkingMemoryColumn{K,V,W}, k::K) where {K,V,W}
    segment_id = fastrange_hash(k, num_segments(c))
    segment = c.segments[segment_id]
    return convert(V, get(segment, k, zero(W)))
end

Base.length(c::PDWorkingMemoryColumn) = sum(length, c.segments)
Base.empty!(c::PDWorkingMemoryColumn) = foreach(empty!, c.segments)
Base.keytype(::PDWorkingMemoryColumn{K}) where {K} = K
Base.valtype(::PDWorkingMemoryColumn{<:Any,V}) where {V} = V
Base.eltype(::PDWorkingMemoryColumn{K,V}) where {K,V} = Pair{K,V}
num_segments(c::PDWorkingMemoryColumn{<:Any,<:Any,<:Any,<:Any,<:Any,N}) where {N} = N
segment_type(::Type{<:PDWorkingMemoryColumn{K,<:Any,W}}) where {K,W} = Dict{K,W}
StochasticStyle(c::PDWorkingMemoryColumn) = c.style

function DVec(c::PDWorkingMemoryColumn{K,V}) where {K,V}
    dv = DVec{K,V}(; style=c.style)
    for seg in c.segments
        for (k, v) in seg
            dv[k] = v
        end
    end
    return dv
end

"""
    PDWorkingMemory(t::PDVec)

The working memory that handles threading and MPI distribution for operations that involve
operators, such as FCIQMC propagation, operator-vector multiplication and three-way
dot products with [`PDVec`](@ref)s.

The working memory is structured as a two-dimensional array of segments, which themselves
are `Dict`s (see [`PDVec`](@ref)). The number of rows in this array is equal to the number
of segments across all MPI ranks (covering the entire address space), while the number of
columns corresponds to the number of segments in the current MPI rank (i.e. column
corresponds to the part of the address space that is local to the current rank).

The purpose of this organisation is to allow spawning in parallel without using locks or
atomic operations. The spawning is performed by applying the following sequence of
operations:

- [`perform_spawns!`](@ref): each segment in the [`PDVec`](@ref) is multiplied by the
  operator independently, with the results being stored in a column of the working memory.
- [`collect_local!`](@ref): the rows of the working memory are summed to the first column.
- [`synchronize_remote!`](@ref): the segments corresponding to other MPI ranks are
  distributed and transferred to the first column.
- [`move_and_compress!`](@ref): the results are stochastically compressed and moved to the
  result [`PDVec`](@ref)

When used with three-argument dot products, a full copy of the left-hand side vector is
materialized in the first column of the working memory on all ranks.
"""
struct PDWorkingMemory{
    K,
    V,
    W<:AbstractInitiatorValue{V},
    S<:StochasticStyle{V},
    I<:InitiatorRule,
    C<:Communicator,
    N,
}
    columns::Vector{PDWorkingMemoryColumn{K,V,W,I,S,N}}
    style::S
    initiator::I
    communicator::C
end
function PDWorkingMemory(t::PDVec{K,V,S,D,I}; style=t.style) where {K,V,S,D,I}
    if !(style isa StochasticStyle{V})
        throw(ArgumentError("Incompatible style $style given to `PDWorkingMemory`"))
    end
    nrows = total_num_segments(t.communicator, num_segments(t))
    columns = [PDWorkingMemoryColumn(t, style) for _ in 1:num_segments(t)]

    W = initiator_valtype(t.initiator, V)
    return PDWorkingMemory(columns, style, t.initiator, t.communicator)
end

function Base.show(io::IO, w::PDWorkingMemory{K,V}) where {K,V}
    print(io, "PDWorkingMemory{$K,$V} with $(length(w.columns)) columns")
end

StochasticStyle(w::PDWorkingMemory) = w.style
Base.keytype(w::PDWorkingMemory{K}) where {K} = K
Base.valtype(w::PDWorkingMemory{<:Any,V}) where {V} = V
Base.eltype(w::PDWorkingMemory{K,V}) where {K,V} = Pair{K,V}

"""
    num_rows(w::PDWorkingMemory) -> Int

Number of rows in the working memory. The number of rows is equal to the number of segments
accross all MPI ranks.

See [`PDWorkingMemory`](@ref).
"""
num_rows(w::PDWorkingMemory) = length(w.columns[1].segments)

"""
    num_columns(w::PDWorkingMemory) -> Int

Number of columns in the working memory. The number of rows is equal to the number of
segments in the local MPI rank.

See [`PDWorkingMemory`](@ref).
"""
num_columns(w::PDWorkingMemory) = length(w.columns)

function Base.length(w::PDWorkingMemory)
    result = sum(length, w.columns)
    return merge_remote_reductions(w.communicator, +, result)
end

"""
    FirstColumnIterator{W,D} <: AbstractVector{D}

Iterates segments in the first column of a working memory that belong to a specified
rank.

See [`PDWorkingMemory`](@ref), [`remote_segments`](@ref) and [`local_segments`](@ref).
"""
struct FirstColumnIterator{W,D} <: AbstractVector{D}
    working_memory::W
    rank::Int
end

"""
    remote_segments(w::PDWorkingMemory, rank_id)

Returns iterator over the segments in the first column of `w` that belong to rank
`rank_id`. Iterates `Dict`s.

See [`PDWorkingMemory`](@ref).
"""
function remote_segments(w::PDWorkingMemory, rank)
    return FirstColumnIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end

"""
    local_segments(w::PDWorkingMemory)

Returns iterator over the segments in the first column of `w` on the current rank. Iterates
`Dict`s.

See [`PDWorkingMemory`](@ref).
"""
function local_segments(w::PDWorkingMemory)
    rank = mpi_rank(w.communicator)
    return FirstColumnIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end

Base.size(it::FirstColumnIterator) = (num_columns(it.working_memory),)

function Base.getindex(it::FirstColumnIterator, index)
    row_index = index + it.rank * num_columns(it.working_memory)
    return it.working_memory.columns[1].segments[row_index]
end

# Internal function to ensure type stability in `perform_spawns!`
function _spawn_column!(ham, column, segment, boost)
    empty!(column)
    _, stats = step_stats(column.style)
    for (k, v) in segment
        stats += apply_column!(column, ham, k, v, boost)
    end
    return stats
end

"""
    perform_spawns!(w::PDWorkingMemory, v::PDVec, ham, boost)

Perform spawns from `v` through `ham` to `w`. `boost` increases the number of spawns without
affecting the expectation value of the process.

See [`PDVec`](@ref) and [`PDWorkingMemory`](@ref).
"""
function perform_spawns!(w::PDWorkingMemory, t::PDVec, ham, boost)
    if num_columns(w) ≠ num_segments(t)
        error("working memory incompatible with vector")
    end
    stat_names, init_stats = step_stats(w.style)
    stats = Folds.sum(zip(w.columns, t.segments); init=init_stats) do (column, segment)
        _spawn_column!(ham, column, segment, boost)
    end ::typeof(init_stats)
    return stat_names, stats
end

"""
    collect_local!(w::PDWorkingMemory)

Sum each row in `w` and store the result in the first column. This step must be performed
before using [`local_segments`](@ref) or [`remote_segments`](@ref) to move the values
elsewhere.

See [`PDWorkingMemory`](@ref).
"""
function collect_local!(w::PDWorkingMemory)
    ncols = num_columns(w)
    Folds.foreach(1:num_rows(w)) do i
        for j in 2:ncols
            dict_add!(w.columns[1].segments[i], w.columns[j].segments[i])
        end
    end
end

"""
    synchronize_remote!([::Communicator,] w::PDWorkingMemory) -> names, values

Synchronize non-local segments across MPI and add the results to the first
column. Controlled by the [`Communicator`](@ref). This can only be perfomed after
[`collect_local!`](@ref).

Should return a `Tuple` of names and a `Tuple` of values to report.

See [`PDWorkingMemory`](@ref).
"""
function synchronize_remote!(w::PDWorkingMemory)
    synchronize_remote!(w.communicator, w)
end

"""
    move_and_compress!(dst::PDVec, src::PDWorkingMemory)
    move_and_compress!(::CompressionStrategy, dst::PDVec, src::PDWorkingMemory)

Move the values in `src` to `dst`, compressing the according to the
[`CompressionStrategy`](@ref) on the way. This step can only be performed after
[`collect_local!`](@ref) and [`synchronize_remote!`](@ref).

See [`PDWorkingMemory`](@ref).
"""
function move_and_compress!(dst::PDVec, src::PDWorkingMemory)
    compression = CompressionStrategy(StochasticStyle(src))
    stat_names, init = step_stats(compression)
    stats = Folds.mapreduce(add, dst.segments, local_segments(src); init) do dst_seg, src_seg
        empty!(dst_seg)
        compress!(
            compression, dst_seg,
            (from_initiator_value(src.initiator, v) for (k, v) in pairs(src_seg)),
        )
    end
    return dst, stat_names, stats
end

"""
    first_column(::PDWorkingMemory)

Return the first column of the working memory. This is where the vectors are collected
with [`collect_local!`](@ref), [`synchronize_remote!`](@ref), [`copy_to_local!`](@ref).

See [`PDWorkingMemory`](@ref).
"""
function first_column(w::PDWorkingMemory{K,V,W,S}) where {K,V,W,S}
    return w.columns[1]
end

function copy_to_local!(w, v::AbstractDVec)
    return v
end

function copy_to_local!(w, t::PDVec)
    return copy_to_local!(w.communicator, w, t)
end

working_memory(t::PDVec) = PDWorkingMemory(t)

function Interfaces.apply_operator!(
    working_memory::PDWorkingMemory, target::PDVec, source::PDVec, ham, boost=1,
)
    stat_names, stats = perform_spawns!(working_memory, source, ham, boost)
    collect_local!(working_memory)
    sync_stat_names, sync_stats = synchronize_remote!(working_memory)
    target, comp_stat_names, comp_stats = move_and_compress!(target, working_memory)

    stat_names = (stat_names..., comp_stat_names..., sync_stat_names...)
    stats = (stats..., comp_stats..., sync_stats...)

    return stat_names, stats, working_memory, target
end
