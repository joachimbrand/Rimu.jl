"""
    PDWorkingMemoryColumn

A column in [`PDWorkingMemory`](@ref). Supports [`deposit!`](@ref) and
[`StochasticStyle`](@ref) and acts as a target for spawning.
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

"""
    PDWorkingMemory(t::PDVec)

The working memory that handles threading and MPI distribution for operations that involve
operators, such as FCIQMC propagation, operator-vector multiplication and three-way
dot products with [`PDVec`](@ref)s.

The working memory is structured in a series of columns, where each has a number of segments
(see [`PDVec`](@ref)) equal to the number of segments across all MPI ranks. The purpose of
this organisation is to allow spawning in parallel without using locks or atomic operations.

The steps performed on a `PDWorkingMemory` during a typical operation are
[`perform_spawns!`](@ref), [`collect_local!`](@ref), [`synchronize_remote!`](@ref), and
[`move_and_compress!`](@ref).

When used with three-argument dot products, a full copy of the left-hand side vector is
materialized in the first column of the working memory on all ranks.
"""
struct PDWorkingMemory{
    K,V,W<:AbstractInitiatorValue{V},S<:StochasticStyle{V},I<:InitiatorRule,C<:Communicator,N
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
accross all ranks.
"""
num_rows(w::PDWorkingMemory) = length(w.columns[1].segments)

"""
    num_columns(w::PDWorkingMemory) -> Int

Number of colums in the working memory. The number of rows is equal to the number of
segments in the local rank.
"""
num_columns(w::PDWorkingMemory) = length(w.columns)

function Base.length(w::PDWorkingMemory)
    result = sum(length, w.columns)
    return merge_remote_reductions(w.communicator, +, result)
end

"""
    MainSegmentIterator{W,D} <: AbstractVector{D}

Iterates the main segments of a specified rank. See [`remote_segments`](@ref) and
[`local_segments`](@ref).
"""
struct MainSegmentIterator{W,D} <: AbstractVector{D}
    working_memory::W
    rank::Int
end

"""
    remote_segments(w::PDWorkingMemory, rank_id)

Iterate over the main segments that belong to rank `rank_id`. Iterates `Dict`s.
"""
function remote_segments(w::PDWorkingMemory, rank)
    return MainSegmentIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end

"""
    local_segments(w::PDWorkingMemory)

Iterate over the main segments on the current rank. Iterates `Dict`s.
"""
function local_segments(w::PDWorkingMemory)
    rank = mpi_rank(w.communicator)
    return MainSegmentIterator{typeof(w),segment_type(eltype(w.columns))}(w, rank)
end

Base.size(it::MainSegmentIterator) = (num_columns(it.working_memory),)

function Base.getindex(it::MainSegmentIterator, index)
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
    perform_spawns!(w::PDWorkingMemory, t::PDVec, ham, boost)

Perform spawns from `t` through `ham` to `w`.
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

Collect each row in `w` into its main segment. This step must be performed before using
[`local_segments`](@ref) or [`remote_segments`](@ref) to move the values elsewhere.
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
    synchronize_remote!(w::PDWorkingMemory)

Synchronize non-local segments across MPI. Controlled by the [`Communicator`](@ref). This
can only be perfomed after [`collect_local!`](@ref).
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
"""
function move_and_compress!(dst::PDVec, src::PDWorkingMemory)
    compression = CompressionStrategy(StochasticStyle(src))
    Folds.foreach(dst.segments, local_segments(src)) do dst_seg, src_seg
        empty!(dst_seg)
        # TODO: this does not collect the correct stats
        # it's also messy as calling pairs on the generator seems to do the correct thing
        compress!(
            compression, dst_seg,
            (from_initiator_value(src.initiator, v) for (k, v) in pairs(src_seg)),
        )
    end
    return dst
end

"""
    main_column(::WorkingMemory) -> PDVec

Return the "main" column of the working memory wrapped in a [`PDVec`](@ref).
"""
function main_column(w::PDWorkingMemory{K,V,W,S}) where {K,V,W,S}
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
    working_memory::PDWorkingMemory, target::PDVec, source::PDVec, ham,
    boost=1, compress::Val{C}=Val(true)
) where {C}

    stat_names, stats = perform_spawns!(working_memory, source, ham, boost)
    collect_local!(working_memory)
    synchronize_remote!(working_memory)
    target = move_and_compress!(target, working_memory)

    return stat_names, stats, working_memory, target
end
