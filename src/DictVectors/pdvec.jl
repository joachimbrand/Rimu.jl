"""
    fastrange_hash(k, n)

Map `k` to to bucket in range 1:n. See [fastrange](https://github.com/lemire/fastrange).
"""
function fastrange_hash(k, n::Int)
    h = hash(k)
    return (((h % UInt128) * (n % UInt128)) >> 64) % Int + 1
end

"""
    PDVec{K,V}(; kwargs...)
    PDVec(iter; kwargs...)
    PDVec(pairs...; kwargs...)

Dictionary-based vector-like data structure for use with FCIQMC and
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl). While mostly behaving like a `Dict`,
it supports various linear algebra operations such as `norm` and `dot`.

The P in `PDVec` stands for parallel. `PDVec`s perform `mapreduce`, `foreach`, and various
linear algebra operations in a threaded manner. If MPI is available, these operations are
automatically distributed as well. As such it is not recommended to iterate over `pairs`,
`keys`, or `values` directly unless explicitly performing them on the [`localpart`](@ref) of
the vector.

## Keyword arguments

* `style = `[`default_style`](@ref)`(V)`: A [`StochasticStyle`](@ref) that is used to select
  the spawning strategy in the FCIQMC algorithm.

* `initiator = `[`NoInitiator`](@ref)`()`: An [`InitiatorRule`](@ref), used in FCIQMC to
  remove the sign problem.

* `communicator`: A [`Communicator`](@ref) that controls how operations are performed when
  using MPI. The defaults are [`NotDistributed`](@ref) when not using MPI and
  [`PointToPoint`](@ref) when using MPI.

* `num_segments = Threads.nthreads()`: Number of segments to divide the vector into. This is
  best left at its default value. See the extended help for more info.

* `executor = Folds.ThreadedEx()`: Experimental. Change the threaded executor to use. See
  [FoldsThreads.jl](https://juliafolds.github.io/FoldsThreads.jl/dev/) for more info on
  executors.

# Extended Help

## Segmentation

The vector is split into `num_segments` subdictionaries called segments. Which dictionary a
key-value pair is mapped to is determined by the hash of the key. The purpose of this
segmentation is to allow parallel processing - functions such as `mapreduce`, `add!` or
`dot` (full list below) process each subdictionary on a separate thread.

For parallel binary operations, the numbers of segments in both vectors must match. To
ensure this, it is best to leave the number of segments at its default value.

### Example

```julia
julia> add = FermiFS2C((1,1,0,0), (0,0,1,1));

julia> op = HubbardMom1D(add; t=4/π^2, u=4);

julia> pv = PDVec(add => 1.0)
1-element PDVec: style = IsDeterministic{Float64}()
  fs"|↑↑↓↓⟩" => 1.0

julia> pv = op * pv
7-element PDVec: style = IsDeterministic{Float64}()
  fs"|↑↓↑↓⟩" => 1.0
  fs"|↑↑↓↓⟩" => 4.0
  fs"|↓↑↓↑⟩" => 1.0
  fs"|↓↑↑↓⟩" => -1.0
  fs"|⇅⋅⋅⇅⟩" => 1.0
  fs"|↑↓↓↑⟩" => -1.0
  fs"|⋅⇅⇅⋅⟩" => 1.0

julia> map!(x -> -x, values(pv)); pv
7-element PDVec: style = IsDeterministic{Float64}()
  fs"|↑↓↑↓⟩" => -1.0
  fs"|↑↑↓↓⟩" => -4.0
  fs"|↓↑↓↑⟩" => -1.0
  fs"|↓↑↑↓⟩" => 1.0
  fs"|⇅⋅⋅⇅⟩" => -1.0
  fs"|↑↓↓↑⟩" => 1.0
  fs"|⋅⇅⇅⋅⟩" => -1.0

julia> dest = similar(pv)
0-element PDVec: style = IsDeterministic{Float64}()

julia> map!(x -> x + 2, dest, values(pv))
7-element PDVec: style = IsDeterministic{Float64}()
  fs"|↑↓↑↓⟩" => 1.0
  fs"|↑↑↓↓⟩" => -2.0
  fs"|↓↑↓↑⟩" => 1.0
  fs"|↓↑↑↓⟩" => 3.0
  fs"|⇅⋅⋅⇅⟩" => 1.0
  fs"|↑↓↓↑⟩" => 3.0
  fs"|⋅⇅⇅⋅⟩" => 1.0

julia> sum(values(pv))
-6.0

julia> dot(dest, pv)
10.0

julia> dot(dest, op, pv)
44.0
```

## MPI

When MPI is active, all parallel reductions are automatically reduced across MPI ranks
with a call to `MPI.Allreduce`.

In a distributed setting, `PDVec` does not support iteration without first making it
explicit the iteration is only to be performed on the local segments of the vector. This is
done with [`localpart`](@ref). In general, even when not using MPI, it is best practice to
use [`localpart`](@ref) when explicit iteration is required.

## Use with KrylovKit

`PDVec` can be used with [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) to find
the eigenpairs of an operator in a matrix-free manner. For best performance, wrap the
operator in `DictVectors.OperatorMulPropagator` first (see example below).

Performing the `eigsolve` this way allow it to be run in a threaded and distributed manner.
Using multiple MPI ranks with this method does not distribute the memory load effectively,
but does result in significant speedups.

### Example

```julia
julia> using KrylovKit

julia> add = BoseFS((0,0,5,0,0));

julia> op = HubbardMom1D(add; u=6.0);

julia> pv = PDVec(add => 1.0);

julia> propagator = DictVectors.OperatorMulPropagator(op, pv);

julia> results = eigsolve(propagator, pv, 4, :SR; issymmetric=true);

julia> results[1][1:4]
4-element Vector{Float64}:
 -3.4311156892322234
  1.1821748602612363
  3.7377753753082823
  6.996390417443125
```

## Parallel functionality

The following functions are parallelised and MPI-compatible:

* [`mapreduce`](@ref) and derivatives (`sum`, `prod`, `reduce`...),
* [`all`](@ref), [`any`](@ref),
* [`map!`](@ref) (on values only),
* `rmul!`, `lmul!`, `mul!`, `*`,
* `add!`, `axpy!`, `axpby!`, `+`, `-`,
* `dot`,
* `*`, [`mul!`](@ref) and [`dot!`](@ref) with operators.
"""
struct PDVec{
    K,V,S<:StochasticStyle{V},I<:InitiatorRule,C<:Communicator,E
} <: AbstractDVec{K,V}
    segments::Vector{Dict{K,V}}
    style::S
    initiator::I
    communicator::C
    executor::E
end

function PDVec{K,V}(
    ; style=default_style(V), num_segments=Threads.nthreads(),
    initiator_threshold=0, initiator=initiator_threshold > 0,
    communicator=nothing,
    executor=nothing,
) where {K,V}
    W = eltype(style)
    segments = [Dict{K,W}() for _ in 1:num_segments]

    if initiator == false
        irule = NoInitiator()
    elseif initiator == true
        initiator_threshold = initiator_threshold == 0 ? 1 : initiator_threshold
        irule = Initiator(initiator_threshold)
    elseif initiator isa InitiatorRule
        irule = initiator
    else
        throw(ArgumentError("Invalid initiator $initiator"))
    end

    # This is a bit clunky. If you modify the communicator by hand, you have to make sure it
    # knows to hold values of type W. When we introduce more communicators, they should
    # probably be constructed by a function, similar to how it's done in RMPI.
    IW = initiator_valtype(irule, W)
    if isnothing(communicator)
        if MPI.Comm_size(MPI.COMM_WORLD) > 1
            comm = PointToPoint{K,IW}()
        else
            comm = NotDistributed()
        end
    elseif communicator isa Communicator
        comm = communicator
    else
        throw(ArgumentError("Invalid communicator $communicator"))
    end

    if isnothing(executor)
        if Threads.nthreads() == 1 || num_segments == 1
            ex = NonThreadedEx()
        else
            ex = ThreadedEx()
        end
    else
        ex = executor
    end

    return PDVec(segments, style, irule, comm, ex)
end
function PDVec(pairs; kwargs...)
    K = typeof(first(pairs)[1])
    V = typeof(first(pairs)[2])
    t = PDVec{K,V}(; kwargs...)
    for (k, v) in pairs
        t[k] = v
    end
    return t
end
function PDVec(pairs::Vararg{Pair}; kwargs...)
    return PDVec(pairs; kwargs...)
end
function PDVec(dict::Dict{K,V}; kwargs...) where {K,V}
    t = PDVec{K,V}(; kwargs...)
    for (k, v) in dict
        t[k] = v
    end
    return t
end
function PDVec(dv::AbstractDVec{K,V}; style=StochasticStyle(dv), kwargs...) where {K,V}
    t = PDVec{K,V}(; style, kwargs...)
    return copyto!(t, dv)
end

function Base.summary(io::IO, t::PDVec)
    len = length(t)
    print(io, "$len-element")
    if num_segments(t) ≠ Threads.nthreads()
        print(io, ", ", num_segments(t), "-segment")
    end
    if t.communicator isa LocalPart
        print(io, " localpart(PDVec): ")
        comm = t.communicator.communicator
    else
        print(io, " PDVec: ")
        comm = t.communicator
    end
    print(io, "style = ", t.style)
    if t.initiator ≠ NoInitiator()
        print(io, ", initiator=", t.initiator)
    end
    if comm ≠ NotDistributed()
        print(io, ", communicator=", t.communicator)
    end
end

###
### Properties and utilities
###
"""
    is_distributed(t::PDVec)

Return true if `t` is MPI-distributed.
"""
is_distributed(t::PDVec) = is_distributed(t.communicator)

"""
    num_segments(t::PDVec)

Return the number of segments in `t`.
"""
num_segments(t::PDVec) = length(t.segments)

StochasticStyle(t::PDVec) = t.style

function Base.length(t::PDVec)
    result = sum(length, t.segments)
    return merge_remote_reductions(t.communicator, +, result)
end

Base.isempty(t::PDVec) = iszero(length(t))

"""
    are_compatible(t, u)

Return true if `t` and `u` have the same number of segments and show a warning otherwise.
"""
function are_compatible(t, u)
    if num_segments(t) == num_segments(u)
        return true
    elseif !is_distributed(t) && !is_distributed(u)
        @warn string(
            "vectors have different numbers of segments. ",
            "This prevents parallelization.",
        ) maxlog=1
        return false
    else
        throw(ArgumentError(
            "vectors have different numbers of segments. ",
            "This is not supported when using MPI."
        ))
    end
end

function Base.isequal(t::PDVec, u::PDVec)
    if length(localpart(t)) == length(localpart(u))
        if are_compatible(t, u)
            result = Folds.all(zip(t.segments, u.segments), u.executor) do (t_seg, u_seg)
                isequal(t_seg, u_seg)
            end
        else
            result = Folds.all(u.segments, u.executor) do seg
                for (k, v) in seg
                    isequal(t[k], v) || return false
                end
                return true
            end
        end
    else
        result = false
    end
    return merge_remote_reductions(t.communicator, &, result)
end

"""
     target_segment(t::PDVec, key) -> target, is_local

Determine the target segment from `key` hash. For MPI distributed vectors, this may return
numbers that are out of range and `is_local=false`.
"""
function target_segment(t::PDVec{K}, k::K) where {K}
    return target_segment(t.communicator, k, num_segments(t))
end

###
### getting and setting
###
function Base.getindex(t::PDVec{K,V}, k::K) where {K,V}
    segment_id, is_local = target_segment(t, k)
    if is_local
        return get(t.segments[segment_id], k, zero(V))
    else
        error("Attempted to access non-local key `$k`")
    end
end
function Base.setindex!(t::PDVec{K,V}, val, k::K) where {K,V}
    v = V(val)
    segment_id, is_local = target_segment(t, k)
    if is_local
        if iszero(v)
            delete!(t.segments[segment_id], k)
        else
            t.segments[segment_id][k] = v
        end
    end
    return v
end
function deposit!(t::PDVec{K,V}, k::K, val, parent=nothing) where {K,V}
    iszero(val) && return nothing
    segment_id, is_local = target_segment(t, k)
    if is_local
        segment = t.segments[segment_id]
        new_val = get(segment, k, zero(V)) + V(val)
        if iszero(new_val)
            delete!(segment, k)
        else
            segment[k] = new_val
        end
    end
    return nothing
end

function Base.delete!(t::PDVec{K,V}, k::K) where {K,V}
    t[k] = zero(V)
    return t
end

###
### empty(!), similar, copy, etc.
###
# TODO: this does not work with changing eltypes. Must fix the initiator,
# executor, and communicator.
function Base.empty(
    t::PDVec{K,V}; style=t.style, initiator=t.initiator, communicator=t.communicator,
    num_segments=length(t.segments)
) where {K,V}
    return PDVec{K,V}(; style, initiator, communicator, num_segments, executor=t.executor)
end
function Base.empty(t::PDVec{K}, ::Type{V}; kwargs...) where {K,V}
    return PDVec{K,V}(; kwargs..., executor=t.executor)
end
function Base.empty(t::PDVec, ::Type{K}, ::Type{V}; kwargs...) where {K,V}
    return PDVec{K,V}(; kwargs..., executor=t.executor)
end
Base.similar(t::PDVec, args...; kwargs...) = empty(t, args...; kwargs...)

function Base.empty!(t::PDVec)
    Folds.foreach(empty!, t.segments, t.executor)
    return t
end

function Base.sizehint!(t::PDVec, n)
    n_per_segment = cld(n, length(t.segments))
    Folds.foreach(d -> sizehint!(d, n_per_segment), t.segments, t.executor)
    return t
end

function Base.copyto!(dst::PDVec, src::PDVec)
    if are_compatible(dst, src)
        Folds.foreach(dst.segments, src.segments, src.executor) do d_seg, s_seg
            copy!(d_seg, s_seg)
        end
        return dst
    else
        empty!(dst)
        for (k, v) in pairs(src)
            dst[k] = v
        end
    end
    return dst
end
function Base.copy!(dst::PDVec, src::PDVec)
    return copyto!(dst, src)
end
function Base.copy(src::PDVec)
    return copy!(empty(src), src)
end

function localpart(t::PDVec{K,V,S,I,<:Any,E}) where {K,V,S,I,E}
    return PDVec{K,V,S,I,LocalPart,E}(
        t.segments, t.style, t.initiator, LocalPart(t.communicator), t.executor
    )
end
function localpart(t::PDVec{K,V,S,I,<:LocalPart,E}) where {K,V,S,I,E}
    return t
end

###
### Iterators, map, mapreduce
###
"""
    PDVecKeys
    PDVecValues
    PDVecPairs

Iterators over keys/values/pairs of the [`PDVec`](@ref). Iteration is only supported over
the `localpart`. Use reduction operations (`reduce`, `mapreduce`, `sum`, ...) if possible
when using them.
"""
struct PDVecIterator{F,T,V<:PDVec}
    selector::F
    vector::V

    function PDVecIterator(selector::F, ::Type{T}, t::V) where {F,T,V}
        return new{F,T,V}(selector, t)
    end
end

num_segments(t::PDVecIterator) = num_segments(t.vector)
is_distributed(t::PDVecIterator) = is_distributed(t.vector)
Base.eltype(t::Type{<:PDVecIterator{<:Any,T}}) where {T} = T
Base.length(t::PDVecIterator) = length(t.vector)

const PDVecKeys{T,V} = PDVecIterator{typeof(keys),T,V}
const PDVecVals{T,V} = PDVecIterator{typeof(values),T,V}
const PDVecPairs{T,V} = PDVecIterator{typeof(pairs),T,V}

Base.show(io::IO, t::PDVecKeys) = print(io, "PDVecKeys{", eltype(t), "}(...)")
Base.show(io::IO, t::PDVecVals) = print(io, "PDVecVals{", eltype(t), "}(...)")
Base.show(io::IO, t::PDVecPairs) = print(io, "PDVecPairs{", eltype(t), "}(...)")

Base.keys(t::PDVec) = PDVecIterator(keys, keytype(t), t)
Base.values(t::PDVec) = PDVecIterator(values, valtype(t), t)
Base.pairs(t::PDVec) = PDVecIterator(pairs, eltype(t), t)

function Base.iterate(t::PDVecIterator)
    if !(t.vector.communicator isa LocalPart) && !(t.vector.communicator isa NotDistributed)
        throw(CommunicatorError(
            "iteration over distributed vectors is not supported.",
            "Use `localpart` to iterate over the local part only."
        ))
    end
    return iterate(t, 1)
end
function Base.iterate(t::PDVecIterator, segment_id::Int)
    segments = t.vector.segments
    if segment_id > length(segments)
        return nothing
    end
    it = iterate(t.selector(segments[segment_id]))
    if isnothing(it)
        return iterate(t, segment_id + 1)
    else
        return it[1], (segment_id, it[2])
    end
end
function Base.iterate(t::PDVecIterator, (segment_id, state))
    segments = t.vector.segments
    it = iterate(t.selector(segments[segment_id]), state)
    if isnothing(it)
        return iterate(t, segment_id + 1)
    else
        return it[1], (segment_id, it[2])
    end
end

"""
    mapreduce(f, op, keys(::PDVec); kwargs...)
    mapreduce(f, op, values(::PDVec); kwargs...)
    mapreduce(f, op, pairs(::PDVec); kwargs...)

Perform a parallel reduction operation on [`PDVec`](@ref)s. MPI-compatible. Is used in the
definition of various functions from Base such as `reduce`, `sum`, `prod`, etc.
"""
function Base.mapreduce(f, op, t::PDVecIterator; kwargs...)
    result = Folds.mapreduce(
        op, Iterators.filter(!isempty, t.vector.segments), t.vector.executor; kwargs...
    ) do segment
        mapreduce(f, op, t.selector(segment); kwargs...)
    end
    return merge_remote_reductions(t.vector.communicator, op, result)
end

"""
    all(predicate, keys(::PDVec); kwargs...)
    all(predicate, values(::PDVec); kwargs...)
    all(predicate, pairs(::PDVec); kwargs...)

Determine whether `predicate` returns `true` for all elements of iterator on
[`PDVec`](@ref). Parallel MPI-compatible.
"""
function Base.all(f, t::PDVecIterator)
    result = Folds.all(t.vector.segments) do segment
        all(f, t.selector(segment))
    end
    return merge_remote_reductions(t.vector.communicator, &, result)
end

"""
    any(predicate, keys(::PDVec); kwargs...)
    any(predicate, values(::PDVec); kwargs...)
    any(predicate, pairs(::PDVec); kwargs...)

Determine whether `predicate` returns `true` for any element in iterator on
[`PDVec`](@ref). Parallel and MPI-compatible.
"""
function Base.any(f, t::PDVecIterator)
    result = Folds.any(t.vector.segments) do segment
        any(f, t.selector(segment))
    end
    return merge_remote_reductions(t.vector.communicator, |, result)
end

"""
    map!(f, values(::PDVec))
    map!(f, dst, values(::PDVec))

In-place parallel `map!` on values of a [`PDVec`](@ref). If `dst` is provided, results are
written there. Only defined for `values` as efficiently changing keys in a thread-safe and
distributed way is not possible.
"""
function Base.map!(f, t::PDVecVals)
    Folds.foreach(t.vector.segments, t.vector.executor) do segment
        for (k, v) in segment
            new_val = f(v)
            if !iszero(new_val)
                segment[k] = new_val
            else
                delete!(segment, k)
            end
        end
    end
    return t
end
function Base.map!(f, dst::PDVec, src::PDVecVals)
    if dst === src.vector
        map!(f, src)
    elseif are_compatible(dst, src)
        Folds.foreach(dst.segments, src.vector.segments, src.vector.executor) do d, s
            empty!(d)
            for (k, v) in s
                new_val = f(v)
                if !iszero(new_val)
                    d[k] = new_val
                end
            end
        end
    else
        empty!(dst)
        for (k, v) in pairs(src.vector)
            dst[k] = f(v)
        end
    end
    return dst
end

###
### High-level linear algebra functions
###
function LinearAlgebra.rmul!(t::PDVec, α::Number)
    if iszero(α)
        empty!(t)
    else
        map!(Base.Fix2(*, α), values(t))
    end
    return t
end
function LinearAlgebra.lmul!(α::Number, t::PDVec)
    if iszero(α)
        empty!(t)
    else
        map!(Base.Fix1(*, α), values(t))
    end
    return t
end
function LinearAlgebra.mul!(dst::PDVec, src::PDVec, α::Number)
    return map!(Base.Fix1(*, α), dst, values(src))
end

function add!(dst::PDVec, src::PDVec, α=true)
    if are_compatible(dst, src)
        Folds.foreach(dst.segments, src.segments, src.executor) do d, s
            add!(d, s, α)
        end
    else
        for (k, v) in pairs(src)
            deposit!(dst, k, α * v)
        end
    end
    return dst
end

function LinearAlgebra.dot(l::PDVec, r::PDVec)
    T = promote_type(valtype(l), valtype(r))
    if are_compatible(l, r)
        l_segs = l.segments
        r_segs = r.segments
        res = Folds.sum(zip(l_segs, r_segs), r.executor; init=zero(T)) do (l_seg, r_seg)
            sum(r_seg; init=zero(T)) do (k, v)
                conj(get(l_seg, k, zero(valtype(l_seg)))) * v
            end
        end::T
        return merge_remote_reductions(r.communicator, +, res)
    else
        res = sum(pairs(r); init=zero(T)) do (k, v)
            conj(l[k]) + v
        end
        return res
    end
end
function LinearAlgebra.dot(fd::FrozenDVec, p::PDVec)
    res = zero(promote_type(valtype(fd), valtype(p)))
    for (k, v) in pairs(fd)
        res += conj(p[k]) * v
    end
    return merge_remote_reductions(p.communicator, +, res)
end
function LinearAlgebra.dot(l::AbstractDVec, r::PDVec)
    res = sum(pairs(r)) do (k, v)
        conj(l[k]) * v
    end
    return res
end
LinearAlgebra.dot(p::PDVec, fd::FrozenDVec) = conj(dot(fd, p))
LinearAlgebra.dot(l::PDVec, r::AbstractDVec) = conj(dot(r, l))

function Base.real(v::PDVec)
    dst = similar(v, real(valtype(v)))
    return map!(real, dst, values(v))
end
function Base.imag(v::PDVec)
    dst = similar(v, real(valtype(v)))
    return map!(imag, dst, values(v))
end
