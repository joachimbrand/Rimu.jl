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
it supports various linear algebra operations such as `norm` and `dot`, and the interface defined in [VectorInterface](https://github.com/Jutho/VectorInterface.jl).

The P in `PDVec` stands for parallel. `PDVec`s perform `mapreduce`, `foreach`, and various
linear algebra operations in a threaded manner. If MPI is available, these operations are
automatically distributed as well. As such it is not recommended to iterate over `pairs`,
`keys`, or `values` directly unless explicitly performing them on the [`localpart`](@ref) of
the vector.

See also: [`AbstractDVec`](@ref), [`DVec`](@ref), [`InitiatorDVec`](@ref).

## Keyword arguments

* `style = `[`default_style`](@ref)`(V)`: A [`StochasticStyle`](@ref) that is used to select
  the spawning strategy in the FCIQMC algorithm.

* `initiator = `[`NonInitiator`](@ref)`()`: An [`InitiatorRule`](@ref), used in FCIQMC to
  remove the sign problem.

* `communicator`: A [`Communicator`](@ref) that controls how operations are performed when
  using MPI. The defaults are [`NotDistributed`](@ref) when not using MPI and
  [`AllToAll`](@ref) when using MPI.

# Extended Help

## Segmentation

The vector is split into `Threads.nthreads()` subdictionaries called segments. Which
dictionary a key-value pair is mapped to is determined by the hash of the key. The purpose
of this segmentation is to allow parallel processing - functions such as `mapreduce`, `add!`
or `dot` (full list below) process each subdictionary on a separate thread.

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

`PDVec` is compatible with `eigsolve` from
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl). When used, the diagonalisation is
performed in a threaded and distributed manner. Using multiple MPI ranks with this method
does not distribute the memory load effectively, but does result in significant speedups.

### Example

```julia
julia> using KrylovKit

julia> add = BoseFS((0,0,5,0,0));

julia> op = HubbardMom1D(add; u=6.0);

julia> pv = PDVec(add => 1.0);

julia> results = eigsolve(op, pv, 4, :SR);

julia> results[1][1:4]
4-element Vector{Float64}:
 -3.4311156892322234
  1.1821748602612363
  3.7377753753082823
  6.996390417443125
```

## Parallel functionality

The following functions are threaded MPI-compatible:

* From Base: `mapreduce` and derivatives (`sum`, `prod`, `reduce`...), `all`,
  `any`,`map!` (on `values` only), `+`, `-`, `*`

* From LinearAlgebra: `rmul!`, `lmul!`, `mul!`, `axpy!`, `axpby!`, `dot`, `norm`,
  `normalize`, `normalize!`

* The full interface defined in
  [VectorInterface](https://github.com/Jutho/VectorInterface.jl)

"""
struct PDVec{
    K,V,N,S<:StochasticStyle{V},I<:InitiatorRule,C<:Communicator
} <: AbstractDVec{K,V}
    segments::NTuple{N,Dict{K,V}}
    style::S
    initiator::I
    communicator::C
end

function PDVec{K,V}(args...; kwargs...) where {K,V}
    return PDVec{K,V,Threads.nthreads()}(args...; kwargs...)
end
function PDVec{K,V,N}(
    ; style=default_style(V),
    initiator_threshold=0, initiator=initiator_threshold > 0,
    communicator=nothing,
) where {K,V,N}
    W = eltype(style)
    segments = ntuple(_ -> Dict{K,W}(), Val(N))

    if initiator == false
        irule = NonInitiator()
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
            comm = AllToAll{K,IW}()
        else
            comm = NotDistributed()
        end
    elseif communicator isa Communicator
        comm = communicator
    else
        throw(ArgumentError("Invalid communicator $communicator"))
    end

    return PDVec(segments, style, irule, comm)
end
function PDVec(pairs; kwargs...)
    K = eltype(first.(pairs))
    V = eltype(last.(pairs))
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
    if t.initiator ≠ NonInitiator()
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
num_segments(t::PDVec{<:Any,<:Any,N}) where {N} = N

StochasticStyle(t::PDVec) = t.style

function Base.length(t::PDVec)
    result = sum(length, t.segments)
    return merge_remote_reductions(t.communicator, +, result)
end

Base.isempty(t::PDVec) = iszero(length(t))

"""
    check_compatibility(t, u)

Return true if `t` and `u` have the same number of segments and throw otherwise.
"""
function check_compatibility(t, u)
    if num_segments(t) == num_segments(u)
        return true
    else
        throw(ArgumentError("vectors have different numbers of segments."))
    end
end

function Base.isequal(l::PDVec, r::PDVec)
    check_compatibility(l, r)
    if length(localpart(l)) == length(localpart(r))
        result = Folds.all(zip(l.segments, r.segments)) do (l_seg, r_seg)
            isequal(l_seg, r_seg)
        end
    else
        result = false
    end
    return merge_remote_reductions(l.communicator, &, result)
end

"""
     target_segment(t::PDVec, key) -> target, is_local

Determine the target segment from `key` hash. For MPI distributed vectors, this may return
numbers that are out of range and `is_local=false`.
"""
function target_segment(t::PDVec{K}, k::K) where {K}
    return target_segment(t.communicator, k, num_segments(t))
end
# Special case for single-threaded operation.
function target_segment(t::PDVec{K,<:Any,1,<:Any,<:Any,NotDistributed}, k::K) where {K}
    return 1, true
end

###
### getting and setting
###
function Base.getindex(t::PDVec{K,V}, k::K) where {K,V}
    segment_id, is_local = target_segment(t, k)
    if is_local
        return get(t.segments[segment_id], k, zero(V))
    else
        throw(CommunicatorError("Attempted to access non-local key `$k`"))
    end
end
function Base.setindex!(t::PDVec{K,V}, val, k::K) where {K,V}
    v = V(val)
    segment_id, is_local = target_segment(t, k)
    # Adding a key that is not local is supported. This is done to allow easy construction
    # of vectors even when using MPI.
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
function Base.empty(
    t::PDVec{K,V,N}; style=t.style, initiator=t.initiator, communicator=t.communicator,
) where {K,V,N}
    return PDVec{K,V,N}(; style, initiator, communicator)
end
function Base.empty(t::PDVec{K,V}, ::Type{V}; kwargs...) where {K,V}
    return empty(t; kwargs...)
end
function Base.empty(t::PDVec{K,<:Any,N}, ::Type{V}; kwargs...) where {K,V,N}
    return PDVec{K,V,N}(; kwargs...)
end
function Base.empty(t::PDVec{<:Any,<:Any,N}, ::Type{K}, ::Type{V}; kwargs...) where {K,V,N}
    return PDVec{K,V,N}(; kwargs...)
end
function Base.empty!(t::PDVec)
    Folds.foreach(empty!, t.segments, )
    return t
end
Base.similar(t::PDVec, args...; kwargs...) = empty(t, args...; kwargs...)

function Base.sizehint!(t::PDVec, n)
    n_per_segment = cld(n, length(t.segments))
    Folds.foreach(d -> sizehint!(d, n_per_segment), t.segments)
    return t
end

function Base.copyto!(dst::PDVec, src::PDVec)
    check_compatibility(dst, src)
    Folds.foreach(dst.segments, src.segments) do d_seg, s_seg
        copy!(d_seg, s_seg)
    end
    return dst
end
function Base.copy!(dst::PDVec, src::PDVec)
    return copyto!(dst, src)
end
function Base.copy(src::PDVec)
    return copy!(empty(src), src)
end

function localpart(t::PDVec{K,V,N,S,I}) where {K,V,S,I,N}
    return PDVec{K,V,N,S,I,LocalPart}(
        t.segments, t.style, t.initiator, LocalPart(t.communicator),
    )
end
function localpart(t::PDVec{<:Any,<:Any,<:Any,<:Any,<:Any,<:LocalPart})
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
the [`localpart`](@ref). Use reduction operations (`reduce`, `mapreduce`, `sum`, ...) if possible
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

"""
    PDVecKeys

Iterator over the keys of a [`PDVec`](@ref). Alias for
[`PDVecIterator`](@ref)`{typeof(keys)}`.
"""
const PDVecKeys{T,V} = PDVecIterator{typeof(keys),T,V}
"""
    PDVecVals

Iterator over the values of a [`PDVec`](@ref). Alias for
[`PDVecIterator`](@ref)`{typeof(values)}`.
"""
const PDVecVals{T,V} = PDVecIterator{typeof(values),T,V}
"""
    PDVecPairs

Iterator over the pairs of a [`PDVec`](@ref). Alias for
[`PDVecIterator`](@ref)`{typeof(pairs)}`.
"""
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
            "Direct iteration over distributed vectors is not supported.",
            "Use `localpart` to iterate over the local part only, ",
            "or use a reduction function such as mapreduce."
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
    mapreduce(f, op, keys(::PDVec); init)
    mapreduce(f, op, values(::PDVec); init)
    mapreduce(f, op, pairs(::PDVec); init)

Perform a parallel reduction operation on [`PDVec`](@ref)s. MPI-compatible. Is used in the
definition of various functions from Base such as `reduce`, `sum`, `prod`, etc.

`init`, if provided, must be a neutral element for `op`.
"""
function Base.mapreduce(f::F, op::O, t::PDVecIterator; kwargs...) where {F,O}
    result = Folds.mapreduce(
        op, Iterators.filter(!isempty, t.vector.segments); kwargs...
    ) do segment
        mapreduce(f, op, t.selector(segment))
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
    result::Bool = Folds.all(t.vector.segments) do segment
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
    result::Bool = Folds.any(t.vector.segments) do segment
        any(f, t.selector(segment))
    end
    return merge_remote_reductions(t.vector.communicator, |, result)
end

"""
    map!(f, values(::PDVec))
    map!(f, dst::PDVec, values(::PDVec))

In-place parallel `map!` on values of a [`PDVec`](@ref). If `dst` is provided, results are
written there. Only defined for `values` as efficiently changing keys in a thread-safe and
distributed way is not possible.
"""
function Base.map!(f, t::PDVecVals)
    Folds.foreach(t.vector.segments) do segment
        for (k, v) in segment
            segment[k] = f(v)
        end
        # Filtered separately to prevent messing up the dict while iterating it.
        filter!(p -> !iszero(p[2]), segment)
    end
    return t.vector
end
function Base.map!(f, dst::PDVec, src::PDVecVals)
    check_compatibility(dst, src)
    if dst === src.vector
        map!(f, src)
    else
        Folds.foreach(dst.segments, src.vector.segments) do d, s
            empty!(d)
            for (k, v) in s
                new_val = f(v)
                if !iszero(new_val)
                    d[k] = new_val
                end
            end
        end
    end
    return dst
end

"""
    map(f, values(::PDVec))

Out-of-place parallel `map` on values of a [`PDVec`](@ref). Returns a new
[`PDVec`](@ref). Only defined for `values` as efficiently changing keys in a thread-safe and
distributed way is not possible.
"""
function Base.map(f, src::PDVecVals)
    return map!(f, copy(src.vector), src)
end

"""
    filter!(pred, [dst::PDVec, ], keys(::PDVec))
    filter!(pred, [dst::PDVec, ], values(::PDVec))
    filter!(pred, [dst::PDVec, ], pairs(::PDVec))

In-place parallel `filter!` on an iterator over a [`PDVec`](@ref). If `dst` is provided,
results are written there.
"""
function Base.filter!(f, src::PDVecIterator)
    Folds.foreach(src.vector.segments) do segment
        if src.selector === pairs
            filter!(f, segment)
        elseif src.selector === keys
            filter!(p -> f(p[1]), segment)
        elseif src.selector === values
            filter!(p -> f(p[2]), segment)
        end
    end
    return src.vector
end

function Base.filter!(f, dst::PDVec, src::PDVecIterator)
    if dst === src.vector
        return filter!(f, src::PDVecIterator)
    end
    Folds.foreach(dst.segments, src.vector.segments) do d, s
        empty!(d)
        for ((k, v), x) in zip(s, src.selector(s))
            if f(x)
                d[k] = v
            end
        end
    end
    return dst
end

"""
    filter(f, keys(::PDVec))
    filter(f, values(::PDVec))
    filter(f, pairs(::PDVec))

Out-of-place parallel `filter` on an iterator over a [`PDVec`](@ref). Returns a new
[`PDVec`](@ref).
"""
function Base.filter(f, src::PDVecIterator)
    new_src = copy(src.vector)
    return filter!(f, src.selector(new_src))
end

###
### High-level linear algebra functions
###
function VectorInterface.scale!(t::PDVec, α::Number)
    if iszero(α)
        empty!(t)
    else
        map!(Base.Fix1(*, α), values(t))
    end
    return t
end
function VectorInterface.scale!(dst::PDVec, src::PDVec, α::Number)
    return map!(Base.Fix1(*, α), dst, values(src))
end

"""
    dict_add!(d::Dict, s, α=true, β=true)

Internal function similar to `add!`, but on `Dict`s. `s` can be any iterator of pairs.
"""
function dict_add!(d::Dict, s, α=true, β=true)
    if iszero(β)
        empty!(d)
    elseif β ≠ one(β)
        map!(Base.Fix1(*, β), values(d))
    end
    for (key, s_value) in s
        d_value = get(d, key, zero(valtype(d)))
        new_value = d_value + α * s_value
        if iszero(new_value)
            delete!(d, key)
        else
            d[key] = new_value
        end
    end
    return d
end

function VectorInterface.add!(dst::PDVec, src::PDVec, α::Number=true, β::Number=true)
    check_compatibility(dst, src)
    Folds.foreach(dst.segments, src.segments) do d, s
        dict_add!(d, s, α, β)
    end
    return dst
end

function VectorInterface.inner(l::PDVec, r::PDVec)
    check_compatibility(l, r)
    if l === r
        return sum(abs2, values(l); init=zero(scalartype(l)))
    else
        T = promote_type(valtype(l), valtype(r))
        l_segs = l.segments
        r_segs = r.segments
        res = Folds.sum(zip(l_segs, r_segs); init=zero(T)) do (l_seg, r_seg)
            sum(r_seg; init=zero(T)) do (k, v)
                conj(get(l_seg, k, zero(valtype(l_seg)))) * v
            end
        end::T
        return merge_remote_reductions(r.communicator, +, res)
    end
end
function VectorInterface.inner(fd::FrozenDVec, p::PDVec)
    res = zero(promote_type(valtype(fd), valtype(p)))
    for (k, v) in pairs(fd)
        res += conj(p[k]) * v
    end
    return merge_remote_reductions(p.communicator, +, res)
end
function VectorInterface.inner(l::AbstractDVec, r::PDVec)
    res = sum(pairs(r)) do (k, v)
        conj(l[k]) * v
    end
    return res
end
VectorInterface.inner(p::PDVec, fd::FrozenDVec) = conj(dot(fd, p))
VectorInterface.inner(l::PDVec, r::AbstractDVec) = conj(dot(r, l))

function Base.real(v::PDVec)
    dst = similar(v, real(valtype(v)))
    return map!(real, dst, values(v))
end
function Base.imag(v::PDVec)
    dst = similar(v, real(valtype(v)))
    return map!(imag, dst, values(v))
end

###
### Operator linear algebra operations
###
"""
    mul!(y::PDVec, A::AbstractHamiltonian, x::PDVec[, w::PDWorkingMemory])

Perform `y = A * x` in-place. The working memory `w` is required to facilitate
threaded/distributed operations. If not passed a new instance will be allocated. `y` and `x`
may be the same vector.
"""
function LinearAlgebra.mul!(
    y::PDVec, op::AbstractHamiltonian, x::PDVec,
    w=PDWorkingMemory(y; style=IsDeterministic()),
)
    if w.style ≢ IsDeterministic()
        throw(ArgumentError(
            "Attempted to use `mul!` with non-deterministic working memory. " *
            "Use `apply_operator!` instead."
        ))
    end
    _, _, wm, y = apply_operator!(w, y, x, op)
    return y
end

"""
    dot(y::PDVec, A::AbstractHamiltonian, x::PDVec[, w::PDWorkingMemory])

Perform `y ⋅ A ⋅ x`. The working memory `w` is required to facilitate threaded/distributed
operations with non-diagonal `A`. If needed and not passed a new instance will be
allocated. `A` can be replaced with a tuple of operators.
"""
function LinearAlgebra.dot(t::PDVec, op::AbstractHamiltonian, u::PDVec, w)
    return dot(LOStructure(op), t, op, u, w)
end
function LinearAlgebra.dot(t::PDVec, op::AbstractHamiltonian, u::PDVec)
    return dot(LOStructure(op), t, op, u)
end
function LinearAlgebra.dot(
    ::IsDiagonal, t::PDVec, op::AbstractHamiltonian, u::PDVec, w=nothing
)
    T = typeof(zero(valtype(t)) * zero(valtype(u)) * zero(eltype(op)))
    return sum(pairs(u); init=zero(T)) do (k, v)
        T(conj(t[k]) * diagonal_element(op, k) * v)
    end
end
function LinearAlgebra.dot(
    ::LOStructure, left::PDVec, op::AbstractHamiltonian, right::PDVec, w=nothing
)
    # First two cases: only one vector is distrubuted. Avoid shuffling things around
    # by placing that one on the left to reduce the need for communication.
    if !is_distributed(left) && is_distributed(right)
        return dot(AdjointUnknown(), left, op, right)
    elseif is_distributed(left) && !is_distributed(right)
        return conj(dot(AdjointUnknown(), right, op', left))
    end
    # Other cases: both vectors distributed or not distributed. Put the shorter vector
    # on the right as is done for regular DVecs.
    if length(left) < length(right)
        return conj(dot(AdjointUnknown(), right, op', left, w))
    else
        return dot(AdjointUnknown(), left, op, right, w)
    end
end
# Default variant: also called from other LOStructures.
function LinearAlgebra.dot(
    ::AdjointUnknown, t::PDVec, op::AbstractHamiltonian, source::PDVec, w=nothing
)
    if is_distributed(t)
        if isnothing(w)
            target = copy_to_local!(PDWorkingMemory(t), t)
        else
            target = copy_to_local!(w, t)
        end
    else
        target = t
    end
    return dot_from_right(target, op, source)
end

function dot_from_right(target, op, source::PDVec)
    T = typeof(zero(valtype(target)) * zero(valtype(source)) * zero(eltype(op)))

    result = sum(pairs(source); init=zero(T)) do (k, v)
        res = conj(target[k]) * diagonal_element(op, k) * v
        for (k_off, v_off) in offdiagonals(op, k)
            res += conj(target[k_off]) * v_off * v
        end
        res
    end
    return result::T
end

function LinearAlgebra.dot(t::PDVec, ops::Tuple, source::PDVec, w=nothing)
    if is_distributed(t) && any(LOStructure(op) ≢ IsDiagonal() for op in ops)
        if isnothing(w)
            target = copy_to_local!(PDWorkingMemory(t), t)
        else
            target = copy_to_local!(w, t)
        end
    else
        target = t
    end
    return map(ops) do op
        dot_from_right(target, op, source)
    end
end

"""
    FrozenPDVec

Parallel version of [`FrozenDVec`](@ref). See: [`freeze`](@ref), [`PDVec`](@ref).
"""
struct FrozenPDVec{K,V,N} <: AbstractProjector
    segments::NTuple{N,Vector{Pair{K,V}}}
end
Base.keytype(::FrozenPDVec{K}) where {K} = K
Base.valtype(::FrozenPDVec{<:Any,V}) where {V} = V
Base.eltype(::FrozenPDVec{K,V}) where {K,V} = Pair{K,V}
Base.pairs(fd::FrozenPDVec) = Iterators.flatten(fd.segments)

function freeze(dv::PDVec{K,V,N}) where {K,V,N}
    return FrozenPDVec{K,V,N}(map(collect, dv.segments))
end

function VectorInterface.inner(fd::FrozenPDVec, dv::AbstractDVec)
    T = promote_type(valtype(fd), valtype(dv))
    return Folds.sum(fd.segments) do segment
        sum(segment; init=zero(T)) do (k, v)
            dv[k] ⋅ v
        end
    end
end

function VectorInterface.inner(fd::FrozenPDVec, dv::PDVec)
    T = promote_type(valtype(fd), valtype(dv))
    res = Folds.sum(zip(fd.segments, dv.segments); init=zero(T)) do (l_seg, r_seg)
        sum(l_seg; init=zero(T)) do (k, v)
           T(conj(v) * get(r_seg, k, zero(valtype(r_seg))))
        end
    end::T
    return merge_remote_reductions(dv.communicator, +, res)
end
