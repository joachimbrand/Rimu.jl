"""
    InitiatorDVec{K,V} <: AbstractDVec{K,V}

Dictionary-based vector-like data structure for use with
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem) and
[`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl). See [`AbstractDVec`](@ref).
Functionally identical to [`DVec`](@ref), but contains [`InitiatorValue`](@ref)s internally
in order to facilitate initiator methods. Initiator methods for controlling the Monte Carlo
sign problem were first introduced in
[J. Chem. Phys. 132, 041103 (2010)](https://doi.org/10.1063/1.3302277).
How the initiators are handled is controlled by specifying an [`InitiatorRule`](@ref) with
the `initiator` keyword argument (see below).

See also: [`AbstractDVec`](@ref), [`DVec`](@ref), [`PDVec`](@ref).

## Constructors

* `InitiatorDVec(dict::AbstractDict[; style, initiator, capacity])`: create an
  `InitiatorDVec` with `dict` for storage.  Note that the data may or may not be copied.

* `InitiatorDVec(args...[; style, initiator, capacity])`: `args...` are passed to the `Dict`
  constructor. The `Dict` is used for storage.

* `InitiatorDVec{K,V}([; style, initiator, capacity])`: create an empty `InitiatorDVec{K,V}`.

* `InitiatorDVec(dv::AbstractDVec[; style, initiator, capacity])`: create an `InitiatorDVec`
   with the same contents as `dv`. The `style` is inherited from `dv` by default.

## Keyword  arguments

* `style`: A valid [`StochasticStyle`](@ref).  The default is selected based on the
  `InitiatorDVec`'s `valtype` (see [`default_style`](@ref)). If a style is given and
  the `valtype` does not match the `style`'s `eltype`, the values are converted to an
  appropriate type.

* `initiator = Initiator(1)`: A valid [`InitiatorRule`](@ref). See [`Initiator`](@ref).

* `capacity`: Indicative size as `Int`. Optional. Sets the initial size of the
  `InitiatorDVec` via `Base.sizehint!`.

"""
struct InitiatorDVec{
    K,V,W<:AbstractInitiatorValue{V},D<:AbstractDict{K,W},
    S<:StochasticStyle{V},I<:InitiatorRule
} <: AbstractDVec{K,V}
    storage::D
    style::S
    initiator::I
end

###
### Constructors
###
# Vararg
function InitiatorDVec(args::Vararg{Pair}; kwargs...)
    return InitiatorDVec(args; kwargs...)
end
# Iterator
function InitiatorDVec(pairs; kwargs...)
    storage = Dict(pairs)
    return InitiatorDVec(storage; kwargs...)
end
# Dict with InitiatorValues
function InitiatorDVec(
    dict::AbstractDict{K,W};
    style=default_style(V),
    initiator_threshold=one(eltype(style)),
    initiator=Initiator(eltype(style)(initiator_threshold)),
    capacity=0,
) where {K,V,W<:AbstractInitiatorValue{V}}
    T = eltype(style)
    if T === V
        storage = dict
        sizehint!(storage, capacity)
    else
        storage = Dict{K,W}()
        sizehint!(storage, capacity)
        for (k, v) in pairs(dict)
            storage[k] = W(v)
        end
    end
    return InitiatorDVec(storage, style, initiator)
end
# Dict with regular values
function InitiatorDVec(
    dict::AbstractDict{K,V};
    style=default_style(V),
    initiator_threshold=one(eltype(style)),
    initiator=Initiator(eltype(style)(initiator_threshold)),
    capacity=0,
) where {K,V}
    T = eltype(style)
    W = initiator_valtype(initiator, T)
    storage = Dict{K,W}()
    sizehint!(storage, capacity)
    for (k, v) in pairs(dict)
        storage[k] = W(v)
    end
    return InitiatorDVec(storage, style, initiator)
end
# Empty
function InitiatorDVec{K,V}(
    ;
    style=default_style(V),
    initiator_threshold=one(V),
    initiator=Initiator(V(initiator_threshold)),
    kwargs...
) where {K,V}
    W = initiator_valtype(initiator, eltype(style))
    return InitiatorDVec(Dict{K,W}(); style, initiator, kwargs...)
end
# From another DVec
function InitiatorDVec(
    dv::AbstractDVec{K,V};
    style=StochasticStyle(dv),
    initiator_threshold=one(eltype(style)),
    initiator=Initiator(eltype(style)(initiator_threshold)),
    capacity=0,
) where {K,V}
    return InitiatorDVec(copy(storage(dv)); style, initiator, capacity)
end

function Base.empty(dvec::InitiatorDVec{K,V}; style=dvec.style) where {K,V}
    return InitiatorDVec{K,V}(; style, initiator=dvec.initiator)
end
function Base.empty(dvec::InitiatorDVec{K,V}, ::Type{V}; style=dvec.style) where {K,V}
    return empty(dvec; style)
end
function Base.empty(dvec::InitiatorDVec{K}, ::Type{V}; style=default_style(V)) where {K,V}
    return InitiatorDVec{K,V}(; style, initiator=dvec.initiator)
end
function Base.empty(dvec::InitiatorDVec, ::Type{K}, ::Type{V}; style=default_style(V)) where {K,V}
    return InitiatorDVec{K,V}(; style, initiator=dvec.initiator)
end

###
### Show
###
function Base.summary(io::IO, dvec::InitiatorDVec{K,V}) where {K,V}
    len = length(dvec)
    entries = length(dvec) == 1 ? "entry" : "entries"
    print(io, "InitiatorDVec{$K,$V} with $len $entries, style = $(dvec.style), initiator = $(dvec.initiator)")
end

###
### Interface
###
StochasticStyle(dv::InitiatorDVec) = dv.style
storage(dv::InitiatorDVec) = dv.storage

function Base.getindex(dvec::InitiatorDVec{<:Any,V,W}, add) where {V,W}
    return from_initiator_value(
        dvec.initiator, get(dvec.storage, add, zero(W))
    )
end
function Base.setindex!(dvec::InitiatorDVec{<:Any,<:Any,W}, v, k) where {W}
    if iszero(v)
        delete!(dvec.storage, k)
    else
        dvec.storage[k] = W(v)
    end
    return v
end

function Base.get(dvec::InitiatorDVec, args...)
    return from_initiator_value(dvec.initiator, get(dvec.storage, args...))
end
function Base.get!(dvec::InitiatorDVec{<:Any,<:Any,W}, key, default) where {W}
    return from_initiator_value(
        dvec.initiator, get!(dvec.storage, key, W(default))
    )
end
function Base.get!(f::Function, dvec::InitiatorDVec{<:Any,<:Any,W}, key) where {W}
    return from_initiator_value(
        dvec.initiator, get!(dvec.storage, key, W(f()))
    )
end

@delegate InitiatorDVec.storage [haskey, getkey, pop!, isempty, length, keys]
@delegate_return_parent InitiatorDVec.storage [delete!, empty!, sizehint!]

"""
    deposit!(w::InitiatorDVec, add, val, p_add=>p_val)
Add `val` into `w` at address `add` as an [`AbstractInitiatorValue`](@ref).
"""
function deposit!(w::InitiatorDVec{<:Any,V,W}, add, val, (p_add, p_val)) where {V,W}
    i = w.initiator
    old_val = get(w.storage, add, zero(W))
    new_val = to_initiator_value(w.initiator, add, V(val), (p_add, p_val)) + old_val
    if iszero(new_val)
        delete!(w.storage, add)
    else
        w.storage[add] = new_val
    end
    return w
end

function deposit!(w::InitiatorDVec{<:Any,<:Any,W}, add, val::W, _) where {W}
    dict = storage(w)
    old_val = get(dict, add, zero(W))
    dict[add] = old_val + val
    return w
end

###
### Iterators
###
# These are needed because `Iterators.map` does not infer `eltype` correctly and does not work
# with SplittablesBase.jl.
"""
    InitiatorIterator

Iterator over pairs or values of an `InitiatorDVec`.
"""
struct InitiatorIterator{T,D,I}
    iter::D
    initiator::I

    InitiatorIterator{T}(iter::D, initiator::I) where {T,D,I} = new{T,D,I}(iter, initiator)
end

Base.length(p::InitiatorIterator) = length(p.iter)
Base.IteratorSize(::InitiatorIterator) = Base.HasLength()
Base.IteratorEltype(::InitiatorIterator) = Base.HasEltype()
Base.eltype(::InitiatorIterator{T}) where {T} = T

function Base.pairs(dvec::InitiatorDVec{K,V}) where {K,V}
    InitiatorIterator{Pair{K,V}}(pairs(storage(dvec)), dvec.initiator)
end
function Base.iterate(p::InitiatorIterator{<:Pair}, args...)
    it = iterate(p.iter, args...)
    isnothing(it) && return nothing
    (k, v), st = it
    return k => from_initiator_value(p.initiator, v), st
end

function Base.values(dvec::InitiatorDVec{<:Any,V}) where {V}
    InitiatorIterator{V}(values(storage(dvec)), dvec.initiator)
end
function Base.iterate(p::InitiatorIterator, args...)
    it = iterate(p.iter, args...)
    isnothing(it) && return nothing
    v, st = it
    return from_initiator_value(p.initiator, v), st
end
