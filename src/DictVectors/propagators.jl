"""
    abstract type AbstractPropagator{T}

An operator equipped with an instance of [`PDWorkingMemory`](@ref). The main operation on
it is [`propagate!`](@ref), which performs a (stochastic) analog of operator application
and returns spawning statistics.

An `AbstractPropagator` must define either [`spawn_column!`](@ref) and
[`working_memory`](@ref), or [`propagate!`](@ref).

See [`FCIQMCPropagator`](@ref) and [`OperatorMulPropagator`](@ref).
"""
abstract type AbstractPropagator{T} end
Base.eltype(::Type{<:AbstractPropagator{T}}) where {T} = T

"""
    spawn_column!(target, propagator, key, value)

Spawn a full of a column at index `key` with value in vector `value` and store them to
`target` dictionary. Returns a tuple of statistics that are recorded when running Rimu.
"""
spawn_column!

"""
    propagate!(target, propagator, source)

Perform a single application of a (possibly stochastic) `propagator`.
"""
function propagate!(dst::PDVec, prop::AbstractPropagator, src)
    w = working_memory(prop)
    stats = perform_spawns!(w, src, prop)
    collect_local!(w)
    synchronize_remote!(w)
    move_and_compress!(dst, w)
    return stats
end

function LinearAlgebra.mul!(dst, prop::AbstractPropagator, src)
    propagate!(dst, prop, src)
    return dst
end

function Base.:*(prop::AbstractPropagator, v::PDVec)
    w = similar(v, promote_type(eltype(prop), valtype(v)))
    return mul!(w, prop, v)
end
(prop::AbstractPropagator)(v) = prop * v

"""
    FCIQMCPropagator(hamiltonian, shift, dτ, working_memory)

[`AbstractPropagator`](@ref) that performs

```math
w = v + dτ (S - H) v
```

where ``S`` is the `shift`, ``H`` the `hamiltonian`, and ``w`` and ``v`` the vectors.
"""
struct FCIQMCPropagator{H,S,T,W<:PDWorkingMemory{<:Any,T}} <: AbstractPropagator{T}
    hamiltonian::H
    shift::S
    dτ::Float64
    working_memory::W
end

working_memory(f::FCIQMCPropagator) = f.working_memory

function spawn_column!(column, f::FCIQMCPropagator, k, v)
    return perform_spawns!(
        f.working_memory.style, column, f.hamiltonian, k, v, f.shift, f.dτ
    )
end

"""
    OperatorMulPropagator(operator, ::PDVec) <: AbstractPropagator
    OperatorMulPropagator(operator, ::PDWorkingMemory) <: AbstractPropagator

[`AbstractPropagator`](@ref) that performs matrix-vector multiplications.

If a vector is passed to the constructor, a working memory with a deterministic stochastic
style is created.
"""
struct OperatorMulPropagator{O,T,W<:PDWorkingMemory{<:Any,T}} <: AbstractPropagator{T}
    operator::O
    working_memory::W
end
function OperatorMulPropagator(operator, t::PDVec)
    if eltype(operator) === valtype(t)
        wm = PDWorkingMemory(t; style=IsDeterministic{eltype(operator)}())
    else
        T = promote_type(eltype(operator), valtype(t))
        wm = PDWorkingMemory(similar(t, T); style=IsDeterministic{T}())
    end
    return OperatorMulPropagator(operator, wm)
end

working_memory(o::OperatorMulPropagator) = o.working_memory

function spawn_column!(column, o::OperatorMulPropagator, k, v)
    T = eltype(o.operator)
    return perform_spawns!(
        o.working_memory.style, column, o.operator, k, v, one(T), -one(T)
    )
end
