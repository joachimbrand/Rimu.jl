"""
    ReplicaStrategy{N}

Supertype for strategies that can be passed to [`lomc!`](@ref) and control how many
replicas are used, and what information is computed and returned. The number of replicas
is `N`.

## Concrete implementations

* [`NoStats`](@ref): run (possibly one) replica(s), but don't report any additional info.
* [`AllOverlaps`](@ref): report overlaps between all pairs of replica vectors.

## Interface

A subtype of `ReplicaStrategy{N}` must implement the following
function:

* [`Rimu.replica_stats`](@ref) - return a
  tuple of `String`s or `Symbols` of names for replica statistics and a tuple of the values.
  These will be reported to the `DataFrame` returned by [`lomc!`](@ref).
"""
abstract type ReplicaStrategy{N} end

num_replicas(::ReplicaStrategy{N}) where {N} = N

"""
    replica_stats(RS::ReplicaStrategy{N}, replicas::NTuple{N,ReplicaState}) -> (names, values)

Return the names and values of statistics related to `N` replicas consistent with the
[`ReplicaStrategy`](@ref) `RS`. `names`
should be a tuple of `Symbol`s or `String`s and `values` should be a tuple of the same
length. This fuction will be called every [`reporting_interval`](@ref) steps from [`lomc!`](@ref), 
or once per time step if `reporting_interval` is not defined.

Part of the [`ReplicaStrategy`](@ref) interface. See also [`ReplicaState`](@ref).
"""
replica_stats

"""
    NoStats(N=1) <: ReplicaStrategy{N}

The default [`ReplicaStrategy`](@ref). `N` replicas are run, but no statistics are
collected.

See also [`lomc!`](@ref).
"""
struct NoStats{N} <: ReplicaStrategy{N} end
NoStats(N=1) = NoStats{N}()

replica_stats(::NoStats, _) = (), ()

"""
    AllOverlaps(n=2, operator=nothing, fsquared=nothing) <: ReplicaStrategy{n}

Run `n` replicas and report overlaps between all pairs of replica vectors. If operator is
not `nothing`, the overlap `dot(c1, operator, c2)` is reported as well. If operator is a tuple
of operators, the overlaps are computed for all operators.

Column names in the report are of the form c{i}_dot_c{j} for vector-vector overlaps, and
c{i}_Op{k}_c{j} for operator overlaps.

See [`lomc!`](@ref), [`ReplicaStrategy`](@ref) and [`AbstractHamiltonian`](@ref) (for an
interface for implementing operators).

If `fsquared` is not `nothing` then this strategy calculates the overlaps with respect to a 
similarity transformation of the Hamiltonian
```math
    G = f^{-1} H f
```
The expectation value of an operator `A` is then
```math
    \\langle A \\rangle = \\langle \\psi | A | \\psi \\rangle 
        = \\frac{\\langle \\phi | f A f | \\phi \\rangle}{\\frac{\\langle \\phi | f^2 | \\phi \\rangle}
```
where `| \\phi \\rangle = f | \\psi \\rangle` is the (right) eigenvector 
of `G`.

The `fsquared` argument must be the operator `f^2` and the `operator`s must
be the transformed operators `f A f`.

See e.g. [`GutzwillerSampling`](@ref).
"""
struct AllOverlaps{N,M,O<:NTuple{M,AbstractHamiltonian},S<:Union{AbstractHamiltonian,Nothing}} <: ReplicaStrategy{N}
    operators::O
    fsquared::S
end

function AllOverlaps(num_replicas=2, operator=nothing, fsquared=nothing)
    if isnothing(operator)
        operators = ()
    elseif operator isa Tuple
        operators = operator
    else
        operators = (operator,)
    end
    if !isnothing(fsquared)
        !(typeof(fsquared)<:SimTransOperator) && throw(ArgumentError("fsquared must be transformed"))
        for op in operators
            !(typeof(op)<:SimTransOperator) && throw(ArgumentError("Operators must be transformed"))
        end
    end
    return AllOverlaps{num_replicas,length(operators),typeof(operators),typeof(fsquared)}(operators,fsquared)
end

function replica_stats(rs::AllOverlaps, replicas::NTuple{N}) where {N}
    # Not using broadcasting because it wasn't inferred properly.
    vecs = ntuple(i -> replicas[i].v, Val(N))
    if isnothing(rs.fsquared)
        return all_overlaps(rs.operators, vecs)
    else
        return all_overlaps(rs.operators, rs.fsquared, vecs)
    end
end

"""
    all_overlaps(operators, vectors)

Get all overlaps between vectors and operators. This function is overloaded for `MPIData`.
"""
function all_overlaps(operators::Tuple, vecs::NTuple{N,AbstractDVec}) where {N}
    T = promote_type((valtype(v) for v in vecs)..., eltype.(operators)...)
    names = String[]
    values = T[]
    for i in 1:N, j in i+1:N
        push!(names, "c$(i)_dot_c$(j)")
        push!(values, dot(vecs[i], vecs[j]))
        for (k, op) in enumerate(operators)
            push!(names, "c$(i)_Op$(k)_c$(j)")
            push!(values, dot(vecs[i], op, vecs[j]))
        end
    end

    num_reports = (N * (N - 1) ÷ 2) * (length(operators) + 1)
    return SVector{num_reports,String}(names).data, SVector{num_reports,T}(values).data
end

"""
    all_overlaps(operators, fsquared, vectors)

Get all overlaps between vectors and operators, transformed by similarity operator `f`. 
This function is overloaded for `MPIData`.
"""
function all_overlaps(operators::Tuple, fsquared::SimTransOperator, vecs::NTuple{N,AbstractDVec}) where {N}
    T = promote_type((valtype(v) for v in vecs)..., eltype.(operators)..., eltype(fsquared))
    names = String[]
    values = T[]
    for i in 1:N, j in i+1:N
        push!(names, "c$(i)_dot_c$(j)")
        push!(values, dot(vecs[i], vecs[j]))
        push!(names, "c$(i)_f^2_c$(j)")
        push!(values, dot(vecs[i], fsquared, vecs[j]))
        for (k, op) in enumerate(operators)
            push!(names, "c$(i)_f.Op$(k).f_c$(j)")
            push!(values, dot(vecs[i], op, vecs[j]))
        end
    end

    num_reports = (N * (N - 1) ÷ 2) * (length(operators) + 2)
    return SVector{num_reports,String}(names).data, SVector{num_reports,T}(values).data
end