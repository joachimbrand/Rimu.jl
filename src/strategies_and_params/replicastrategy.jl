"""
    ReplicaStrategy{N}

An abstract type that controles how [`lomc!`](@ref) uses replicas. A subtype of
`ReplicaStrategy{N}` operates on `N` replicas and must implement the following function:

* [`replica_stats(::ReplicaStrategy{N}, ::NTuple{N,ReplicaState})`](@ref) - return a tuple
  of `String`s or `Symbols` of replica statistic names and a tuple of the values.  These
  will be reported to the `DataFrame` returned by [`lomc!`](@ref)

Concrete implementations:

* [`NoStats`](@ref): run (possibly one) replica(s), but don't report any additional info.
* [`AllOverlaps`](@ref): report overlaps between all pairs of replica vectors.

"""
abstract type ReplicaStrategy{N} end

num_replicas(::ReplicaStrategy{N}) where {N} = N

"""
    replica_stats(::ReplicaStrategy{N}, replicas::NTuple{N,ReplicaState}) -> (names, values)

Return the names and values of statistics reported by `ReplicaStrategy`. `names` should be
a tuple of `Symbol`s or `String`s and `values` should be a tuple of the same length.
"""
replica_stats

"""
    NoStats(N=1) <: ReplicaStrategy{N}

The default [`ReplicaStrategy`](@ref). `N` replicas are run, but no statistics are collected.
"""
struct NoStats{N} <: ReplicaStrategy{N} end
NoStats(N=1) = NoStats{N}()

replica_stats(::NoStats, _) = (), ()

"""
    AllOverlaps(n=2, operator=nothing) <: ReplicaStrategy{n}

Run `n` replicas and report overlaps between all pairs of replica vectors. If operator is
not `nothing`, the overlap `dot(c1, operator, c2)` is reported as well. If operator is a tuple
of operators, the overlaps are computed for all operators.

Column names in the report are of the form c{i}_dot_c{j} for vector-vector overlaps, and
c{i}_Op{k}_c{j} for operator overlaps.

See [`ReplicaStrategy`](@ref) and [`AbstractHamiltonian`](@ref) (for an interface for
implementing operators).
"""
struct AllOverlaps{N,O} <: ReplicaStrategy{N}
    operators::O
end

function AllOverlaps(num_replicas=2, operator=nothing)
    if isnothing(operator)
        operators = ()
    elseif operator isa Tuple
        operators = operator
    else
        operators = (operator,)
    end
    return AllOverlaps{num_replicas, typeof(operators)}(operators)
end

function replica_stats(rs::AllOverlaps{N}, replicas) where {N}
    T = promote_type((valtype(r.v) for r in replicas)..., eltype.(rs.operators)...)
    names = String[]
    values = T[]
    for i in 1:N, j in i+1:N
        push!(names, "c$(i)_dot_c$(j)")
        push!(values, dot(localpart(replicas[i].v), localpart(replicas[j].v)))
        for (k, op) in enumerate(rs.operators)
            push!(names, "c$(i)_Op$(k)_c$(j)")
            push!(values, dot(replicas[i].v, op, replicas[j].v))
        end
    end

    num_reports = (N * (N - 1) ÷ 2) * (length(rs.operators) + 1)
    return SVector{num_reports,String}(names).data, SVector{num_reports,T}(values).data
end
