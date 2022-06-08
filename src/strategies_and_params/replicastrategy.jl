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

# TODO: add custom names
"""
    AllOverlaps(num_replicas=2; operator=nothing, transform=nothing, vecnorm=true) <: ReplicaStrategy{num_replicas}

Run `num_replicas` replicas and report overlaps between all pairs of replica vectors. If operator is
not `nothing`, the overlap `dot(c1, operator, c2)` is reported as well. If operator is a tuple
of operators, the overlaps are computed for all operators.

Column names in the report are of the form `c{i}_dot_c{j}` for vector-vector overlaps, and
`c{i}_Op{k}_c{j}` for operator overlaps.

See [`lomc!`](@ref), [`ReplicaStrategy`](@ref) and [`AbstractHamiltonian`](@ref) (for an
interface for implementing operators).

If `transform` is not `nothing` then this strategy calculates the overlaps with respect to a 
similarity transformation `G` of the Hamiltonian (see e.g. [`GutzwillerSampling`](@ref).)
```math
    \\hat{G} = f \\hat{H} f^{-1}.
```
The expectation value of an operator `A` is then
```math
    \\langle A \\rangle = \\langle \\psi | A | \\psi \\rangle 
        = \\frac{\\langle \\phi | f^{-1} A f^{-1} | \\phi \\rangle}{\\langle \\phi | f^{-2} | \\phi \\rangle}
```
where 
```math
    | \\phi \\rangle = f | \\psi \\rangle
``` 
is the (right) eigenvector of ``\\hat{G}``.

For a k-tuple of input operators `(A_1,..., A_k)`, overlaps of ``f^{-1} A_k f^{-1}`` are reported 
as `c{i}_Op{k}_c{j}`. Overlaps of ``f^{-2}`` are reported *last* as `c{i}_Op{k+1}_c{j}`.

In either case, the untransformed vector-vector overlap `c{i}_dot_c{j}` 
can be omitted with the flag `vecnorm=false`.
"""
struct AllOverlaps{N,M,O<:NTuple{M,AbstractHamiltonian},B} <: ReplicaStrategy{N}
    operators::O
end

function AllOverlaps(num_replicas=2; operator=nothing, transform=nothing, vecnorm=true)
    if isnothing(operator)
        operators = ()
    elseif operator isa Tuple
        operators = operator
    else
        operators = (operator,)
    end
    if isnothing(transform)
        ops = operators
    else
        fsq = Rimu.Hamiltonians.TransformUndoer(transform)
        ops = (map(op -> Rimu.Hamiltonians.TransformUndoer(transform, op), operators)..., fsq)
    end    
    if !vecnorm && length(ops) == 0
        return NoStats(num_replicas)
    end
    return AllOverlaps{num_replicas,length(ops),typeof(ops),vecnorm}(ops)
end
@deprecate AllOverlaps(num_replicas, operator) AllOverlaps(num_replicas; operator)

function replica_stats(rs::AllOverlaps{N,<:Any,<:Any,B}, replicas::NTuple{N}) where {N,B}
    # Not using broadcasting because it wasn't inferred properly.
    vecs = ntuple(i -> replicas[i].v, Val(N))
    return all_overlaps(rs.operators, vecs, B)
end

"""
    all_overlaps(operators, vectors, vecnorm=true)

Get all overlaps between vectors and operators. This function is overloaded for `MPIData`.
The flag `vecnorm` can disable the vector-vector overlap `c{i}_dot_c{j}`.
"""
function all_overlaps(operators::Tuple, vecs::NTuple{N,AbstractDVec}, vecnorm=true) where {N}
    T = promote_type((valtype(v) for v in vecs)..., eltype.(operators)...)
    names = String[]
    values = T[]
    for i in 1:N, j in i+1:N
        if vecnorm
            push!(names, "c$(i)_dot_c$(j)")
            push!(values, dot(vecs[i], vecs[j]))
        end
        for (k, op) in enumerate(operators)
            push!(names, "c$(i)_Op$(k)_c$(j)")
            push!(values, dot(vecs[i], op, vecs[j]))
        end
    end
    
    reports_per_replica = vecnorm ? length(operators) + 1 : length(operators)
    num_reports = (N * (N - 1) ÷ 2) * reports_per_replica 
    return SVector{num_reports,String}(names).data, SVector{num_reports,T}(values).data
end