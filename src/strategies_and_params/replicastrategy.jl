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
    AllOverlaps(n=2, operator=nothing, transform=nothing, vecnorm=true) <: ReplicaStrategy{n}

Run `n` replicas and report overlaps between all pairs of replica vectors. If operator is
not `nothing`, the overlap `dot(c1, operator, c2)` is reported as well. If operator is a tuple
of operators, the overlaps are computed for all operators.

Column names in the report are of the form c{i}_dot_c{j} for vector-vector overlaps, and
c{i}_Op{k}_c{j} for operator overlaps.

See [`lomc!`](@ref), [`ReplicaStrategy`](@ref) and [`AbstractHamiltonian`](@ref) (for an
interface for implementing operators).

If `transform` is not `nothing` then this strategy calculates the overlaps with respect to a 
similarity transformation of the Hamiltonian (See e.g. [`GutzwillerSampling`](@ref).)
```math
    G = f^{-1} H f.
```
The expectation value of an operator `A` is then
```math
    \\langle A \\rangle = \\langle \\psi | A | \\psi \\rangle 
        = \\frac{\\langle \\phi | f A f | \\phi \\rangle}{\\frac{\\langle \\phi | f^2 | \\phi \\rangle}
```
where `| \\phi \\rangle = f^2 | \\psi \\rangle` is the (right) eigenvector of `G`.

In this case, overlaps of `f^2` are reported first as c{i}_Op1_c{j}, and then all 
transformed operators are reported. That is, for a tuple of input operators `(A, B,...)`, 
overlaps of `(f A f, f B f,...)` are reported as `(c{i}_Op2_c{j}, c{i}_Op3_c{j}, ...)`.

In either case, the untransformed vector-vector overlap `c{i}_dot_c{j}` 
can be omitted with the flag `vecnorm=false`.
"""
struct AllOverlaps{N,M,O<:NTuple{M,AbstractHamiltonian},B} <: ReplicaStrategy{N}
    operators::O
end

function AllOverlaps(num_replicas=2, operator=nothing, transform=nothing, vecnorm=true)
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
        !hasproperty(transform, :hamiltonian) && throw(ArgumentError("Invalid Hamiltonian transformation."))
        f2 = SimTransOperator(transform)
        ops = (f2, map(op -> SimTransOperator(transform, op), operators)...)
    end    
    if !vecnorm && length(ops) == 0
        return NoStats(num_replicas)
    end
    return AllOverlaps{num_replicas,length(ops),typeof(ops),vecnorm}(ops)
end

function replica_stats(rs::AllOverlaps{N,<:Any,<:Any,B}, replicas::NTuple{N}) where {N,B}
    # Not using broadcasting because it wasn't inferred properly.
    vecs = ntuple(i -> replicas[i].v, Val(N))
    return all_overlaps(rs.operators, vecs, B)
end

"""
    all_overlaps(operators, vectors, norm=true)

Get all overlaps between vectors and operators. This function is overloaded for `MPIData`.
The flag `vecnorm` can disable the vector-vector overlap if a transformed Hamiltonian is supplied
to the `ReplicaStrategy`.
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