"""
    PostStepStrategy

Subtypes of `PostStepStrategy` can be used to perform arbitrary computation on a replica
after a FCIQMC step is finished and report the results.

A subtype of this type must implement [`post_step(::PostStepStrategy, ::ReplicaState)`](@ref).
"""
abstract type PostStepStrategy end

"""
    post_step(::PostStepStrategy, ::ReplicaState) -> kvpairs

Should return a tuple of `:key => value` pairs. See also [`PostStepStrategy`](@ref).
"""
post_step

# When startegies are a Tuple, apply all of them.
function post_step(t::Tuple{}, replica)
    return ()
end
function post_step((t,ts...)::Tuple, replica)
    head = post_step(t, replica)
    rest = post_step(ts, replica)
    return (head..., rest...)
end

"""
    Projector(name=projector) <: PostStepStrategy

After each step, compute `dot(projector, v)` and report it in the `DataFrame` under `name`.
"""
struct Projector{P} <: PostStepStrategy
    name::Symbol
    projector::P
end
Projector(;kwarg...) = Projector(only(keys(kwarg)), only(values(kwarg)))

function post_step(p::Projector, replica)
    return (p.name => dot(p.projector, replica.v))
end

"""
    SignCoherence(reference) <: PostStepStrategy

After each step, compute the proportion of configurations that have the same sign as they do
in the `reference_dvec`.
"""
struct SignCoherence{R} <: PostStepStrategy
    reference::R
end

function post_step(sc::SignCoherence, replica)
    vector = replica.v
    num_correct = mapreduce(+, vector; init=zero(valtype(vector))) do ((k, v))
        sign(v) == sign(sc.reference[k])
    end
    coherent_configs = num_correct / length(vector)
    return (:coherent_configs => coherent_configs,)
end
