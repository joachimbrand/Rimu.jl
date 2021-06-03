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

struct ProjectedEnergy{P,Q,H} <: PostStepStrategy
    vproj::P
    hproj::Q
    ham::H
end

function ProjectedEnergy(projector, hamiltonian::AbstractHamiltonian)
    return ProjectedEnergy(Hamiltonians.LOStructure(hamiltonian), projector, hamiltonian)
end
function ProjectedEnergy(::Hamiltonians.AdjointUnknown, projector, hamiltonian)
    @warn "typeof(hamiltonian) has unknown adjoint. This will negatively affect performance."
    return ProjectedEnergy(freeze(projector), nothing, ham)
end
function ProjectedEnergy(::Hamiltonians.LOStructure, projector, ham)
    vproj = freeze(projector)
    hproj = freeze(ham' * projector)
    return ProjectedEnergy(vproj, hproj, ham)
end

# Method with non-adjointable hamiltonian
function post_step(p::ProjectedEnergy{<:Any,Nothing}, replica)
    return (
        :vproj => dot(p.vproj, replica.v),
        :hproj => dot(p.vproj, p.ham, replica.v),
    )
end
function post_step(p::ProjectedEnergy, replica)
    return (
        :vproj => dot(p.vproj, replica.v),
        :hproj => conj(dot(p.hproj, replica.v)),
    )
end
