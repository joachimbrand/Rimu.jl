"""
    PostStepStrategy

Subtypes of `PostStepStrategy` can be used to perform arbitrary computation on a replica
after a FCIQMC step is finished and report the results.

Note: a tuple of multiple strategies can be passed to [`lomc!`](@ref). In that case, all
reported column names must be distinct.

A subtype of this type must implement [`post_step(::PostStepStrategy, ::ReplicaState)`](@ref).
"""
abstract type PostStepStrategy end

"""
    post_step(::PostStepStrategy, ::ReplicaState) -> kvpairs

Compute statistics after FCIQMC step. Should return a tuple of `:key => value` pairs. See
also [`PostStepStrategy`](@ref).
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

After each step, compute `dot(projector, dv)` and report it in the `DataFrame` under `name`.
`projector` can be an [`AbstractDVec`](@ref), or an [`AbstractProjector`](@ref).
"""
struct Projector{P} <: PostStepStrategy
    name::Symbol
    projector::P
end
function Projector(;kwarg...)
    length(kwarg) ≠ 1 && error("exactly one keyword argument must be passed to `Projector`")
    return Projector(only(keys(kwarg)), freeze(only(values(kwarg))))
end

function post_step(p::Projector, replica)
    return (p.name => dot(p.projector, replica.v),)
end

"""
    ProjectedEnergy(hamiltonian, projector; hproj=:vproj, vproj=:vproj) <: PostStepStrategy

After every step, compute `hproj = dot(projector, hamiltonian, dv)` and `vproj =
dot(projector, dv)`, where `dv` is the instantaneous coefficient vector.  `projector` can be
an [`AbstractDVec`](@ref), or an [`AbstractProjector`](@ref).

Reports to columns `hproj` and `vproj`, which can be used to compute projective energy,
e.g. with [`Rimu.StatsTools.ratio_of_means`](@ref). The keyword arguments `hproj` and
`vproj` can be used to change the names of these columns. This can be used to make the names
unique when computing projected energies with different projectors in the same run.
"""
struct ProjectedEnergy{H,P,Q} <: PostStepStrategy
    vproj_name::Symbol
    hproj_name::Symbol
    ham::H
    vproj::P
    hproj::Q
end

function ProjectedEnergy(
    hamiltonian::AbstractHamiltonian, projector;
    vproj=:vproj, hproj=:hproj
)
    hproj_vec = compute_hproj(Hamiltonians.LOStructure(hamiltonian), hamiltonian, projector)
    return ProjectedEnergy(vproj, hproj, hamiltonian, freeze(projector), hproj_vec)
end
function compute_hproj(::Hamiltonians.AdjointUnknown, hamiltonian, projector)
    @warn "$(typeof(hamiltonian)) has an unknown adjoint. This will be slow."
    return nothing
end
function compute_hproj(::Hamiltonians.LOStructure, ham, projector)
    return freeze(ham' * projector)
end

function post_step(p::ProjectedEnergy{<:Any,<:Any,Nothing}, replica)
    return (
        p.vproj_name => dot(p.vproj, replica.v),
        p.hproj_name => dot(p.vproj, p.ham, replica.v),
    )
end
function post_step(p::ProjectedEnergy, replica)
    return (
        p.vproj_name => dot(p.vproj, replica.v),
        p.hproj_name => conj(dot(p.hproj, replica.v)),
    )
end

"""
    SignCoherence(reference) <: PostStepStrategy

After each step, compute the proportion of configurations that have the same sign as they do
in the `reference_dvec`. Reports to a column named `coherence`.
"""
struct SignCoherence{R} <: PostStepStrategy
    reference::R
end

function post_step(sc::SignCoherence, replica)
    vector = replica.v
    num_correct = mapreduce(+, pairs(vector); init=zero(valtype(vector))) do ((k, v))
        sign(v) == sign(sc.reference[k])
    end
    return (:coherence => num_correct / length(vector),)
end
