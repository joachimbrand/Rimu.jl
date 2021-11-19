"""
    PostStepStrategy

Subtypes of `PostStepStrategy` can be used to perform arbitrary computation on a replica
after an FCIQMC step is finished and report the results.

# Implemented strategies:

* [`ProjectedEnergy`](@ref)
* [`Projector`](@ref)
* [`SignCoherence`](@ref)
* [`WalkerLoneliness`](@ref)
* [`Timer`](@ref)

Note: A tuple of multiple strategies can be passed to [`lomc!`](@ref). In that case, all
reported column names must be distinct.

# Interface:

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
    length(kwarg) ≠ 1 && throw(
        ArgumentError("exactly one keyword argument must be passed to `Projector`")
    )
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
e.g. with [`Rimu.StatsTools.projected_energy`](@ref). The keyword arguments `hproj` and
`vproj` can be used to change the names of these columns. This can be used to make the names
unique when computing projected energies with different projectors in the same run.

See also [`Rimu.StatsTools.ratio_of_means`](@ref),
[`Rimu.StatsTools.mixed_estimator`](@ref).
"""
struct ProjectedEnergy{H,P,Q} <: PostStepStrategy
    vproj_name::Symbol
    hproj_name::Symbol
    ham::H
    vproj::P
    hproj::Q
end

function ProjectedEnergy(
    hamiltonian, projector;
    vproj=:vproj, hproj=:hproj
)
    hproj_vec = compute_hproj(LOStructure(hamiltonian), hamiltonian, projector)
    return ProjectedEnergy(vproj, hproj, hamiltonian, freeze(projector), hproj_vec)
end
function compute_hproj(::AdjointUnknown, hamiltonian, projector)
    @warn "$(typeof(hamiltonian)) has an unknown adjoint. This will be slow."
    return nothing
end
function compute_hproj(::LOStructure, ham, projector)
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
        p.hproj_name => dot(p.hproj, replica.v),
    )
end

"""
    SignCoherence(reference[; name=:coherence]) <: PostStepStrategy

After each step, compute the proportion of configurations that have the same sign as they do
in the `reference_dvec`. Reports to a column named `name`, which defaults to `coherence`.
"""
struct SignCoherence{R} <: PostStepStrategy
    name::Symbol
    reference::R
end
SignCoherence(ref; name=:coherence) = SignCoherence(name, ref)

function post_step(sc::SignCoherence, replica)
    vector = replica.v
    return (sc.name => coherence(valtype(vector), sc.reference, vector),)
end

function coherence(::Type{<:Real}, reference, vector)
    accumulator, overlap = mapreduce(+, pairs(vector); init=MultiScalar(0.0, 0)) do ((k, v))
        ref = reference[k]
        MultiScalar(Float64(sign(ref) * sign(v)), Int(!iszero(ref)))
    end
    return iszero(overlap) ? 0.0 : accumulator / overlap
end
function coherence(::Type{<:Complex}, reference, vector)
    z = MultiScalar(0.0 + 0im, 0)
    accumulator, overlap = mapreduce(+, pairs(vector); init=z) do ((k, v))
        ref = sign(reference[k])
        MultiScalar(
            ComplexF64(sign(real(v)) * sign(ref) + im * sign(imag(v)) * sign(ref)),
            Int(!iszero(ref))
        )
    end
    return iszero(overlap) ? 0.0 : accumulator / overlap
end

"""
    WalkerLoneliness(threshold=1) <: PostStepStrategy

After each step, compute the proportion of configurations that are occupied by at most
`threshold` walkers. Reports to a column named `loneliness`.
"""
struct WalkerLoneliness{T} <: PostStepStrategy
    threshold::T
end
WalkerLoneliness() = WalkerLoneliness(1)

function post_step(wl::WalkerLoneliness, replica)
    vector = replica.v
    return (:loneliness => loneliness(valtype(vector), vector, wl.threshold),)
end

function loneliness(::Type{<:Real}, vector, threshold)
    num_lonely = mapreduce(+, values(vector), init=0) do v
        abs(v) ≤ threshold
    end
    return num_lonely / length(vector)
end

function loneliness(::Type{<:Complex}, vector, threshold)
    num_lonely = mapreduce(+, values(vector), init=0 + 0im) do v
        (abs(real(v)) ≤ threshold) + im*(abs(imag(v)) ≤ threshold)
    end
    return num_lonely / length(vector)
end

"""
    Timer <: PostStepStrategy

Record current time after every step. See [`Base.time`](@ref) for information on what time
is recorded.
"""
struct Timer <: PostStepStrategy end

post_step(::Timer, _) = (:time => time(),)