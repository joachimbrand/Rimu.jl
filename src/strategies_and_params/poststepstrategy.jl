"""
    PostStepStrategy

Subtypes of `PostStepStrategy` can be used to perform arbitrary computation on a single
state after an FCIQMC step is finished and report the results.

# Implemented strategies:

* [`ProjectedEnergy`](@ref)
* [`Projector`](@ref)
* [`SignCoherence`](@ref)
* [`WalkerLoneliness`](@ref)
* [`Timer`](@ref)

Note: A tuple of multiple strategies can be passed to [`ProjectorMonteCarloProblem`](@ref).
In that case, all reported column names must be distinct.

# Interface:

A subtype of this type must implement
[`post_step_action(::PostStepStrategy, ::SingleState, step::Int)`](@ref).
"""
abstract type PostStepStrategy end

"""
    post_step_action(::PostStepStrategy, ::SingleState, step) -> kvpairs

Compute statistics after FCIQMC step. Should return a tuple of `:key => value` pairs.
This function is only called every [`reporting_interval`](@ref) steps, as defined by the
`ReportingStrategy`.

See also [`PostStepStrategy`](@ref), [`ReportingStrategy`](@ref).
"""
post_step_action

# When startegies are a Tuple, apply all of them.
function post_step_action(::Tuple{}, _, _)
    return ()
end
function post_step_action((t,ts...)::Tuple, single_state, step)
    head = post_step_action(t, single_state, step)
    rest = post_step_action(ts, single_state, step)
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

function post_step_action(p::Projector, single_state, _)
    return (p.name => dot(p.projector, single_state.v),)
end

"""
    ProjectedEnergy(hamiltonian, projector; hproj=:hproj, vproj=:vproj) <: PostStepStrategy

After every step, compute `hproj = dot(projector, hamiltonian, dv)` and `vproj =
dot(projector, dv)`, where `dv` is the instantaneous coefficient vector.  `projector` can be
an [`AbstractDVec`](@ref), or an [`AbstractProjector`](@ref).

Reports to columns `hproj` and `vproj`, which can be used to compute projective energy,
e.g. with [`projected_energy`](@ref). The keyword arguments `hproj` and
`vproj` can be used to change the names of these columns. This can be used to make the names
unique when computing projected energies with different projectors in the same run.

See also [`projected_energy`](@ref), [`ratio_of_means`](@ref), [`mixed_estimator`](@ref),
and [`PostStepStrategy`](@ref).
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
    hproj_vec = compute_hproj(hamiltonian, projector)
    return ProjectedEnergy(vproj, hproj, hamiltonian, freeze(projector), hproj_vec)
end
compute_hproj(hamiltonian, projector::AbstractProjector) = nothing
# compute `dot` products with `AbstractProjector`s lazily
function compute_hproj(hamiltonian, projector)
    return compute_hproj(LOStructure(hamiltonian), hamiltonian, projector)
end
function compute_hproj(::AdjointUnknown, hamiltonian, projector)
    @warn "$(typeof(hamiltonian)) has an unknown adjoint. This will be slow."
    return nothing
end
function compute_hproj(::LOStructure, ham, projector)
    return freeze(ham' * projector)
end

function post_step_action(p::ProjectedEnergy{<:Any,<:Any,Nothing}, single_state, _)
    return (
        p.vproj_name => dot(p.vproj, single_state.v),
        p.hproj_name => dot(p.vproj, p.ham, single_state.v),
    )
end
function post_step_action(p::ProjectedEnergy, single_state, _)
    return (
        p.vproj_name => dot(p.vproj, single_state.v),
        p.hproj_name => dot(p.hproj, single_state.v),
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

function post_step_action(sc::SignCoherence, single_state, _)
    vector = single_state.v
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

function post_step_action(wl::WalkerLoneliness, single_state, _)
    vector = single_state.v
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

Record current time after every step. See `Base.Libc.time` for information on what time
is recorded.
"""
struct Timer <: PostStepStrategy end

post_step_action(::Timer, _, _) = (:time => time(),)

"""
    SingleParticleDensity(; save_every=1, component) <: PostStepStrategy

[`PostStepStrategy`](@ref)  to  compute the diagonal [`single_particle_density`](@ref).
It records a `Tuple` with the same `eltype` as the vector.

Computing the density at every time step can be expensive. This cost can be reduced by
setting the `save_every` argument to a higher value. If the value is set, a vector of zeros
is recorded when the saving is skipped.

If the address type has multiple components, the `component` argument can be used to compute
the density on a per-component basis.

The density is not normalized, and must be divided by the vector `norm(⋅,2)` squared.

# See also

* [`single_particle_density`](@ref)
* [`DensityMatrixDiagonal`](@ref)
"""
struct SingleParticleDensity <: PostStepStrategy
    save_every::Int
    component::Int

    SingleParticleDensity(;save_every=1, component=0) = new(save_every, component)
end

function post_step_action(d::SingleParticleDensity, single_state, step)
    component = d.component
    if component == 0
        name = :single_particle_density
    else
        name = Symbol("single_particle_density_", component)
    end
    vector = single_state.v
    if step % d.save_every == 0
        return (name => single_particle_density(vector; component),)
    else
        V = valtype(vector)
        M = num_modes(keytype(vector))
        return (name => ntuple(_ -> 0.0, Val(M)),)
    end
end

"""
    single_particle_density(dvec; component)
    single_particle_density(add; component)

Compute the diagonal single particle density of vector `dvec` or address `add`. If the
`component` argument is given, only that component of the addresses is taken into
account. The result is always normalized so that `sum(result) ≈ num_particles(address)`.

# Examples

```jldoctest
julia> v = DVec(fs"|⋅↑⇅↓⋅⟩" => 1.0, fs"|↓↓⋅↑↑⟩" => 0.5)
DVec{FermiFS2C{2, 2, 5, 4, FermiFS{2, 5, BitString{5, 1, UInt8}}, FermiFS{2, 5, BitString{5, 1, UInt8}}},Float64} with 2 entries, style = IsDeterministic{Float64}()
  fs"|↓↓⋅↑↑⟩" => 0.5
  fs"|⋅↑⇅↓⋅⟩" => 1.0

julia> single_particle_density(v)
(0.2, 1.0, 1.6, 1.0, 0.2)

julia> single_particle_density(v; component=1)
(0.0, 1.6, 1.6, 0.4, 0.4)
```

# See also

* [`SingleParticleDensity`](@ref)
"""
function single_particle_density(dvec; component=0)
    K = keytype(dvec)
    V = float(valtype(dvec))
    M = num_modes(K)
    N = num_particles(K)

    result = mapreduce(
        +, pairs(dvec);
        init=MultiScalar(ntuple(_ -> zero(V), Val(M)))
    ) do (k, v)
        MultiScalar(v^2 .* single_particle_density(k; component))
    end

    return result.tuple ./ sum(result.tuple) .* N
end

function single_particle_density(add::SingleComponentFockAddress; component=0)
    return float.(Tuple(onr(add)))
end
function single_particle_density(add::Union{CompositeFS,BoseFS2C}; component=0)
    if component == 0
        return float.(Tuple(sum(onr(add))))
    else
        return float.(Tuple(onr(add)[component]))
    end
end
