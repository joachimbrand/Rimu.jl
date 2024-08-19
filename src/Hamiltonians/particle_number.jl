"""
    ParticleNumberOperator() <: AbstractOperator{Float64}

The number operator in Fock space. This operator is diagonal in the Fock basis and
returns the number of particles in the Fock state. It works with any address type that
is a subtype of [`AbstractFockAddress`](@ref).

```jldoctest; filter = r"(\\d*)\\.(\\d{4})\\d+" => s"\\1.\\2***"
julia> p = ExactDiagonalizationProblem(FroehlichPolaron(fs"|0 0⟩{}"; mode_cutoff=5, v=3));

julia> gs = solve(p).vectors[1]; # normalised ground state vector

julia> dot(gs, ParticleNumberOperator(), gs) # particle number expectation value
2.8823297252925917
```

See also [`AbstractHamiltonian`](@ref).
"""
struct ParticleNumberOperator <: AbstractOperator{Float64} end

LOStructure(::Type{<:ParticleNumberOperator}) = IsDiagonal()

function diagonal_element(::ParticleNumberOperator, addr::AbstractFockAddress)
    return float(num_particles(addr))
end
function allows_address_type(::ParticleNumberOperator, ::Type{B}) where {B}
    return B <: AbstractFockAddress
end
