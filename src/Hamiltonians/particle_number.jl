"""
    ParticleNumberOperator([address]) <: AbstractHamiltonian

The number operator in Fock space. This operator is diagonal in the Fock basis and
returns the number of particles in the Fock state. Passing an address is optional.

```jldoctest
julia> h = FroehlichPolaron(fs"|0 0⟩{}"; mode_cutoff=5, v=3); bsr = BasisSetRepresentation(h);

julia> gs = DVec(zip(bsr.basis, eigen(Matrix(bsr)).vectors[:,1])); # ground state

julia> dot(gs, ParticleNumberOperator(), gs) # particle number expectation value
2.8823297252925917
```

See also [`AbstractHamiltonian`](@ref).
"""
struct ParticleNumberOperator{A} <: AbstractHamiltonian{Float64}
    address::A
end
ParticleNumberOperator() = ParticleNumberOperator(BoseFS(1,))

function Base.show(io::IO, n::ParticleNumberOperator)
    io = IOContext(io, :compact => true)
    print(io, "ParticleNumberOperator(")
    n.address === BoseFS(1,) || show(io, n.address) # suppress if default
    print(io, ")")
end

LOStructure(::Type{<:ParticleNumberOperator}) = IsDiagonal()
starting_address(n::ParticleNumberOperator) = n.address

function diagonal_element(::ParticleNumberOperator, addr::AbstractFockAddress)
    return float(num_particles(addr))
end
allowed_address_type(::ParticleNumberOperator) = AbstractFockAddress
