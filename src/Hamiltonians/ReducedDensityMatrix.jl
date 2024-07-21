"""
    SingleParticleReducedDensityMatrix(i, j) <: AbstractHamiltonian

Represent a {i,j} element of the single-particle reduced density matrix:

```math
\\hat{ρ}^{(1)}_{i,j} = \\langle \\psi | \\hat a^†_{i} \\hat a_{j} | \\psi \\rangle
```

where ``i`` and ``j`` are the `mode` and ``| \\psi \\rangle`` is the `state-ket` of the given Hamiltonian.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`TwoParticleReducedDensityMatrix`](@ref)
"""

struct SingleParticleReducedDensityMatrix <: AbstractHamiltonian{Float64}
    I::Int
    J::Int
end

SingleParticleReducedDensityMatrix(i,j) = SingleParticleReducedDensityMatrix{}(i,j)

function Base.show(io::IO, spd::SingleParticleReducedDensityMatrix)
    print(io, "SingleParticleReducedDensityMatrix($(spd.I), $(spd.J))")
end

LOStructure(::Type{T}) where T<:SingleParticleReducedDensityMatrix = AdjointUnknown()

function diagonal_element(spd::SingleParticleReducedDensityMatrix, add::SingleComponentFockAddress)
    src = find_mode(add, spd.J)
    dst = find_mode(add,spd.I)
    address, value = excitation(add, (dst,), (src,))
    if spd.I == spd.J
        return value
    else
        return 0.0
    end
end

function num_offdiagonals(spd::SingleParticleReducedDensityMatrix, address::SingleComponentFockAddress)
    if spd.I == spd.J
        return 0
    else
        return 2
    end
end

function get_offdiagonal(spd::SingleParticleReducedDensityMatrix, add::SingleComponentFockAddress, chosen)
    src = find_mode(add, spd.J)
    dst = find_mode(add,spd.I)
    if chosen == 2
        src, dst = dst, src
    end
    address, value = excitation(add, (dst,), (src,))
    if (chosen == 2 && add isa FermiFS)
        return address, -value/2
    else
        return address, value/2
    end
end

"""
    TwoParticleReducedDensityMatrix(i, j, k, l) <: AbstractHamiltonian

Represent a {ij, kl} element of the two-particle reduced density matrix:

```math
\\hat{ρ}^{(2)}_{ij, kl} = \\langle \\psi | \\hat a^†_{i} \\hat a^†_{j} \\hat a_{l} \\hat a_{k} | \\psi \\rangle
```

where ``i``, ``j``, ``k``, and ``l`` are the `mode` and ``| \\psi \\rangle`` is the `state-ket` of the given Hamiltonian.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleReducedDensityMatrix`](@ref)
"""

struct TwoParticleReducedDensityMatrix <: AbstractHamiltonian{Float64}
    I::Int
    J::Int
    K::Int
    L::Int
end

TwoParticleReducedDensityMatrix(i,j,k,l) = TwoParticleReducedDensityMatrix{}(i,j,k,l)

function Base.show(io::IO, spd::TwoParticleReducedDensityMatrix)
    print(io, "TwoParticleReducedDensityMatrix($(spd.I), $(spd.J), $(spd.K), $(spd.L))")
end

LOStructure(::Type{T}) where T<:TwoParticleReducedDensityMatrix = AdjointUnknown()

function diagonal_element(spd::TwoParticleReducedDensityMatrix, add::SingleComponentFockAddress)
    src = find_mode(add, (spd.L, spd.K))
    dst = find_mode(add,(spd.I, spd.J))
    address, value = excitation(add, (dst...,), (src...,))
    if (spd.I, spd.J) == (spd.K, spd.L)
        return value
    else
        return 0.0
    end
end

function num_offdiagonals(spd::TwoParticleReducedDensityMatrix, address::SingleComponentFockAddress)
    if (spd.I, spd.J) == (spd.K, spd.L)
        return 0
    else
        return 6
    end
end

function get_offdiagonal(spd::TwoParticleReducedDensityMatrix, add::BoseFS, chosen)
    if chosen<=2
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.I, spd.J))
    elseif chosen <=4 && chosen>2
        src = find_mode(add, (spd.K, spd.L))
        dst = find_mode(add,(spd.I, spd.J))
    else
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.J, spd.I))
    end
    if chosen%2 == 0
        src, dst = dst, src
    end
    address, value = excitation(add, (dst...,), (src...,))
    return address, value/6
end

function get_offdiagonal(spd::TwoParticleReducedDensityMatrix, add::FermiFS, chosen)
    if chosen <= 2
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.I, spd.J))
    elseif chosen <= 4 && chosen > 2
        src = find_mode(add, (spd.K, spd.L))
        dst = find_mode(add,(spd.I, spd.J))
    else
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.J, spd.I))
    end
    if chosen%2 == 0
        src, dst = dst, src
    end
    address, value = excitation(add, (dst...,), (src...,))
    if chosen <= 2
        return address, value/6
    else
        return address, -value/6
    end
end

