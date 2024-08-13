"""
    SingleParticleExcitation(i, j) <: AbstractHamiltonian

Represent the ``{i,j}`` element of the single-particle reduced density matrix:

```math
\\hat{ρ}^{(1)}_{i,j} = \\hat a^†_{i} \\hat a_{j}
```

where `i <: Int` and `j <: Int` specify the mode numbers.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`TwoParticleExcitation`](@ref)
"""
struct SingleParticleExcitation{I,J} <: AbstractHamiltonian{Float64}
end

SingleParticleExcitation(I::Int,J::Int) = SingleParticleExcitation{I,J}()

function Base.show(io::IO, spd::SingleParticleExcitation{I,J}) where {I,J}
    print(io, "SingleParticleExcitation($(I), $(J))")
end

LOStructure(::Type{T}) where T<:SingleParticleExcitation = AdjointUnknown()

function diagonal_element(spd::SingleParticleExcitation{I,J}, add::SingleComponentFockAddress) where {I,J}
    if I != J
        return 0.0
    end
    src = find_mode(add, J)
    return src.occnum
end

function num_offdiagonals(spd::SingleParticleExcitation{I,J}, address::SingleComponentFockAddress) where {I,J}
    if I == J
        return 0
    else
        return 1
    end
end

function get_offdiagonal(spd::SingleParticleExcitation{I,J}, add::SingleComponentFockAddress, chosen) where {I,J}
    src = find_mode(add, J)
    dst = find_mode(add,I)
    address, value = excitation(add, (dst,), (src,))
    return address, value
end

"""
    TwoParticleExcitation(i, j, k, l) <: AbstractHamiltonian

Represent {ij, kl} element of the two-particle reduced density matrix:

```math
\\hat{ρ}^{(2)}_{ij, kl} =  \\hat a^†_{i} \\hat a^†_{j} \\hat a_{l} \\hat a_{k} 
```

where ``i``, ``j``, ``k``, and ``l`` are the `mode`

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleExcitation`](@ref)
"""
struct TwoParticleExcitation{I,J,K,L} <: AbstractHamiltonian{Float64}
end

TwoParticleExcitation(I::Int,J::Int,K::Int,L::Int) = TwoParticleExcitation{I,J,K,L}()

function Base.show(io::IO, spd::TwoParticleExcitation{I,J,K,L}) where {I,J,K,L}
    print(io, "TwoParticleExcitation($(I), $(J), $(K), $(L))")
end

LOStructure(::Type{<:TwoParticleExcitation}) = AdjointUnknown()

function diagonal_element(spd::TwoParticleExcitation{I,J,K,L}, add::SingleComponentFockAddress) where {I,J,K,L}
    src = find_mode(add, (L, K))
    dst = find_mode(add,(I, J))
    address, value = excitation(add, (dst...,), (src...,))
    if (I, J) == (K, L)
        return value
    else
        return 0.0
    end
end

function num_offdiagonals(spd::TwoParticleExcitation{I,J,K,L}, address::SingleComponentFockAddress) where {I,J,K,L}
    if (I, J) == (K, L)
        return 0
    else
        return 1
    end
end

function get_offdiagonal(spd::TwoParticleExcitation{I,J,K,L}, add::SingleComponentFockAddress, chosen) where {I,J,K,L}
    src = find_mode(add, (L, K))
    dst = find_mode(add,(I, J))
    address, value = excitation(add, (dst...,), (src...,))
    return address, value
end
