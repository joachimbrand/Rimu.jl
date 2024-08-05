"""
    SingleParticleExcitation(i, j) <: AbstractHamiltonian

Represent {i,j} element of the single-particle reduced density matrix:

```math
\\hat{ρ}^{(1)}_{i,j} = \\hat a^†_{i} \\hat a_{j}
```

where ``i`` and ``j`` are the `mode`

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`TwoParticleExcitation`](@ref)
"""
struct SingleParticleExcitation{I,J} <: AbstractHamiltonian{Float64}
end

SingleParticleExcitation(I::Int,J::Int) = SingleParticleExcitation{I,J}()

function Base.show(io::IO, spd::SingleParticleExcitation)
    print(io, "SingleParticleExcitation($(spd.I), $(spd.J))")
end

LOStructure(::Type{T}) where T<:SingleParticleExcitation = AdjointUnknown()
Base.getproperty(::SingleParticleExcitation{I}, ::Val{:I}) where I = I
Base.getproperty(::SingleParticleExcitation{<:Any,J}, ::Val{:J}) where J = J

function diagonal_element(spd::SingleParticleExcitation, add::SingleComponentFockAddress)
    if spd.I != spd.J
        return 0.0
    end
    src = find_mode(add, spd.J)
    return src.occnum
end

function num_offdiagonals(spd::SingleParticleExcitation, address::SingleComponentFockAddress)
    if spd.I == spd.J
        return 0
    else
        return 1
    end
end

function get_offdiagonal(spd::SingleParticleExcitation, add::SingleComponentFockAddress, chosen)
    src = find_mode(add, spd.J)
    dst = find_mode(add,spd.I)
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

SingleParticleExcitation(I::Int,J::Int,K::Int,L::Int) = TwoParticleExcitation{I,J,K,L}()

function Base.show(io::IO, spd::TwoParticleExcitation)
    print(io, "TwoParticleExcitation($(spd.I), $(spd.J), $(spd.K), $(spd.L))")
end

LOStructure(::Type{T}) where T<:TwoParticleExcitation = AdjointUnknown()
Base.getproperty(::TwoParticleExcitation{I}, ::Val{:I}) where I = I
Base.getproperty(::TwoParticleExcitation{<:Any,J}, ::Val{:J}) where J = J
Base.getproperty(::TwoParticleExcitation{<:Any,<:Any,K}, ::Val{:K}) where K = K
Base.getproperty(::TwoParticleExcitation{<:Any,<:Any,<:Any,L}, ::Val{:L}) where L = L

function diagonal_element(spd::TwoParticleExcitation, add::SingleComponentFockAddress)
    src = find_mode(add, (spd.L, spd.K))
    dst = find_mode(add,(spd.I, spd.J))
    address, value = excitation(add, (dst...,), (src...,))
    if (spd.I, spd.J) == (spd.K, spd.L)
        return value
    else
        return 0.0
    end
end

function num_offdiagonals(spd::TwoParticleExcitation, address::SingleComponentFockAddress)
    if (spd.I, spd.J) == (spd.K, spd.L)
        return 0
    else
        return 1
    end
end

function get_offdiagonal(spd::TwoParticleExcitation, add::SingleComponentFockAddress, chosen)
    src = find_mode(add, (spd.L, spd.K))
    dst = find_mode(add,(spd.I, spd.J))
    address, value = excitation(add, (dst...,), (src...,))
    return address, value
end
