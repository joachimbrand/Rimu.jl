"""
    SingleParticleExcitation(i, j) <: AbstractHamiltonian

Represent the ``{i,j}`` element of the single-particle reduced density matrix:

```math
ρ̂^{(1)}_{i,j} = â^†_{i} â_{j}
```

where `i <: Int` and `j <: Int` specify the mode numbers.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`TwoParticleExcitation`](@ref)
"""
struct SingleParticleExcitation{I,J} <: AbstractOperator{Float64}
end

SingleParticleExcitation(I::Int,J::Int) = SingleParticleExcitation{I,J}()

function Base.show(io::IO, ::SingleParticleExcitation{I,J}) where {I,J}
    print(io, "SingleParticleExcitation($(I), $(J))")
end

LOStructure(::Type{<:SingleParticleExcitation}) = AdjointUnknown()
function allows_address_type(::SingleParticleExcitation{I,J}, ::Type{A}) where {I,J,A}
    return A <: SingleComponentFockAddress && I ≤ num_modes(A) && J ≤ num_modes(A)
end

function diagonal_element(
    ::SingleParticleExcitation{I,J}, addr::SingleComponentFockAddress
) where {I,J}
    if I != J
        return 0.0
    else
        src = find_mode(addr, J)
        return src.occnum
    end
end

function num_offdiagonals(
    ::SingleParticleExcitation{I,J}, ::SingleComponentFockAddress
) where {I,J}
    if I == J
        return 0
    else
        return 1
    end
end

function get_offdiagonal(
    ::SingleParticleExcitation{I,J}, addr::SingleComponentFockAddress, _
) where {I,J}
    src = find_mode(addr, J)
    dst = find_mode(addr, I)
    address, value = excitation(addr, (dst,), (src,))
    return address, value
end

"""
    TwoParticleExcitation(i, j, k, l) <: AbstractHamiltonian

Represent the ``{ij, kl}`` element of the two-particle reduced density matrix:

```math
ρ̂^{(2)}_{ij, kl} =  â^†_{i} â^†_{j} â_{l} â_{k}
```

where `i`, `j`, `k`, and `l` (all `<: Int`) specify the mode numbers.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleExcitation`](@ref)
"""
struct TwoParticleExcitation{I,J,K,L} <: AbstractOperator{Float64}
end

TwoParticleExcitation(I::Int,J::Int,K::Int,L::Int) = TwoParticleExcitation{I,J,K,L}()

function Base.show(io::IO, ::TwoParticleExcitation{I,J,K,L}) where {I,J,K,L}
    print(io, "TwoParticleExcitation($(I), $(J), $(K), $(L))")
end

LOStructure(::Type{<:TwoParticleExcitation}) = AdjointUnknown()
function allows_address_type(::TwoParticleExcitation{I,J,K,L}, ::Type{A}) where {I,J,K,L,A}
    return A <: SingleComponentFockAddress && I ≤ num_modes(A) && J ≤ num_modes(A) &&
            K ≤ num_modes(A) && L ≤ num_modes(A)
end

function diagonal_element(
    ::TwoParticleExcitation{I,J,K,L}, addr::SingleComponentFockAddress
) where {I,J,K,L}
    if (I, J) == (K, L) || (I, J) == (L, K)
        src = find_mode(addr, (L, K))
        dst = find_mode(addr, (I, J))
        return excitation(addr, dst, src)[2]
    else
        return 0.0
    end
end

function num_offdiagonals(
    ::TwoParticleExcitation{I,J,K,L}, ::SingleComponentFockAddress
) where {I,J,K,L}
    if (I, J) == (K, L) || (I, J) == (L, K)
        return 0
    else
        return 1
    end
end

function get_offdiagonal(
    ::TwoParticleExcitation{I,J,K,L}, addr::SingleComponentFockAddress, _
) where {I,J,K,L}
    src = find_mode(addr, (L, K))
    dst = find_mode(addr, (I, J))
    address, value = excitation(addr, dst, src)
    return address, value
end
