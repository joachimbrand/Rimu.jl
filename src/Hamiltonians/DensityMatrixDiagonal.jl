"""
    DensityMatrixDiagonal(mode; component=0) <: AbstractHamiltonian

Represent a diagonal element of the single-particle density:

```math
\\hat{n}_{i,σ} = \\hat a^†_{i,σ} \\hat a_{i,σ}
```

where ``i`` is the `mode` and ``σ`` is the `component`. If `component` is zero, the sum over
all components is computed.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleExcitation`](@ref)
"""
struct DensityMatrixDiagonal{C} <: AbstractHamiltonian{Float64}
    mode::Int
end
DensityMatrixDiagonal(mode; component=0) = DensityMatrixDiagonal{component}(mode)

function allows_address_type(dmd::DensityMatrixDiagonal{C}, ::Type{A}) where {C, A}
    return C ≤ num_components(A) && dmd.mode ≤ num_modes(A)
end

function diagonal_element(dmd::DensityMatrixDiagonal{1}, add::SingleComponentFockAddress)
    return float(find_mode(add, dmd.mode).occnum)
end
function diagonal_element(dmd::DensityMatrixDiagonal{0}, add::SingleComponentFockAddress)
    return float(find_mode(add, dmd.mode).occnum)
end

function diagonal_element(dmd::DensityMatrixDiagonal{0}, add::CompositeFS)
    return float(sum(a -> find_mode(a, dmd.mode).occnum, add.components))
end
function diagonal_element(dmd::DensityMatrixDiagonal{C}, add::CompositeFS) where {C}
    comp = add.components[C]
    return float(find_mode(comp, dmd.mode).occnum)
end

function diagonal_element(dmd::DensityMatrixDiagonal{0}, add::BoseFS2C)
    return float(find_mode(add.bsa, dmd.mode).occnum + find_mode(add.bsb, dmd.mode).occnum)
end
function diagonal_element(dmd::DensityMatrixDiagonal{1}, add::BoseFS2C)
    comp = add.bsa
    return float(find_mode(comp, dmd.mode).occnum)
end
function diagonal_element(dmd::DensityMatrixDiagonal{2}, add::BoseFS2C)
    comp = add.bsb
    return float(find_mode(comp, dmd.mode).occnum)
end

num_offdiagonals(dmd::DensityMatrixDiagonal, _) = 0
LOStructure(::Type{<:DensityMatrixDiagonal}) = IsDiagonal()
