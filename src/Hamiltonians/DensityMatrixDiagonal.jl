"""
    DensityMatrixDiagonal(;mode, component=1)

Compute the diagonal of the density matrix:

```math
\\hat{ρ}_d = \\hat{n}_i,σ
```

where ``i`` is the `mode` and ``σ`` is the `component`.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
"""
struct DensityMatrixDiagonal{C} <: AbstractHamiltonian{Float64}
    mode::Int
end
function DensityMatrixDiagonal(; mode, component=1)
    if component ≤ 0
        throw(ArgumentError("Invalid component `$component`"))
    end
    return DensityMatrixDiagonal{component}(mode)
end

function diagonal_element(spd::DensityMatrixDiagonal{C}, add::CompositeFS) where {C}
    comp = add.components[C]
    return float(find_mode(comp, spd.mode).occnum)
end
function diagonal_element(spd::DensityMatrixDiagonal{1}, add::BoseFS2C)
    comp = add.bsa
    return float(find_mode(comp, spd.mode).occnum)
end
function diagonal_element(spd::DensityMatrixDiagonal{2}, add::BoseFS2C)
    comp = add.bsb
    return float(find_mode(comp, spd.mode).occnum)
end
function diagonal_element(spd::DensityMatrixDiagonal{1}, add::SingleComponentFockAddress)
    return float(find_mode(add, spd.mode).occnum)
end

num_offdiagonals(spd::DensityMatrixDiagonal, _) = 0
LOStructure(::Type{<:DensityMatrixDiagonal}) = IsHermitian()
