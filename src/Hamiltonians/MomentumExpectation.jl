struct MomentumExpectation{C} <: AbstractHamiltonian{Float64} end

MomentumExpectation(component) = MomentumExpectation{component}()

LOStructure(::Type{MomentumExpectation}) = IsHermitian()
num_offdiagonals(ham::MomentumExpectation, add) = 0

momenta(M) = -cld(M,2)+1:fld(M,2)

function diagonal_element(mom::MomentumExpectation{1}, add::SingleComponentFockAddress)
    M = num_modes(add)
    return float(dot(momenta(M), OccupiedModeMap(add)))
end
function diagonal_element(mom::MomentumExpectation{1}, add::BoseFS2C)
    M = num_modes(add)
    return float(dot(momenta(M), OccupiedModeMap(add.bsa)))
end
function diagonal_element(mom::MomentumExpectation{2}, add::BoseFS2C)
    M = num_modes(add)
    return float(dot(momenta(M), OccupiedModeMap(add.bsb)))
end
function diagonal_element(mom::MomentumExpectation{N}, add::CompositeFS) where {N}
    M = num_modes(add)
    return float(dot(momenta(M), OccupiedModeMap(add.components[N])))
end
