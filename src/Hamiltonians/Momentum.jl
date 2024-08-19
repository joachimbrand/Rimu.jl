"""
    Momentum(component=0; fold=true) <: AbstractHamiltonian

The momentum operator ``P̂``.

The component argument controls which component of the address is taken into
consideration. A value of 0 sums the contributions of all components. If `fold` is true, the
momentum is folded into the Brillouin zone.

```jldoctest
julia> address = BoseFS((1, 0, 2, 1, 2, 1, 1, 3))
BoseFS{11,8}(1, 0, 2, 1, 2, 1, 1, 3)

julia> v = DVec(address => 10);

julia> rayleigh_quotient(Momentum(), DVec(address => 1))
-2.0

julia> rayleigh_quotient(Momentum(fold=false), DVec(address => 1))
14.0
```
"""
struct Momentum{C} <: AbstractOperator{Float64}
    fold::Bool
end
Momentum(component=0; fold=true) = Momentum{component}(fold)
Base.show(io::IO, mom::Momentum{C}) where {C} = print(io, "Momentum($C; fold=$(mom.fold))")

LOStructure(::Type{<:Momentum}) = IsDiagonal()
function allows_address_type(::Momentum{C}, ::Type{A}) where {C, A}
    return C ≤ num_components(A)
end

@inline function _momentum(address::SingleComponentFockAddress, fold)
    M = num_modes(address)
    momentum = float(dot(-cld(M,2)+1:fld(M,2), OccupiedModeMap(address)))
    if fold
        return mod1(momentum + cld(M,2), M) - cld(M,2)
    else
        return momentum
    end
end
@inline function _momentum(adds, fold)
    M = num_modes(adds[1])
    momentum = sum(a -> _momentum(a, false), adds)
    if fold
        return mod1(momentum + cld(M,2), M) - cld(M,2)
    else
        return momentum
    end
end

diagonal_element(m::Momentum{0}, a::SingleComponentFockAddress) = _momentum(a, m.fold)
diagonal_element(m::Momentum{1}, a::SingleComponentFockAddress) = _momentum(a, m.fold)

diagonal_element(m::Momentum{0}, a::BoseFS2C) = _momentum((a.bsa, a.bsb), m.fold)
diagonal_element(m::Momentum{1}, a::BoseFS2C) = _momentum(a.bsa, m.fold)
diagonal_element(m::Momentum{2}, a::BoseFS2C) = _momentum(a.bsb, m.fold)

diagonal_element(m::Momentum{0}, a::CompositeFS) = _momentum(a.components, m.fold)
diagonal_element(m::Momentum{N}, a::CompositeFS) where {N} = _momentum(a.components[N], m.fold)
