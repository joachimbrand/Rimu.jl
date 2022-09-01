"""
    Momentum(component=0; fold=true) <: AbstractHamiltonian

The momentum operator ``\\hat{p}``.

The component argument controls which component of the address is taken into
consideration. A value of 0 sums the contributions of all components. If `fold` is true, the
momentum is folded into the Brillouin zone.

```jldoctest
julia> add = BoseFS((1, 0, 2, 1, 2, 1, 1, 3))
BoseFS{11,8}((1, 0, 2, 1, 2, 1, 1, 3))

julia> v = DVec(add => 10);

julia> rayleigh_quotient(Momentum(), DVec(add => 1))
-2.0

julia> rayleigh_quotient(Momentum(fold=false), DVec(add => 1))
14.0
```
"""
struct Momentum{C} <: AbstractHamiltonian{Float64}
    fold::Bool
end
Momentum(component=0; fold=true) = Momentum{component}(fold)
Base.show(io::IO, mom::Momentum{C}) where {C} = print(io, "Momentum($C; fold=$(mom.fold))")

LOStructure(::Type{<:Momentum}) = IsDiagonal()
num_offdiagonals(ham::Momentum, add) = 0

function fold_mod(m, M)
    m = mod(m, M)
    return m > M ÷ 2 ? m - M : m
end

function momentum(add::SingleComponentFockAddress; fold=false)
    M = num_modes(add)
    res = 0
    for i in OccupiedModeMap(add)
        res += (i.mode - cld(M, 2)) * i.occnum
    end
    if fold
        return float(fold_mod(res, M))
    else
        return float(res)
    end
end
function momentum(add::CompositeFS; fold=false)
    M = num_modes(add)
    res = sum(add.components) do fs
        momentum(fs; fold=false)
    end
    if fold
        return float(fold_mod(res, M))
    else
        return float(res)
    end
end

diagonal_element(m::Momentum{0}, a::SingleComponentFockAddress) = momentum(a; fold=m.fold)
diagonal_element(m::Momentum{1}, a::SingleComponentFockAddress) = momentum(a; fold=m.fold)

# TODO: this is to be removed along with BoseFS2C
diagonal_element(m::Momentum{0}, a::BoseFS2C) =
    momentum(CompositeFS(a.bsa, a.bsb); fold=m.fold)
diagonal_element(m::Momentum{1}, a::BoseFS2C) = momentum(a.bsa; fold=m.fold)
diagonal_element(m::Momentum{2}, a::BoseFS2C) = momentum(a.bsb; fold=m.fold)

diagonal_element(m::Momentum{0}, a::CompositeFS) = momentum(a; fold=m.fold)
diagonal_element(m::Momentum{N}, a::CompositeFS) where {N} =
    momentum(a.components[N]; fold=m.fold)
