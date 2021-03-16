"""
    GutzwillerSampling(::AbstractHamiltonian; g)

Wrapper over any `AbstractHamiltonian` that makes it use importance sampling. In importance
sampling, a hop from address `i` to `j` is weighted by weight `w`, where

```math
w = exp[-g(H_{jj} - H_{ii})]
```

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS{3}((1,1,1)); u=6.0, t=1.0)
HubbardMom1D(BoseFS{3,3}((1, 1, 1)); u=6.0, t=1.0)

julia> I = GutzwillerSampling(H, g=0.3)
GutzwillerSampling(HubbardMom1D(BoseFS{3,3}((1, 1, 1)); u=6.0, t=1.0); g=0.3)

julia> hop(H, BoseFS((2, 1, 0)), 1)
(BoseFS{3,3}((1, 0, 2)), 2.0)

julia> hop(I, BoseFS((2, 1, 0)), 1)
(BoseFS{3,3}((1, 0, 2)), 0.8131393194811987)
```
"""
struct GutzwillerSampling{A,T,H<:AbstractHamiltonian{T},G} <: AbstractHamiltonian{T}
    hamiltonian::H
end

function GutzwillerSampling(h, g)
    G = eltype(h)(g)
    return GutzwillerSampling{false,eltype(h),typeof(h),G}(h)
end
GutzwillerSampling(h; g) = GutzwillerSampling(h, g)

function Base.show(io::IO, h::GutzwillerSampling{A}) where {A}
    A && print(io, "adjoint(")
    print(io, "GutzwillerSampling(", h.hamiltonian, "; g=", h.g, ")")
    A && print(io, ")")
end

starting_address(h::GutzwillerSampling) = starting_address(h.hamiltonian)

LOStructure(::Type{<:GutzwillerSampling{<:Any,<:Any,H}}) where H = WrapperLO(LOStructure(H))
LOAdjoint(::Type{<:GutzwillerSampling{<:Any,<:Any,H}}) where H = LOAdjoint(H)

function LinearAlgebra.adjoint(h::GutzwillerSampling{A,T,<:Any,G}) where {A,T,G}
    h_adj = h.hamiltonian'
    return GutzwillerSampling{!A,T,typeof(h_adj),G}(h_adj)
end

Base.getproperty(h::GutzwillerSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::GutzwillerSampling{<:Any,<:Any,<:Any,G}, ::Val{:g}) where G = G
Base.getproperty(h::GutzwillerSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)

# Forward some interface functions.
numOfHops(h::GutzwillerSampling, add) = numOfHops(h.hamiltonian, add)
diagME(h::GutzwillerSampling, add) = diagME(h.hamiltonian, add)

function hop(h::GutzwillerSampling, add, chosen)
    return hop(h, add, chosen, diagME(h, add))
end
function hop(h::GutzwillerSampling{A}, add1, chosen, diag1) where A
    add2, matrix_element = hop(h.hamiltonian, add1, chosen)
    diag2 = diagME(h, add2)

    if A # adjoint simply switches sign in exp term. Conj was already done in inner H
        return add2, matrix_element * exp(-h.g * (diag1 - diag2))
    else
        return add2, matrix_element * exp(-h.g * (diag2 - diag1))
    end
end

struct GutzwillerHops{A,G,F,T,H<:AbstractHamiltonian,N<:AbstractHops{F,T}}<:AbstractHops{F,T}
    hamiltonian::H
    diag::T
    hops::N
end

function hops(ham::GutzwillerSampling{A,<:Any,<:Any,G}, a) where {A,G}
    hps = hops(ham.hamiltonian, a)
    diag = diagME(ham, a)
    return GutzwillerHops{A,G,typeof(a),eltype(ham),typeof(ham),typeof(hps)}(ham, diag, hps)
end

function Base.getindex(h::GutzwillerHops{A,G}, i) where {A,G}
    add2, matrix_element = h.hops[i]
    diag2 = diagME(h.hamiltonian, add2)
    if A # adjoint simply switches sign in exp term. Conj was already done in inner H
        return add2, matrix_element * exp(-G * (h.diag - diag2))
    else
        return add2, matrix_element * exp(-G * (diag2 - h.diag))
    end
end

Base.size(h::GutzwillerHops) = size(h.hops)
