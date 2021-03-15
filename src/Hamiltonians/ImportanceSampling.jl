"""
    ImportanceSampling(::AbstractHamiltonian; g)

Wrapper over any `AbstractHamiltonian` that makes it use importance sampling. In importance
sampling, a hop from address `i` to `j` is weighted by weight `w`, where

```math
w = exp[-g(H_{jj} - H_{ii})]
```

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS{3}((1,1,1)); u=6.0, t=1.0)
HubbardMom1D(BoseFS{3,3}((1, 1, 1)); u=6.0, t=1.0)

julia> I = ImportanceSampling(H, g=0.3)
ImportanceSampling(HubbardMom1D(BoseFS{3,3}((1, 1, 1)); u=6.0, t=1.0); g=0.3)

julia> hop(H, BoseFS((2, 1, 0)), 1)
(BoseFS{3,3}((1, 0, 2)), 2.0)

julia> hop(I, BoseFS((2, 1, 0)), 1)
(BoseFS{3,3}((1, 0, 2)), 0.8131393194811987)
```
"""
struct ImportanceSampling{T,H<:AbstractHamiltonian{T},G} <: AbstractHamiltonian{T}
    hamiltonian::H
end

function ImportanceSampling(h, g)
    G = eltype(h)(g)
    return ImportanceSampling{eltype(h), typeof(h), G}(h)
end
ImportanceSampling(h; g) = ImportanceSampling(h, g)

function Base.show(io::IO, h::ImportanceSampling)
    print(io, "ImportanceSampling(", h.hamiltonian, "; g=", h.g, ")")
end

starting_address(h::ImportanceSampling) = starting_address(h.hamiltonian)

Base.getproperty(h::ImportanceSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::ImportanceSampling{<:Any,<:Any,G}, ::Val{:g}) where G = G
Base.getproperty(h::ImportanceSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)

# Forward some interface functions.
numOfHops(h::ImportanceSampling) = numOfHops(h.hamiltonian)
diagME(h::ImportanceSampling, add) = diagME(h.hamiltonian, add)

function hop(h::ImportanceSampling, add, chosen)
    return hop(h, add, chosen, diagME(h, add))
end
function hop(h::ImportanceSampling, add1, chosen, diag1)
    add2, matrix_element = hop(h.hamiltonian, add1, chosen)
    diag2 = diagME(h, add2)

    return add2, matrix_element * exp(-h.g * (diag2 - diag1))
end

struct ImportanceHops{G,A,T,H<:AbstractHamiltonian,N<:AbstractHops{A,T}} <: AbstractHops{A,T}
    hamiltonian::H
    diag::T
    hops::N
end

function hops(ham::ImportanceSampling{<:Any,<:Any,G}, a) where G
    hps = hops(ham.hamiltonian, a)
    diag = diagME(ham, a)
    return ImportanceHops{G,typeof(a),eltype(ham),typeof(ham),typeof(hps)}(ham, diag, hps)
end

function Base.getindex(h::ImportanceHops{G}, i) where G
    add2, matrix_element = h.hops[i]
    diag2 = diagME(h.hamiltonian, add2)
    return add2, matrix_element * exp(-G * (diag2 - h.diag))
end

Base.size(h::ImportanceHops) = size(h.hops)
