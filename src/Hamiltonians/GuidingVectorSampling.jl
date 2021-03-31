"""
    GuidingVectorSampling

Wrapper over any `AbstractHamiltonian` that implements guided vector a.k.a. guided wave
function sampling. In this importance sampling scheme the Hamiltonian is modified as follows.

```math
\\tilde{H}_{ij} = v_i H_{ij} v_j^{-1}
```

and where `v` is the guiding vector. `v_i` and `v_j` are also thresholded to avoid dividing
by zero (see below).

# Constructors

* `GuidingVectorSampling(::AbstractHamiltonian, vector, eps)`
* `GuidingVectorSampling(::AbstractHamiltonian; vector, eps)`

`eps` is a thresholding parameter used to avoid dividing by zero; all values below `eps` are
set to `eps`. It is recommended that `eps` is in the same value range as the guiding
vector. The default value is set to `eps=norm(v, Inf) * 1e-2`

After construction, we can access the underlying hamiltonian with `G.hamiltonian`, the
`eps` parameter with `G.eps`, and the guiding vector with `G.vector`.

# Example

```jldoctest
julia> H = HubbardReal1D(BoseFS{3}((1,1,1)); u=6.0, t=1.0);

julia> v = DVec2(starting_address(H) => 10; capacity=1);

julia> G = GuidingVectorSampling(H, v, 0.1);

julia> get_offdiagonal(H, starting_address(H), 4)
(BoseFS{3,3}((2, 0, 1)), -1.4142135623730951)

julia> get_offdiagonal(G, starting_address(G), 4)
(BoseFS{3,3}((2, 0, 1)), -141.4213562373095)
```
"""
struct GuidingVectorSampling{A,T,H<:AbstractHamiltonian{T},D,E} <: AbstractHamiltonian{T}
    # The A parameter sets whether this is an adjoint or not.
    # The E parameter is the epsilon value.
    hamiltonian::H
    vector::D
end

function GuidingVectorSampling(h, v::AbstractDVec, eps=1e-2 * norm(v, Inf))
    return GuidingVectorSampling{false,eltype(h),typeof(h),typeof(v),eps}(h, v)
end
function GuidingVectorSampling(h; vector, eps=1e-2 * norm(v, Inf))
    return GuidedVectorSampling(h, vector, eps)
end

starting_address(h::GuidingVectorSampling) = starting_address(h.hamiltonian)
LOStructure(::Type{<:GuidingVectorSampling{<:Any,<:Any,H}}) where {H} = _lo_str(LOStructure(H))

function LinearAlgebra.adjoint(h::GuidingVectorSampling{A,T,<:Any,D,E}) where {A,T,D,E}
    h_adj = h.hamiltonian'
    return GuidingVectorSampling{!A,T,typeof(h_adj),D,E}(h_adj, h.vector)
end

Base.getproperty(h::GuidingVectorSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::GuidingVectorSampling{<:Any,<:Any,<:Any,<:Any,E}, ::Val{:eps}) where E = E
Base.getproperty(h::GuidingVectorSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)
Base.getproperty(h::GuidingVectorSampling, ::Val{:vector}) = getfield(h, :vector)

# Forward some interface functions.
num_offdiagonals(h::GuidingVectorSampling, add) = num_offdiagonals(h.hamiltonian, add)
diagonal_element(h::GuidingVectorSampling, add) = diagonal_element(h.hamiltonian, add)

_apply_eps(x, eps) = ifelse(iszero(x), eps, ifelse(abs(x) < eps, sign(x) * eps, x))

function guided_modify_element(value, guide1, guide2, eps)
    if iszero(guide1) && iszero(guide2)
        return value
    else
        guide1 = _apply_eps(guide1, eps)
        guide2 = _apply_eps(guide2, eps)
        return value * (guide2 / guide1)
    end
end

function get_offdiagonal(h::GuidingVectorSampling, add, chosen)
    return get_offdiagonal(h, add, chosen, h.vector[add])
end
function get_offdiagonal(h::GuidingVectorSampling{A}, add1, chosen, guide1) where A
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    guide2 = h.vector[add2]

    if A # adjoint
        return add2, guided_modify_element(matrix_element, guide2, guide1, h.eps)
    else
        return add2, guided_modify_element(matrix_element, guide1, guide2, h.eps)
    end
end

struct GuidingVectorOffdiagonals{
    A,E,F,T,H<:AbstractHamiltonian,D,N<:AbstractOffdiagonals{F,T}
}<:AbstractOffdiagonals{F,T}
    hamiltonian::H
    vector::D
    guide::T
    offdiagonals::N
end

function offdiagonals(h::GuidingVectorSampling{A,T,H,D,E}, a) where {A,T,H,D,E}
    hps = offdiagonals(h.hamiltonian, a)
    guide = h.vector[a]
    return GuidingVectorOffdiagonals{A,E,typeof(a),T,H,D,typeof(hps)}(
        h.hamiltonian, h.vector, guide, hps
    )
end

function Base.getindex(h::GuidingVectorOffdiagonals{A,E}, i) where {A,E}
    add2, matrix_element = h.offdiagonals[i]
    guide1 = h.guide
    guide2 = h.vector[add2]

    if A # adjoint
        return add2, guided_modify_element(matrix_element, guide2, guide1, E)
    else
        return add2, guided_modify_element(matrix_element, guide1, guide2, E)
    end
end

Base.size(h::GuidingVectorOffdiagonals) = size(h.offdiagonals)
