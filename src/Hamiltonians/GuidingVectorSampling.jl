"""
    GuidingVectorSampling

Wrapper over any [`AbstractHamiltonian`](@ref) that implements guided vector a.k.a. guided
wave function sampling. In this importance sampling scheme the Hamiltonian is modified as
follows.

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
julia> H = HubbardReal1D(BoseFS(1,1,1); u=6.0, t=1.0);

julia> v = DVec(starting_address(H) => 10; capacity=1);

julia> G = GuidingVectorSampling(H, v, 0.1);

julia> get_offdiagonal(H, starting_address(H), 4)
(BoseFS{3,3}(2, 0, 1), -1.4142135623730951)

julia> get_offdiagonal(G, starting_address(G), 4)
(BoseFS{3,3}(2, 0, 1), -0.014142135623730952)
```

# Observables

To calculate observables, pass the transformed Hamiltonian `G` to
[`AllOverlaps`](@ref) with keyword argument `transform=G`.
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
function GuidingVectorSampling(h; vector, eps=1e-2 * norm(vector, Inf))
    return GuidingVectorSampling(h, vector, eps)
end

starting_address(h::GuidingVectorSampling) = starting_address(h.hamiltonian)
LOStructure(::Type{<:GuidingVectorSampling{<:Any,<:Any,H}}) where {H} = _lo_str(LOStructure(H))

function LinearAlgebra.adjoint(h::GuidingVectorSampling{A,T,<:Any,D,E}) where {A,T,D,E}
    h_adj = h.hamiltonian'
    return GuidingVectorSampling{!A,T,typeof(h_adj),D,E}(h_adj, h.vector)
end

dimension(h::GuidingVectorSampling, addr) = dimension(h.hamiltonian, addr)

Base.getproperty(h::GuidingVectorSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::GuidingVectorSampling{<:Any,<:Any,<:Any,<:Any,E}, ::Val{:eps}) where E = E
Base.getproperty(h::GuidingVectorSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)
Base.getproperty(h::GuidingVectorSampling, ::Val{:vector}) = getfield(h, :vector)

# Forward some interface functions.
num_offdiagonals(h::GuidingVectorSampling, add) = num_offdiagonals(h.hamiltonian, add)
diagonal_element(h::GuidingVectorSampling, add) = diagonal_element(h.hamiltonian, add)

_apply_eps(x, eps) = ifelse(iszero(x), eps, ifelse(abs(x) < eps, sign(x) * eps, x))

function guided_vector_modify(value, is_adjoint, eps, guide1, guide2)
    if iszero(guide1) && iszero(guide2)
        return value
    else
        guide1 = _apply_eps(guide1, eps)
        guide2 = _apply_eps(guide2, eps)
        if is_adjoint
            return value * (guide1 / guide2)
        else
            return value * (guide2 / guide1)
        end
    end
end

function get_offdiagonal(h::GuidingVectorSampling{A}, add1, chosen) where A
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    guide1 = h.vector[add1]
    guide2 = h.vector[add2]
    return add2, guided_vector_modify(matrix_element, A, h.eps, guide1, guide2)
end

struct GuidingVectorOffdiagonals{
    F,T,A,E,H<:AbstractHamiltonian,D,N<:AbstractOffdiagonals{F,T}
}<:AbstractOffdiagonals{F,T}
    hamiltonian::H
    vector::D
    guide::T
    offdiagonals::N
end

function offdiagonals(h::GuidingVectorSampling{A,T,H,D,E}, a) where {A,T,H,D,E}
    hps = offdiagonals(h.hamiltonian, a)
    guide = h.vector[a]
    return GuidingVectorOffdiagonals{typeof(a),T,A,E,H,D,typeof(hps)}(
        h.hamiltonian, h.vector, guide, hps
    )
end

function Base.getindex(h::GuidingVectorOffdiagonals{F,T,A,E}, i)::Tuple{F,T} where {F,T,A,E}
    add2, matrix_element = h.offdiagonals[i]
    guide1 = h.guide
    guide2 = h.vector[add2]
    return add2, guided_vector_modify(matrix_element, A, E, guide1, guide2)
end

Base.size(h::GuidingVectorOffdiagonals) = size(h.offdiagonals)

"""
    TransformUndoer(k::GuidingVectorSampling, op::AbstractOperator)
    TransformUndoer(k::GuidingVectorSampling)

For a guiding vector similarity transformation ``\\hat{G} = f \\hat{H} f^{-1}``
define the operator ``f^{-1} \\hat{A} f^{-1}``, and special case ``f^{-2}``, in order
to calculate observables. Here ``f`` is a diagonal operator whose entries are
the components of the guiding vector, i.e.``f_{ii} = v_i``.

See [`AllOverlaps`](@ref), [`GuidingVectorSampling`](@ref).
"""
function TransformUndoer(k::GuidingVectorSampling, op::Union{Nothing,AbstractOperator})
    if isnothing(op)
        T = eltype(k)
    else
        T = promote_type(eltype(k), eltype(op))
    end
    return TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

# methods for general operator `f^{-1} A f^{-1}`
LOStructure(::Type{<:TransformUndoer{<:Any,<:GuidingVectorSampling,A}}) where {A} = LOStructure(A)

function LinearAlgebra.adjoint(s::TransformUndoer{T,<:GuidingVectorSampling,<:AbstractOperator}) where {T}
    a_adj = adjoint(s.op)
    return TransformUndoer{T,typeof(s.transform),typeof(a_adj)}(s.transform, a_adj)
end

function diagonal_element(s::TransformUndoer{<:Any,<:GuidingVectorSampling,<:AbstractOperator}, add)
    guide = s.transform.vector[add]
    diagA = diagonal_element(s.op, add)
    return guided_vector_modify(diagA, true, s.transform.eps, 1., 2 * guide)
end

function num_offdiagonals(s::TransformUndoer{<:Any,<:GuidingVectorSampling,<:Any}, add)
    return num_offdiagonals(s.op, add)
end

function get_offdiagonal(s::TransformUndoer{<:Any,<:GuidingVectorSampling,<:Any}, add1, chosen)
    add2, offd = get_offdiagonal(s.op, add1, chosen)
    # Guiding vector `v` is represented as a diagonal operator `f`
    guide1 = s.transform.vector[add1]
    guide2 = s.transform.vector[add2]
    return add2, guided_vector_modify(offd, true, s.transform.eps, 1., guide1 + guide2)
end

# methods for special case `f^{-2}`
LOStructure(::Type{<:TransformUndoer{<:Any,<:GuidingVectorSampling,Nothing}}) = IsDiagonal()

function diagonal_element(s::TransformUndoer{<:Any,<:GuidingVectorSampling,Nothing}, add)
    guide = s.transform.vector[add]
    return guided_vector_modify(1., true, s.transform.eps, 1., 2 * guide)
end

num_offdiagonals(s::TransformUndoer{<:Any,<:GuidingVectorSampling,Nothing}, add) = 0
