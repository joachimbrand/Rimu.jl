"""
    GutzwillerSampling(::AbstractHamiltonian; g)

Wrapper over any [`AbstractHamiltonian`](@ref) that implements Gutzwiller sampling. In this
importance sampling scheme the Hamiltonian is modified as follows
```math
\\tilde{H}_{ij} = H_{ij} e^{-g(H_{ii} - H_{jj})} .
```
This way off-diagonal spawns to higher-energy configurations are discouraged and
spawns to lower-energy configurations encouraged for positive `g`.

# Constructor

* `GutzwillerSampling(::AbstractHamiltonian, g)`
* `GutzwillerSampling(::AbstractHamiltonian; g)`

After construction, we can access the underlying Hamiltonian with `G.hamiltonian` and the
`g` parameter with `G.g`.

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS(1,1,1); u=6.0, t=1.0)
HubbardMom1D(fs"|1 1 1⟩"; u=6.0, t=1.0)

julia> G = GutzwillerSampling(H, g=0.3)
GutzwillerSampling(HubbardMom1D(fs"|1 1 1⟩"; u=6.0, t=1.0); g=0.3)

julia> get_offdiagonal(H, BoseFS(2, 1, 0), 1)
(BoseFS{3,3}(1, 0, 2), 2.0)

julia> get_offdiagonal(G, BoseFS(2, 1, 0), 1)
(BoseFS{3,3}(1, 0, 2), 0.8131393194811987)
```

# Observables

To calculate observables, pass the transformed Hamiltonian `G` to
[`AllOverlaps`](@ref) with keyword argument `transform=G`.
"""
struct GutzwillerSampling{A,T,H<:AbstractHamiltonian{T},G} <: AbstractHamiltonian{T}
    # The A parameter sets whether this is an adjoint or not.
    # The G parameter is g parameter value.
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

LOStructure(::Type{<:GutzwillerSampling{<:Any,<:Any,H}}) where {H} = _lo_str(LOStructure(H))
_lo_str(::LOStructure) = AdjointKnown()
_lo_str(::AdjointUnknown) = AdjointUnknown()

function LinearAlgebra.adjoint(h::GutzwillerSampling{A,T,<:Any,G}) where {A,T,G}
    h_adj = h.hamiltonian'
    return GutzwillerSampling{!A,T,typeof(h_adj),G}(h_adj)
end

dimension(h::GutzwillerSampling, addr) = dimension(h.hamiltonian, addr)

Base.getproperty(h::GutzwillerSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::GutzwillerSampling{<:Any,<:Any,<:Any,G}, ::Val{:g}) where G = G
Base.getproperty(h::GutzwillerSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)

function Base.:(==)(a::GutzwillerSampling{A}, b::GutzwillerSampling{B}) where {A,B}
    return A == B && a.g == b.g && a.hamiltonian == b.hamiltonian
end

# Forward some interface functions.
num_offdiagonals(h::GutzwillerSampling, add) = num_offdiagonals(h.hamiltonian, add)
diagonal_element(h::GutzwillerSampling, add) = diagonal_element(h.hamiltonian, add)

function gutzwiller_modify(matrix_element, is_adjoint, g, diag1, diag2)
    if is_adjoint
        return matrix_element * exp(-g * (diag1 - diag2))
    else
        return matrix_element * exp(-g * (diag2 - diag1))
    end
end

function get_offdiagonal(h::GutzwillerSampling{A}, add1, chosen) where A
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    diag1 = diagonal_element(h, add1)
    diag2 = diagonal_element(h, add2)
    return add2, gutzwiller_modify(matrix_element, A, h.g, diag1, diag2)
end

struct GutzwillerOffdiagonals{
    F,T,A,G,H<:AbstractHamiltonian,N<:AbstractOffdiagonals{F,T}
}<:AbstractOffdiagonals{F,T}
    hamiltonian::H
    diag::T
    offdiagonals::N
end

function offdiagonals(ham::GutzwillerSampling{A,<:Any,<:Any,G}, a) where {A,G}
    hps = offdiagonals(ham.hamiltonian, a)
    diag = diagonal_element(ham, a)
    return GutzwillerOffdiagonals{typeof(a),eltype(ham),A,G,typeof(ham),typeof(hps)}(
        ham, diag, hps
    )
end

function Base.getindex(h::GutzwillerOffdiagonals{F,T,A,G}, i)::Tuple{F,T} where {F,T,A,G}
    add2, matrix_element = h.offdiagonals[i]
    diag2 = diagonal_element(h.hamiltonian, add2)
    return add2, gutzwiller_modify(matrix_element, A, G, h.diag, diag2)
end

Base.size(h::GutzwillerOffdiagonals) = size(h.offdiagonals)

"""
    TransformUndoer(k::GutzwillerSampling, op::AbstractOperator)
    TransformUndoer(k::GutzwillerSampling)

For a Gutzwiller similarity transformation ``\\hat{G} = f \\hat{H} f^{-1}``
define the operator ``f^{-1} \\hat{A} f^{-1}``, and special case ``f^{-2}``, in order
to calculate observables. Here ``f`` is a diagonal operator whose entries are
``f_{ii} = e^{-g H_{ii}}``.

See [`AllOverlaps`](@ref), [`GutzwillerSampling`](@ref).
"""
function TransformUndoer(k::GutzwillerSampling, op::Union{Nothing,AbstractOperator})
    if isnothing(op)
        T = eltype(k)
    else
        T = typeof(zero(eltype(k)) * zero(eltype(op)))
    end
    return TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

# methods for general operator `f^{-1} A f^{-1}`
LOStructure(::Type{<:TransformUndoer{<:Any,<:GutzwillerSampling,A}}) where {A} = LOStructure(A)

function LinearAlgebra.adjoint(s::TransformUndoer{T,<:GutzwillerSampling,<:AbstractOperator}) where {T}
    a_adj = adjoint(s.op)
    return TransformUndoer{T,typeof(s.transform),typeof(a_adj)}(s.transform, a_adj)
end

function diagonal_element(s::TransformUndoer{<:Any,<:GutzwillerSampling,<:AbstractOperator}, add)
    diagH = diagonal_element(s.transform.hamiltonian, add)
    diagA = diagonal_element(s.op, add)
    return gutzwiller_modify(diagA, true, s.transform.g, 0., 2 * diagH)
end

function num_offdiagonals(s::TransformUndoer{<:Any,<:GutzwillerSampling,<:Any}, add)
    return num_offdiagonals(s.op, add)
end

function get_offdiagonal(s::TransformUndoer{<:Any,<:GutzwillerSampling,<:Any}, add, chosen)
    newadd, offd = get_offdiagonal(s.op, add, chosen)
    # Gutzwiller `f` operator is diagonal
    diagH1 = diagonal_element(s.transform.hamiltonian, add)
    diagH2 = diagonal_element(s.transform.hamiltonian, newadd)
    return newadd, gutzwiller_modify(offd, true, s.transform.g, 0., diagH1 + diagH2)
end

# methods for special case `f^{-2}`
LOStructure(::Type{<:TransformUndoer{<:Any,<:GutzwillerSampling,Nothing}}) = IsDiagonal()

function diagonal_element(s::TransformUndoer{<:Any,<:GutzwillerSampling,Nothing}, add)
    diagH = diagonal_element(s.transform.hamiltonian, add)
    return gutzwiller_modify(1., true, s.transform.g, 0., 2 * diagH)
end

num_offdiagonals(s::TransformUndoer{<:Any,<:GutzwillerSampling,Nothing}, add) = 0
