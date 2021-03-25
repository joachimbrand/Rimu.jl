struct GuidingVectorSampling{A,T,H<:AbstractHamiltonian{T},D,G} <: AbstractHamiltonian{T}
    hamiltonian::H
    vector::D
end

function GuidingVectorSampling(h, v::AbstractDVec, g)
    return GuidingVectorSampling{false,eltype(h),typeof(h),typeof(v),g}(h, v)
end
function GuidingVectorSampling(h; v, g)
    return GuidedVectorSampling(h, v, g)
end

starting_address(h::GuidingVectorSampling) = starting_address(h.hamiltonian)
LOStructure(::Type{<:GuidingVectorSampling{<:Any,<:Any,H}}) where {H} = _lo_str(LOStructure(H))

function LinearAlgebra.adjoint(h::GuidingVectorSampling{A,T,<:Any,D,G}) where {A,T,D,G}
    h_adj = h.hamiltonian'
    return GuidingVectorSampling{!A,T,typeof(h_adj),D,G}(h_adj, h.vector)
end

Base.getproperty(h::GuidingVectorSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::GuidingVectorSampling{<:Any,<:Any,<:Any,<:Any,G}, ::Val{:g}) where G = G
Base.getproperty(h::GuidingVectorSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)
Base.getproperty(h::GuidingVectorSampling, ::Val{:vector}) = getfield(h, :vector)

# Forward some interface functions.
num_offdiagonals(h::GuidingVectorSampling, add) = num_offdiagonals(h.hamiltonian, add)
diagonal_element(h::GuidingVectorSampling, add) = diagonal_element(h.hamiltonian, add)

function guided_modify_element(value, guide1, guide2, g)
    if iszero(guide1) && iszero(guide2)
        return value
    else
        guide1 = abs(guide1) < g ? g : guide1
        guide2 = abs(guide2) < g ? g : guide2
        return value * guide1/guide2
    end
end

function get_offdiagonal(h::GuidingVectorSampling, add, chosen)
    return get_offdiagonal(h, add, chosen, h.vector[add])
end
function get_offdiagonal(h::GuidingVectorSampling{A}, add1, chosen, guide1) where A
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    guide2 = h.vector[add2]

    if A # adjoint
        return add2, guided_modify_element(matrix_element, guide2, guide1, h.g)
    else
        return add2, guided_modify_element(matrix_element, guide1, guide2, h.g)
    end
end

struct GuidingVectorOffdiagonals{
    A,G,F,T,H<:AbstractHamiltonian,D,N<:AbstractOffdiagonals{F,T}
}<:AbstractOffdiagonals{F,T}
    hamiltonian::H
    vector::D
    guide::T
    offdiagonals::N
end

function offdiagonals(h::GuidingVectorSampling{A,T,H,D,G}, a) where {A,T,H,D,G}
    hps = offdiagonals(h.hamiltonian, a)
    guide = h.vector[a]
    return GuidingVectorOffdiagonals{A,G,typeof(a),T,H,D,typeof(hps)}(
        h.hamiltonian, h.vector, guide, hps
    )
end

function Base.getindex(h::GuidingVectorOffdiagonals{A,G}, i) where {A,G}
    add2, matrix_element = h.offdiagonals[i]
    guide1 = h.guide
    guide2 = h.vector[add2]

    if A # adjoint
        return add2, guided_modify_element(matrix_element, guide2, guide1, G)
    else
        return add2, guided_modify_element(matrix_element, guide1, guide2, G)
    end
end

Base.size(h::GuidingVectorOffdiagonals) = size(h.offdiagonals)
