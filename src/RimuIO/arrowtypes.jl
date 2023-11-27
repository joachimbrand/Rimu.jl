const COMPLEX = Symbol("JuliaLang.Complex")
ArrowTypes.arrowname(::Type{<:Complex}) = COMPLEX
ArrowTypes.ArrowType(::Type{Complex{T}}) where {T} = Tuple{T,T}
ArrowTypes.JuliaType(::Val{COMPLEX}, ::Type{Tuple{T,T}}) where {T} = Complex{T}
ArrowTypes.toarrow(a::Complex) = (a.re, a.im)
ArrowTypes.fromarrow(::Type{Complex{T}}, t::Tuple{T,T}) where {T} = Complex{T}(t...)

###
### Bitstrings
###
const BITSTRING = Symbol("Rimu.BitString")
ArrowTypes.arrowname(::Type{<:BitString}) = BITSTRING
ArrowTypes.ArrowType(::Type{<:BitString{<:Any,N,T}}) where {N,T} = NTuple{N,T}
ArrowTypes.toarrow(b::BitString) = Tuple(b.chunks)
ArrowTypes.arrowmetadata(::Type{<:BitString{B}}) where {B} = string(B)
function ArrowTypes.JuliaType(::Val{BITSTRING}, ::Type{NTuple{N,T}}, meta) where {N,T}
    return BitString{parse(Int, meta),N,T}
end
function ArrowTypes.fromarrow(::Type{T}, chunks) where {T<:BitString}
    return T(SVector(chunks))
end

###
### Sorted particle lists
###
const SORTEDPARTICLELIST = Symbol("Rimu.SortedParticleList")
ArrowTypes.arrowname(::Type{<:SortedParticleList}) = SORTEDPARTICLELIST
ArrowTypes.ArrowType(::Type{<:SortedParticleList{N,<:Any,T}}) where {N,T} = NTuple{N,T}
ArrowTypes.toarrow(s::SortedParticleList) = Tuple(s.storage)
ArrowTypes.arrowmetadata(::Type{<:SortedParticleList{<:Any,M}}) where {M} = string(M)
function ArrowTypes.JuliaType(
    ::Val{SORTEDPARTICLELIST}, ::Type{NTuple{N,T}}, meta
) where {N,T}
    return SortedParticleList{N,parse(Int, meta),T}
end
function ArrowTypes.fromarrow(::Type{T}, storage) where {T<:SortedParticleList}
    return T(SVector(storage))
end

###
### Single-component
###
const BoseOrFermiFS{N,M,B} = Union{BoseFS{N,M,B},FermiFS{N,M,B}}

function ArrowTypes.ArrowType(::Type{<:BoseOrFermiFS{<:Any,<:Any,B}}) where {B}
    return ArrowTypes.ArrowType(B)
end
function ArrowTypes.toarrow(b::BoseOrFermiFS)
    return ArrowTypes.toarrow(b.bs)
end
function ArrowTypes.arrowmetadata(::Type{<:BoseOrFermiFS{N,M,B}}) where {N,M,B}
    return join((N, M, ArrowTypes.arrowmetadata(B)), '.')
end
function ArrowTypes.fromarrow(::Type{T}, bs) where {B,T<:BoseOrFermiFS{<:Any,<:Any,B}}
    return T(B(SVector(bs)))
end

const BOSEFS_BS = Symbol("Rimu.BoseFS.BitString")
ArrowTypes.arrowname(::Type{<:BoseFS{<:Any,<:Any,<:BitString}}) = BOSEFS_BS
function ArrowTypes.JuliaType(::Val{BOSEFS_BS}, ::Type{NTuple{X,Y}}, meta) where {X,Y}
    N, M, B = split(meta, '.')

    BS = ArrowTypes.JuliaType(Val(BITSTRING), NTuple{X,Y}, B)
    return BoseFS{parse(Int, N),parse(Int, M),BS}
end

const BOSEFS_SPL = Symbol("Rimu.BoseFS.SortedParticleList")
ArrowTypes.arrowname(::Type{<:BoseFS{<:Any,<:Any,<:SortedParticleList}}) = BOSEFS_SPL
function ArrowTypes.JuliaType(::Val{BOSEFS_SPL}, ::Type{NTuple{X,Y}}, meta) where {X,Y}
    N, M, B = split(meta, '.')

    BS = ArrowTypes.JuliaType(Val(SORTEDPARTICLELIST), NTuple{X,Y}, B)
    return BoseFS{parse(Int, N),parse(Int, M),BS}
end

const FERMIFS_BS = Symbol("Rimu.FermiFS.BitString")
ArrowTypes.arrowname(::Type{<:FermiFS{<:Any,<:Any,<:BitString}}) = FERMIFS_BS
function ArrowTypes.JuliaType(::Val{FERMIFS_BS}, ::Type{NTuple{X,Y}}, meta) where {X,Y}
    N, M, B = split(meta, '.')

    BS = ArrowTypes.JuliaType(Val(BITSTRING), NTuple{X,Y}, B)
    return FermiFS{parse(Int, N),parse(Int, M),BS}
end

const FERMIFS_SPL = Symbol("Rimu.FermiFS.SortedParticleList")
ArrowTypes.arrowname(::Type{<:FermiFS{<:Any,<:Any,<:SortedParticleList}}) = FERMIFS_SPL
function ArrowTypes.JuliaType(::Val{FERMIFS_SPL}, ::Type{NTuple{X,Y}}, meta) where {X,Y}
    N, M, B = split(meta, '.')

    BS = ArrowTypes.JuliaType(Val(SORTEDPARTICLELIST), NTuple{X,Y}, B)
    return FermiFS{parse(Int, N),parse(Int, M),BS}
end

###
### CompositeFS
###
const COMPOSITEFS = Symbol("Rimu.CompositeFS")
ArrowTypes.arrowname(::Type{<:CompositeFS}) = COMPOSITEFS
function ArrowTypes.ArrowType(::Type{<:CompositeFS{<:Any,<:Any,<:Any,T}}) where {T}
    # Going over T.parameters is ugly.
    return Tuple{map(ArrowTypes.ArrowType, T.parameters)...}
end
function ArrowTypes.toarrow(c::CompositeFS)
    return map(ArrowTypes.toarrow, c.components)
end
function ArrowTypes.arrowmetadata(::Type{<:CompositeFS{<:Any,<:Any,<:Any,T}}) where {T}
    metas = map(T.parameters) do X
        string(ArrowTypes.arrowname(X), ':', ArrowTypes.arrowmetadata(X))
    end
    return join(metas, ';')
end

# Because a tuple of tuples gets saved as a named tuple with names (1, 2, ...), we need to
# extract the tuple from the second parameter of the second argument.
function ArrowTypes.JuliaType(
    ::Val{COMPOSITEFS}, ::Type{<:NamedTuple{<:Any,T}}, meta,
) where {T}
    metas = split(meta, ';')

    comps = map(Tuple(T.parameters), metas) do X, m
        arrow_name, rest = split(m, ':')
        ArrowTypes.JuliaType(Val(Symbol(arrow_name)), X, rest)
    end

    return CompositeFS{
        length(comps),
        sum(num_particles, comps),
        num_modes(first(comps)),
        Tuple{comps...},
    }
end
function ArrowTypes.fromarrow(
    ::Type{C}, chunks...
) where {T,C<:CompositeFS{<:Any,<:Any,<:Any,T}}
    comps = map(ArrowTypes.fromarrow, Tuple(T.parameters), chunks)
    return C(comps)
end
