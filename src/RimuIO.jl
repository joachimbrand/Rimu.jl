"""
Module to provide file inut and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load(filename)`](@ref) Load Arrow file into dataframe.
"""
module RimuIO

using ..DictVectors

using Arrow, DataFrames, JLSO

export save_df, load_df

const COMPLEX = Symbol("JuliaLang.Complex")
ArrowTypes.arrowname(::Type{<:Complex}) = COMPLEX
ArrowTypes.ArrowType(::Type{Complex{T}}) where {T} = Tuple{T,T}
ArrowTypes.JuliaType(::Val{COMPLEX}, ::Type{Tuple{T,T}}) where {T} = Complex{T}
ArrowTypes.toarrow(a::Complex) = (a.re, a.im)
ArrowTypes.fromarrow(::Type{Complex{T}}, t::Tuple{T,T}) where {T} = Complex{T}(t...)

"""
    RimuIO.save_df(filename, df::DataFrame)
Save dataframe in Arrow format.
"""
save_df(filename, df::DataFrame) = Arrow.write(filename, df)

"""
    RimuIO.load_df(filename) -> DataFrame
Load Arrow file into dataframe.
"""
load_df(filename) = DataFrame(Arrow.Table(filename))

"""
    compress(dvec::AbstractDVec)

"Compress" the `DVec` by throwing away the `Dict`.
"""
function compress(dvec::AbstractDVec)
    return empty(dvec), collect(pairs(storage(dvec)))
end

function save_dvec(filename, state)
    JLSO.save(filename, :dvecs => map(r -> compress(r.v), state.replicas))
    return filename
end

function save_dvec(filename, dvec::AbstractDVec)
    JLSO.save(filename, :dvecs => (compress(dvec),))
    return filename
end

function load_dvec(filename)
    d = JLSO.load(filename)
    result = map(d[:dvecs]) do d
        shell, pairs = d
        for (k, v) in pairs
            storage(shell)[k] = v
        end
        shell
    end
    if length(result) == 1
        return only(result)
    else
        return result
    end
end

end # module RimuIO
