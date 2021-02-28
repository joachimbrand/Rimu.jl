"""
Module to provide file inut and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load(filename)`](@ref) Load Arrow file into dataframe.
"""
module RimuIO

using Arrow, DataFrames

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

end # module RimuIO
