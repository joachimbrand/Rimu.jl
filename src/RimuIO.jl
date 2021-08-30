"""
Module to provide file inut and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save_df(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load_df(filename)`](@ref) Load Arrow file into dataframe.
* [`RimuIO.save_dvec(filename, dv)`](@ref) Save dict vector in BSON format.
* [`RimuIO.load_dvec(filename)`](@ref) Load BSON file into dict vector.
"""
module RimuIO

using ..DictVectors
using ..Interfaces

using Arrow, DataFrames, BSON

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
    RimuIO.save_dvec(filename, dvec)
Save `dvec` in [BSON](https://github.com/JuliaIO/BSON.jl) format.

## Notes

* Only the [`localpart`](@ref) is saved. You may need to re-wrap the result in
  [`MPIData`](@ref) if using MPI.

* When using this function with MPI, make sure to save the vectors from different ranks to
  different files, e.g. by saving as `RimuIO.save_dvec("filename-\$(mpi_rank()).bson", dvec)`.

"""
save_dvec(filename, dvec) = bson(filename, Dict(:dvec => localpart(dvec)))

"""
    RimuIO.load_dvec(filename) -> AbstractDVec
Load `AbstractDVec` stored in [BSON](https://github.com/JuliaIO/BSON.jl).
"""
load_dvec(filename) = BSON.load(filename)[:dvec]

end # module RimuIO
