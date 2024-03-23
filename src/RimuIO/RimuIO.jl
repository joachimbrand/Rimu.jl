"""
Module to provide file input and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save_df(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load_df(filename)`](@ref) Load Arrow file into dataframe.
* [`RimuIO.save_dvec(filename, dv)`](@ref) Save dict vector in BSON format.
* [`RimuIO.load_dvec(filename)`](@ref) Load BSON file into dict vector.
"""
module RimuIO

using Arrow: Arrow, ArrowTypes
using BSON: BSON, bson
using DataFrames: DataFrames, DataFrame, metadata!
using StaticArrays: StaticArrays, SVector

using Rimu.BitStringAddresses: BitStringAddresses, BitString, BoseFS,
    CompositeFS, FermiFS, SortedParticleList,
    num_modes, num_particles
using Rimu.DictVectors: DictVectors
using Rimu.Interfaces: Interfaces, localpart, storage


export save_df, load_df, save_dvec, load_dvec

include("arrowtypes.jl")

"""
    RimuIO.save_df(filename, df::DataFrame; kwargs...)
Save dataframe in Arrow format.

Keyword arguments are passed on to
[`Arrow.write`](https://arrow.apache.org/julia/dev/reference/#Arrow.write). Compression is
enabled by default for large `DataFrame`s (over 10,000 rows).

Table-level metadata of the `DataFrame` is saved as Arrow metadata (with `String` value)
unless overwritten with the keyword argument `metadata`.

See also [`RimuIO.load_df`](@ref).
"""
function save_df(
    filename, df::DataFrame;
    compress = size(df)[1]>10_000 ? :zstd : nothing,
    metadata = nothing,
    kwargs...
)
    if metadata === nothing
        metadata = [key => string(val) for (key, val) in DataFrames.metadata(df)]
    end
    Arrow.write(filename, df; compress, metadata, kwargs...)
end

"""
    RimuIO.load_df(filename; propagate_metadata = true, add_filename = true) -> DataFrame
Load Arrow file into `DataFrame`. Optionally propagate metadata to `DataFrame` and
add the file name as metadata.

See also [`RimuIO.save_df`](@ref).
"""
function load_df(filename; propagate_metadata = true, add_filename = true)
    table = Arrow.Table(filename)
    df = DataFrame(table)
    if propagate_metadata
        meta_data = Arrow.getmetadata(table)
        isnothing(meta_data) || for (key, val) in meta_data
            metadata!(df, key, val)
        end
    end
    add_filename && metadata!(df, "filename", filename) # add filename as metadata
    return df
end

"""
    RimuIO.save_dvec(filename, dvec)
Save `dvec` in [BSON](https://github.com/JuliaIO/BSON.jl) format.

## Notes

* Only the [`localpart`](@ref) is saved. You may need to re-wrap the result in
  [`MPIData`](@ref Main.Rimu.RMPI.MPIData) if using MPI.
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
