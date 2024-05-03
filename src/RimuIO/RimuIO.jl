"""
Module to provide file input and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save_df(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load_df(filename)`](@ref) Load Arrow file into dataframe.
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


export save_df, load_df

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

end # module RimuIO
