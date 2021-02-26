"""
Module to provide file inut and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load(filename)`](@ref) Load Arrow file into dataframe.
"""
module RimuIO

using Arrow, DataFrames

function __init__()
    Arrow.ArrowTypes.registertype!(Complex{Int},Complex{Int})
    Arrow.ArrowTypes.registertype!(Complex{Float64},Complex{Float64})
end

"""
    RimuIO.save(filename, df::DataFrame)
Save dataframe in Arrow format.
"""
save_df(filename, df::DataFrame) = Arrow.write(filename, df)

"""
    RimuIO.load_df(filename) -> DataFrame
Load Arrow file into dataframe.
"""
load_df(filename) = DataFrame(Arrow.Table(filename))

end # module RimuIO
