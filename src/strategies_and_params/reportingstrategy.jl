"""
    Report()

Internal structure that holds the temporary reported values as well as metadata.

See [`report!`](@ref), [`report_metadata!`](@ref), [`get_metadata`](@ref).
"""
struct Report
    data::LittleDict{Symbol,Vector}
    meta::LittleDict{String,String} # `String`s are required for Arrow metadata

    Report() = new(LittleDict{Symbol,Vector}(), LittleDict{String,String}())
end

function Base.show(io::IO, report::Report)
    print(io, "Report")
    if !isempty(report.data)
        print(io, ":")
        keywidth = maximum(length.(string.(keys(report.data))))
        for (k, v) in report.data
            print(io, "\n  $(lpad(k, keywidth)) => $v")
        end
    end
end

Base.empty!(report::Report) = foreach(empty!, values(report.data)) # does not empty metadata
Base.isempty(report::Report) = all(isempty, values(report.data))

function DataFrames.DataFrame(report::Report)
    df = DataFrame(report.data; copycols=false)
    for (key, val) in get_metadata(report) # add metadata
        DataFrames.metadata!(df, key, val)
    end
    return df
end

"""
    report_metadata!(report::Report, key, value)
    report_metadata!(report::Report, kvpairs)

Set metadata `key` to `value` in `report`. `key` and `value` are converted to `String`s.
Alternatively, an iterable of key-value pairs or a `NamedTuple` can be passed.

Throws an error if `key` already exists.

See also [`get_metadata`](@ref), [`report!`](@ref), [`Report`](@ref).
"""
function report_metadata!(report::Report, key, value)
    key = string(key)
    haskey(report.meta, key) && throw(ArgumentError("duplicate metadata key: $key"))
    report.meta[key] = string(value)
    return report
end
function report_metadata!(report::Report, kvpairs)
    for (k, v) in kvpairs
        report_metadata!(report, k, v)
    end
    return report
end
function report_metadata!(report::Report, kvpairs::NamedTuple)
    return report_metadata!(report, pairs(kvpairs))
end

"""
    get_metadata(report::Report, key)

Get metadata `key` from `report`. `key` is converted to a `String`.

See also [`report_metadata!`](@ref), [`Report`](@ref), [`report!`](@ref).
"""
function get_metadata(report::Report, key)
    return report.meta[string(key)]
end
function get_metadata(report::Report)
    return report.meta
end

const SymbolOrString = Union{Symbol,AbstractString}

"""
    report!(report, keys, values, id="")
    report!(report, pairs, id="")

Write `keys`, `values` pairs to `report` that will be converted to a `DataFrame` later.
Alternatively, a named tuple or a collection of pairs can be passed instead of `keys` and
`values`.

The value of `id` is appended to the name of the column, e.g.
`report!(report, :key, value, :_1)` will report `value` to a column named `:key_1`.
"""
function report!(report::Report, key::SymbolOrString, value)
    data = report.data
    if haskey(data, key)
        column = data[key]::Vector{typeof(value)}
        push!(column, value)
    else
        data[key] = [value]
    end
    return report
end
function report!(report::Report, key::SymbolOrString, value, postfix::SymbolOrString)
    report!(report, Symbol(key, postfix), value)
end
function report!(report::Report, keys, vals, postfix::SymbolOrString="")
    @assert length(keys) == length(vals)
    for (k, v) in zip(keys, vals)
        report!(report, k, v, postfix)
    end
    return report
end
function report!(report::Report, nt::NamedTuple, postfix::SymbolOrString="")
    report!(report, pairs(nt), postfix)
    return report
end
function report!(report::Report, kvpairs, postfix::SymbolOrString="")
    for (k, v) in kvpairs
        report!(report, k, v, postfix)
    end
    return report
end

function ensure_correct_lengths(report::Report)
    lens = length.(values(report.data))
    for i in 1:length(report.data)
        lens[i] == lens[1] || throw(ArgumentError("duplicate keys reported to `DataFrame`"))
    end
end

"""
    ReportingStrategy

Abstract type for strategies for reporting data in a DataFrame with [`report!()`](@ref).

# Implemented strategies:

* [`ReportDFAndInfo`](@ref)
* [`ReportToFile`](@ref)

# Interface:

A `ReportingStrategy` can define any of the following:

* [`refine_r_strat`](@ref)
* [`report!`](@ref)
* [`report_after_step`](@ref)
* [`finalize_report!`](@ref)
* [`reporting_interval`](@ref)

"""
abstract type ReportingStrategy end

"""
    refine_r_strat(r_strat::ReportingStrategy) -> r_strat

Initialize the reporting strategy. This can be used to set up filenames or other attributes
that need to be unique for a run of FCIQMC.
"""
refine_r_strat(r_strat::ReportingStrategy) = r_strat

"""
     report!(::ReportingStrategy, step, report::Report, keys, values, id="")
     report!(::ReportingStrategy, step, report::Report, nt, id="")

Report `keys` and `values` to `report`, which will be converted to a `DataFrame` before
[`lomc!`](@ref) exits. Alternatively, a `nt::NamedTuple` can be passed in place of `keys`
and `values`. If `id` is specified, it is appended to all `keys`. This is used to
differentiate between values reported by different replicas.

To overload this function for a new `ReportingStrategy`, overload
`report!(::ReportingStrategy, step, args...)` and apply the report by calling
`report!(args...)`.
"""
function report!(::ReportingStrategy, _, args...)
    report!(args...)
    return nothing
end

"""
    report_after_step(::ReportingStrategy, step, report, state)

This function is called at the very end of a step, after [`reporting_interval`](@ref) steps.
For example, it can be used to print some information to `stdout`.
"""
function report_after_step(::ReportingStrategy, args...)
    return nothing
end

"""
    reporting_interval(::ReportingStrategy)

Get the interval between steps for which non-essential statistics are reported. Defaults to
1 if chosen [`ReportingStrategy`](@ref) does not specify an interval.
"""
reporting_interval(::ReportingStrategy) = 1

"""
    finalize_report!(::ReportingStrategy, report)

Finalize the report. This function is called after all steps in [`lomc!`](@ref) have
finished.
"""
finalize_report!(::ReportingStrategy, report) = DataFrame(report)

function print_stats(io::IO, step, state)
    print(io, "[ ", lpad(step, 11), " | ")
    shift = lpad(round(state.replicas[1].params.shift, digits=4), 10)
    norm = lpad(round(state.replicas[1].pnorm, digits=4), 10)
    println(io, "shift: ", shift, " | norm: ", norm)
    flush(io)
end

"""
    ReportDFAndInfo(; reporting_interval=1, info_interval=100, io=stdout, writeinfo=false) <: ReportingStrategy

The default [`ReportingStrategy`](@ref). Report every `reporting_interval`th step to a `DataFrame`
and write info message to `io` every `info_interval`th reported step (unless `writeinfo == false`). The flag
`writeinfo` is useful for controlling info messages in MPI codes, e.g. by setting
`writeinfo = `[`is_mpi_root()`](@ref Rimu.RMPI.is_mpi_root).
"""
@with_kw struct ReportDFAndInfo <: ReportingStrategy
    reporting_interval::Int = 1
    info_interval::Int = 100
    io::IO = stdout
    writeinfo::Bool = false
end
function report!(s::ReportDFAndInfo, _, args...)
    report!(args...)
end
function report_after_step(s::ReportDFAndInfo, step, _, state)
    if s.writeinfo && step % (s.info_interval * s.reporting_interval) == 0
        print_stats(s.io, step, state)
    end
end
function reporting_interval(s::ReportDFAndInfo)
    return s.reporting_interval
end

"""
    ReportToFile(; kwargs...) <: ReportingStrategy

[`ReportingStrategy`](@ref) that writes the report directly to a file in the
[`Arrow`](https://arrow.apache.org/julia/dev/) format. Useful when dealing with long
jobs or large numbers of replicas, when the report can incur a significant memory cost.

The arrow file can be read back in with [`load_df(filename)`](@ref) or
`using Arrow; Arrow.Table(filename)`.

# Keyword arguments

* `filename = "out.arrow"`: the file to report to. If the file already exists, a new file is
  created.
* `reporting_interval = 1`: interval between simulation steps that are reported.
* `chunk_size = 1000`: the size of each chunk that is written to the file. A `DataFrame` of
  this size is collected in memory and written to disk. When saving, an info message is also
  printed to `io`.
* `save_if = `[`is_mpi_root()`](@ref Rimu.RMPI.is_mpi_root): if this value is true, save the
  report, otherwise ignore it.
* `return_df = false`: if this value is true, read the file and return the data frame at the
  end of computation. Otherwise, an empty `DataFrame` is returned.
* `io = stdout`: The `IO` to print messages to. Set to `devnull` if you don't want to see
  messages printed out.
* `compress = :zstd`: compression algorithm to use. Can be `:zstd`, `:lz4` or `nothing`.

See also [`load_df`](@ref) and [`save_df`](@ref).
"""
mutable struct ReportToFile{C} <: ReportingStrategy
    filename::String
    reporting_interval::Int
    chunk_size::Int
    save_if::Bool
    return_df::Bool
    io::IO
    compress::C # Symbol, Arrow.ZstdCompressor, Arrow.LZ4FrameCompressor or nothing
    writer::Union{Arrow.Writer, Nothing}
end

function ReportToFile(;
    filename = "out.arrow",
    reporting_interval = 1,
    chunk_size = 1000,
    save_if = RMPI.is_mpi_root(),
    return_df = false,
    io = stdout,
    compress = :zstd,
)
    if !(compress isa Union{Nothing, Arrow.ZstdCompressor, Arrow.LZ4FrameCompressor})
        if !(compress == :zstd || compress == :lz4)
            throw(ArgumentError("compress must be nothing, :zstd or :lz4"))
        end
    end
    return ReportToFile(
        filename,
        reporting_interval,
        chunk_size,
        save_if,
        return_df,
        io,
        compress,
        nothing
    )
end

# helper function to check if the writer is open
_isopen(s::ReportToFile) = !isnothing(s.writer) && !s.writer.isclosed

function refine_r_strat(s::ReportToFile)
    if s.save_if
        # If filename exists, add -1 to the end of it. If that exists as well,
        # increment the number after the dash
        new_filename = s.filename
        while isfile(new_filename)
            base, ext = splitext(new_filename)
            m = match(r"(.*)-([0-9]+)$", base)
            if isnothing(m)
                new_filename = string(base, "-1", ext)
            else
                new_filename = string(m[1], "-", parse(Int, m[2]) + 1, ext)
            end
        end
        if s.filename ≠ new_filename
            println(s.io, "File `$(s.filename)` exists.")
            s.filename = new_filename
        end
        println(s.io, "Saving report to `$(s.filename)`.")
    end
    return s
end
function report!(s::ReportToFile, _, args...)
    if s.save_if
        report!(args...)
    end
end
function report_after_step(s::ReportToFile, step, report, state)
    if s.save_if && step % (s.chunk_size * s.reporting_interval) == 0
        # Report some stats:
        print_stats(s.io, step, state)

        if !_isopen(s)
            # If the writer is closed or absent, we need to create a new one
            s.writer = open(
                Arrow.Writer, s.filename;
                compress=s.compress, metadata=report.meta
            )
        end
        Arrow.write(s.writer, report.data)
        empty!(report)
    end
end
# We rely on this function to be called to close the writer.
function finalize_report!(s::ReportToFile, report)
    if s.save_if
        println(s.io, "Finalizing.")
        if !isempty(report)
            if !_isopen(s)
                # If the writer is closed or absent, we need to create a new one
                s.writer = open(
                    Arrow.Writer, s.filename;
                    compress=s.compress, metadata=report.meta
                )
            end
            Arrow.write(s.writer, report.data)
        end
        close(s.writer) # close the writer
        if s.return_df
            return RimuIO.load_df(s.filename)
        end
    end
    return DataFrame()
end
function reporting_interval(s::ReportToFile)
    return s.reporting_interval
end
