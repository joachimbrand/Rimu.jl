"""
    ReportingStrategy

Abstract type for strategies for reporting data in a DataFrame with [`report!()`](@ref). It
also affects the calculation and reporting of projected quantities in the DataFrame.

# Implemented strategies:

* [`EveryTimeStep`](@ref)
* [`EveryKthStep`](@ref)
* [`ReportDFAndInfo`](@ref)
* [`ReportToFile`](@ref)

# Interface

A `ReportingStrategy` can define any of the following:

* [`refine_r_strat`](@ref)
* [`report!`](@ref)
* [`report_after_step`](@ref)
* [`finalize_report!`](@ref)

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

This function is called exactly once at the very end of a step. It can let the
`ReportingStrategy` print some information to output.
"""
function report_after_step(::ReportingStrategy, args...)
    return nothing
end

"""
    finalize_report!(::ReportingStrategy, report)

Finalize the report. This function is called after all steps in [`lomc!`](@ref) have finished.
"""
function finalize_report!(::ReportingStrategy, report)
    DataFrame(report)
end

function print_stats(io::IO, step, state)
    print(io, "[ ", lpad(step, 11), " | ")
    shift = lpad(round(state.replicas[1].params.shift, digits=4), 10)
    norm = lpad(round(state.replicas[1].pnorm, digits=4), 10)
    println(io, "shift: ", shift, " | norm: ", norm)
    flush(io)
end

"""
    ReportDFAndInfo(; k=1, i=100, io=stdout, writeinfo=false) <: ReportingStrategy

The default [`ReportingStrategy`](@ref). Report every `k`th step to a `DataFrame` and write
info message to `io` every `i`th step (unless `writeinfo == false`). The flag `writeinfo` is
useful for controlling info messages in MPI codes, e.g. by setting
`writeinfo=is_mpi_root()`.
"""
@with_kw struct ReportDFAndInfo <: ReportingStrategy
    k::Int = 1
    i::Int = 100
    io::IO = stdout
    writeinfo::Bool = false
end
function report!(s::ReportDFAndInfo, step, args...)
    step % s.k == 0 && report!(args...)
    return nothing
end
function report_after_step(s::ReportDFAndInfo, step, _, state)
    if s.writeinfo && step % s.i == 0
        print_stats(s.io, step, state)
    end
end

"""
    ReportToFile(; kwargs...) <: ReportingStrategy

Reporting strategy that writes the report directly to a file. Useful when dealing with long
jobs or large numbers of replicas, when the report can incur a significant memory cost.

# Keyword arguments

* `filename`: the file to report to. If the file already exists, a new file is created.
* `chunk_size = 1000`: the size of each chunk that is written to the file.
* `save_if = true`: if this value is true, save the report, otherwise ignore it. Use
  `save_if=is_mpi_root()` when running MPI jobs.
* `return_df`: if this value is true, read the file and return the data frame at the end of
  computation. Otherwise, an empty `DataFrame` is returned.
* `io=stdout`: The `IO` to print messages to. Set to `devnull` if you don't want to see
  messages printed out.
"""
@with_kw struct ReportToFile <: ReportingStrategy
    filename::String
    chunk_size::Int = 1000
    save_if::Bool = true
    return_df::Bool = false
    io::IO = stdout
end
function refine_r_strat(s::ReportToFile)
    if s.save_if
        # If filename exists, add -1 to the end of it. If that exists as well,
        # increment the number after the dash
        new_filename = s.filename
        if isfile(new_filename)
            base, ext = splitext(new_filename)
            new_filename = string(base, "-", 1, ext)
        end
        while isfile(new_filename)
            base, ext = splitext(new_filename)
            m = match(r"(.*)-([0-9]+)$", base)
            if !isnothing(m)
                new_filename = string(m[1], "-", parse(Int, m[2]) + 1, ext)
            end
        end
        if s.filename ≠ new_filename
            println(s.io, "File `$(s.filename)` exists. Using `$(new_filename)`.")
            s = @set s.filename = new_filename
        else
            println(s.io, "Saving report to `$(s.filename)`.")
        end
    end
    return s
end
function report!(s::ReportToFile, _, args...)
    if s.save_if
        report!(args...)
    end
    return nothing
end
function report_after_step(s::ReportToFile, step, report, state)
    if s.save_if && step % s.chunk_size == 0
        # Report some stats:
        print_stats(s.io, step, state)

        if isfile(s.filename)
            Arrow.append(s.filename, report.data)
        else
            Arrow.write(s.filename, report.data; file=false)
        end
        empty!(report)
    end
end
function finalize_report!(s::ReportToFile, report)
    if s.save_if
        println(s.io, "Finalizing.")
        if isfile(s.filename)
            Arrow.append(s.filename, report.data)
        else
            Arrow.write(s.filename, report.data; file=false)
        end
        if s.return_df
            return DataFrame(Arrow.Table(s.filename))
        end
    end
    return DataFrame()
end
