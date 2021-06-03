using OrderedCollections # for LittleDict

"""
    Report

Internal structure that hold the temporary reported values. See [`report!`](@ref).
"""
struct Report
    data::LittleDict{Symbol,Vector}

    Report() = new(Dict{Symbol,Vector}())
end

function Base.show(io::IO, report::Report)
    print(io, "Report")
    if !isempty(report.data)
        print(":")
        keywidth = maximum(length.(string.(keys(report.data))))
        for (k, v) in report.data
            print("\n  $(lpad(k, keywidth)) => $v")
        end
    end
end

"""
    report!(report, keys, values, id="")
    report!(report, nt, id="")

Write `keys`, `values` pairs to `report` that will be converted to a `DataFrame` later.
Alternatively, a named tuple can be passed instead of `keys` and `values`.

The value of `id` is appended to the name of the column, e.g.
`report!(report, :key, value, :_1)` will report `value` to a column named `:key_1`.
"""
function report!(report, key, value)
    data = report.data
    if haskey(data, key)
        column = data[key]::Vector{typeof(value)}
        push!(column, value)
    else
        data[key] = [value]
    end
    return report
end
function report!(report, key, value, postfix)
    report!(report, Symbol(key, postfix), value)
end
function report!(report, keys::Tuple, vals, postfix="")
    for (k, v) in zip(keys, vals)
        report!(report, k, v, postfix)
    end
    return report
end
function report!(report, kvpairs::NamedTuple, postfix="")
    for (k, v) in pairs(kvpairs)
        report!(report, k, v, postfix)
    end
    return report
end
function report!(::Integer, report, ::NamedTuple{(),Tuple{}}, args...)
    return report
end
function DataFrames.DataFrame(report::Report)
    DataFrame(report.data; copycols=false)
end

function Base.empty!(report::Report)
    foreach(empty!, values(report.data))
end
