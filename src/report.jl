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
function Base.empty!(report::Report)
    foreach(empty!, values(report.data))
end
function DataFrames.DataFrame(report::Report)
    DataFrame(report.data; copycols=false)
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
