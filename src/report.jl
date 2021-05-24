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

function report!(_::Integer, report, key, value)
    data = report.data
    if haskey(data, key)
        column = data[key]::Vector{typeof(value)}
        push!(column, value)
    else
        data[key] = [value]
    end
    return report
end
function report!(step::Integer, report, key, value, postfix)
    report!(step, report, Symbol(key, postfix), value)
end
function report!(step::Integer, report, keys::Tuple, vals, postfix="")
    for (k, v) in zip(keys, vals)
        report!(step, report, k, v, postfix)
    end
    return report
end
function report!(step::Integer, report, kvpairs::NamedTuple, postfix="")
    for (k, v) in pairs(kvpairs)
        report!(step, report, k, v, postfix)
    end
    return report
end
function report!(::Integer, report, ::NamedTuple{(),Tuple{}}, args...)
    return report
end
function DataFrames.DataFrame(report::Report)
    DataFrame(report.data)
end
