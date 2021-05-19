"""
    Report

Internal structure that hold the temporary reported values. See [`report`](@ref).
"""
struct Report
    data::Dict{Symbol,Vector}
    nrow::Ref{Int}

    Report() = new(Dict{Symbol,Vector}(), Ref(0))
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
    # TODO: take care of adding stuff to the report after a few steps were already taken?
    if haskey(data, key)
        push!(data[key], value)
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
function DataFrames.DataFrame(report::Report)
    DataFrame(report.data)
end
