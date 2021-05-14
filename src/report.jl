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

function report!(report, key, value, postfix)
    report!(report, Symbol(key, postfix), value)
end
function report!(report, key, value)
    data = report.data
    if haskey(data, key)
        #column = data[key]
        #while length(column) < report.nrow[] - 1
        #    push!(column, zero(value))
        #end
        #push!(column, value)
        #report.nrow[] = max(report.nrow[], length(column))
        push!(data[key], value)
    else
        #column = fill(zero(value), max(report.nrow[] - 1, 0))
        #push!(column, value)
        #data[key] = column
        data[key] = [value]
    end
    return report
end

function report!(report, keys::Tuple, vals, postfix="")
    for (k, v) in zip(keys, vals)
        report!(report, k, v, postfix)
    end
    return report
end

function DataFrame(report::Report)
    DataFrame(report.data)
end
