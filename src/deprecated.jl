function EveryTimeStep()
    @warn "`EveryTimeStep` is deprecated. Use `ReportDFAndInfo`."
    return ReportDFAndInfo(; k=1, i=1)
end
function EveryKthStep(; k=10)
    @warn "`EveryTimeStep` is deprecated. Use `ReportDFAndInfo`."
    return ReportDFAndInfo(; k)
end
function ReportDFAndInfo(; projector=nothing, hproj=nothing, kwargs...)
    if !isnothing(projector) || !isnothing(hproj)
        @warn "Reporting strategies no longer accept projectors. Please use `ProjectedEnergy`"
    end
    return ReportDFAndInfo(kwargs...)
end
function ReportToFile(; projector=nothing, hproj=nothing, kwargs...)
    if !isnothing(projector) || !isnothing(hproj)
        @warn "Reporting strategies no longer accept projectors. Please use `ProjectedEnergy`"
    end
    return ReportToFile(kwargs...)
end
