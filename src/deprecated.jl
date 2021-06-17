export EveryTimeStep, EveryKthStep

function EveryTimeStep()
    @warn "`EveryTimeStep` is deprecated. Use `ReportDFAndInfo`."
    return ReportDFAndInfo()
end
function EveryKthStep(; k=10)
    @warn "`EveryTimeStep` is deprecated. Use `ReportDFAndInfo`."
    return ReportDFAndInfo(; k)
end
