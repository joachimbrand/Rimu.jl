module OnlineStatsExt

using Folds
using OnlineStats
using Rimu

function Rimu.post_step(stats::PerAddressOnlineStats, replica)
    Folds.foreach(eachindex(stats.basis)) do i
        k = stats.basis[i]
        acc = stats.stats[i]
        fit!(acc, replica.v[k])
    end
    return (;)
end
function OnlineStats.value(stats::PerAddressOnlineStats)
    vals = value.(stats.stats)

    return Dict(map(=>, stats.basis, Iterators.map(value, stats.stats)))
end

end
