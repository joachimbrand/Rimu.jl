using Rimu
using BenchmarkTools

const SUITE = BenchmarkGroup()
const SKIP = ["bench_micro.jl"]

for file in readdir(@__DIR__)
    if file in SKIP
        @warn "skipping $file"
    elseif startswith(file, "bench_") && endswith(file, ".jl")
        SUITE[file[(length("bench_") + 1):(end - length(".jl"))]] = include(file)
    end
end
