using Documenter
using Rimu
using Rimu.ConsistentRNG
using Rimu.BitStringAddresses
using Rimu.StatsTools
using Literate

EXAMPLES_INPUT = joinpath(@__DIR__, "../scripts")
EXAMPLES_OUTPUT = joinpath(@__DIR__, "src/generated")
EXAMPLES_PAIRS = Pair{String,String}[]

for fn in filter(endswith(".jl"), readdir(EXAMPLES_INPUT))
    fnmd_full = Literate.markdown(
        joinpath(EXAMPLES_INPUT, fn), EXAMPLES_OUTPUT; documenter=true
    )
    title = lstrip(split(filter(contains("Example:"), readlines(open(fnmd_full)))[1], ":")[2])
    fnmd = fn[1:end-2]*"md"
    push!(EXAMPLES_PAIRS, title => joinpath("generated", fnmd))
end

makedocs(;
    modules=[Rimu,Rimu.ConsistentRNG,Rimu.RimuIO],
    format=Documenter.HTML(prettyurls = false),
    pages=[
        "Guide" => "index.md",
        "Examples" => EXAMPLES_PAIRS,
        "User documentation" => [
            "StatsTools" => "statstools.md",
        ],
        "Developer documentation" => [
            "Interfaces" => "interfaces.md",
            "Hamiltonians" => "hamiltonians.md",
            "Dict vectors" => "dictvectors.md",
            "BitString addresses" => "addresses.md",
            "Stochastic styles" => "stochasticstyles.md",
            "RMPI" => "RMPI.md",
            "Random Numbers" => "consistentrng.md",
            "I/O" => "rimuio.md",
            "Documentation generation" => "documentation.md",
            "Code testing" => "testing.md",
        ],
        "API" => "API.md",
    ],
    sitename="Rimu.jl",
    authors="Joachim Brand <j.brand@massey.ac.nz>",
    checkdocs=:exports,
    doctest=false # Doctests are done while testing.
)

deploydocs(
    repo = "github.com/joachimbrand/Rimu.jl.git",
    push_preview = true,
)
