using Documenter
using Rimu
using Rimu.ConsistentRNG
using Rimu.BitStringAddresses
using Rimu.StatsTools
using Literate

EXAMPLES_INPUT = joinpath(@__DIR__, "../scripts")
EXAMPLES_OUTPUT = joinpath(@__DIR__, "src/generated")
EXAMPLES_FILES = filter(endswith(".jl"), readdir(EXAMPLES_INPUT))
EXAMPLES_PAIRS = Pair{String,String}[]
EXAMPLES_NUMS = Int[]

for fn in EXAMPLES_FILES
    fnmd_full = Literate.markdown(joinpath(EXAMPLES_INPUT, fn), EXAMPLES_OUTPUT; documenter=true)
    header = split(filter(
                    contains(r"# Example [0-9]+:"), 
                    readlines(open(fnmd_full))
                )[1], ":")
    push!(EXAMPLES_NUMS, parse(Int, match(r"[0-9]+", header[1]).match))
    margintitle = lstrip(header[2])
    fnmd = fn[1:end-2]*"md"     # full path does not work
    push!(EXAMPLES_PAIRS, margintitle => joinpath("generated", fnmd))
end

makedocs(;
    modules=[Rimu,Rimu.ConsistentRNG,Rimu.RimuIO],
    format=Documenter.HTML(prettyurls = false),
    pages=[
        "Guide" => "index.md",
        "Examples" => EXAMPLES_PAIRS[sortperm(EXAMPLES_NUMS)],
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
