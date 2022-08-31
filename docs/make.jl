using Documenter
using Rimu
using Rimu.ConsistentRNG
using Rimu.BitStringAddresses
using Rimu.StatsTools
using Literate

EXAMPLES_INPUT = joinpath(@__DIR__, "../scripts")
EXAMPLES_OUTPUT = joinpath(@__DIR__, "src/generated")

for fn in filter(endswith(".jl"), readdir(EXAMPLES_INPUT))
    Literate.markdown(
        joinpath(EXAMPLES_INPUT, fn), EXAMPLES_OUTPUT; documenter=true
    )
end

makedocs(;
    modules=[Rimu,Rimu.ConsistentRNG,Rimu.RimuIO],
    format=Documenter.HTML(prettyurls = false),
    pages=[
        "Guide" => "index.md",
        "Examples" => [
            "1D Bose-Hubbard Model" => "generated/BHM-example.md",
            "Using MPI" => "generated/BHM-example-mpi.md",
            "Observables: G_2" => "generated/G2-example.md",
        ],
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
