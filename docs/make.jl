using Documenter
using Rimu
using Rimu.ConsistentRNG
using Rimu.BitStringAddresses
using Rimu.StatsTools
using Literate

EXAMPLES_OUTPUT = joinpath(@__DIR__, "src/generated")

Literate.markdown(
    joinpath(@__DIR__, "../scripts/BHM-example.jl"), EXAMPLES_OUTPUT; documenter=true
)
Literate.markdown(
    joinpath(@__DIR__, "../scripts/BHM-example-mpi.jl"), EXAMPLES_OUTPUT; documenter=true
)

makedocs(;
    modules=[Rimu,Rimu.ConsistentRNG],
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
            "Documentation generation" => "documentation.md",
            "Code testing" => "testing.md",
        ],
        "API" => "API.md",
    ],
    sitename="Rimu.jl",
    authors="Joachim Brand <j.brand@massey.ac.nz>",
    doctest=false # Doctests are done while testing.
)

deploydocs(
    repo = "github.com/joachimbrand/Rimu.jl.git",
    push_preview = true,
)
