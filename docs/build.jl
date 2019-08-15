using Documenter, Rimu
using Rimu.ConsistentRNG, Rimu.BitStringAddresses
using Rimu.FastBufs

makedocs(;
    modules=[Rimu],
    format=Documenter.HTML(prettyurls = false),
    pages=[
        "Home" => "index.md",
        "Developer documentation" => [
            "Hamiltonians" => "hamiltonians.md",
            "Random Numbers" => "consistentrng.md",
            "Documentation generation" => "documentation.md",
            "Code testing" => "testing.md",
        ],
        "API" => "API.md",
    ],
    repo="https://bitbucket.org/joachimbrand/Rimu.jl/src/{commit}{path}#L{line}",
    sitename="Rimu.jl",
    authors="Joachim Brand <j.brand@massey.ac.nz>",
)
