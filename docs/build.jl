using Documenter, Rimu
using Rimu.ConsistentRNG, Rimu.BitStringAddresses
using Rimu.FastBufs

# using Literate
#
# EXAMPLE = joinpath(@__DIR__,"..","scripts","qmcexample.jl")
# PLOTTING = joinpath(@__DIR__,"..","scripts","plotting.jl")
# OUTPUT = joinpath(@__DIR__, "src/generated/")
# mkpath(OUTPUT)
# cp(PLOTTING, joinpath(OUTPUT,"plotting.jl"), force = true)
# Literate.markdown(EXAMPLE, OUTPUT)
# Literate.notebook(EXAMPLE, OUTPUT)

makedocs(;
    modules=[Rimu,Rimu.ConsistentRNG],
    format=Documenter.HTML(prettyurls = false),
    pages=[
        "Guide" => "index.md",
        "Developer documentation" => [
            "Hamiltonians" => "hamiltonians.md",
            "Random Numbers" => "consistentrng.md",
            "Documentation generation" => "documentation.md",
            "Code testing" => "testing.md",
        ],
        # "Example" => "generated/qmcexample.md",
        "API" => "API.md",
    ],
    repo="https://bitbucket.org/joachimbrand/Rimu.jl/src/{commit}{path}#L{line}",
    sitename="Rimu.jl",
    authors="Joachim Brand <j.brand@massey.ac.nz>",
)
