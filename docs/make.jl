using Documenter
using Rimu
using Rimu.BitStringAddresses
using Rimu.StatsTools
using Literate

EXAMPLES_INPUT = joinpath(@__DIR__, "../scripts")
EXAMPLES_OUTPUT = joinpath(@__DIR__, "src/generated")
EXAMPLES_FILES = filter(endswith(".jl"), readdir(EXAMPLES_INPUT))
EXAMPLES_PAIRS = Pair{String,String}[]
EXAMPLES_NUMS = Int[]

"""
    parse_header(filename)

Extract the title and example number from an example `.jl` script `filename`.
Assumes the header has format `# # Example N: Title`, else returns a default.
"""
function parse_header(filename::String)
    try
        header = split(filter(
                    contains(r"# Example [0-9]+:"),
                    readlines(open(filename))
                )[1], ":")
        n = parse(Int, match(r"[0-9]+", header[1]).match)
        t = lstrip(header[2])
        return n, t
    catch
        @warn "couldn't parse example file for title in \"" * filename * "\""
        return 100, "BAD_EXAMPLE"
    end
end

for fn in EXAMPLES_FILES
    fnmd_full = Literate.markdown(
        joinpath(EXAMPLES_INPUT, fn), EXAMPLES_OUTPUT;
        documenter = true, execute = true
        )
    ex_num, margintitle = parse_header(fnmd_full)
    push!(EXAMPLES_NUMS, ex_num)
    fnmd = fn[1:end-2]*"md"     # full path does not work
    push!(EXAMPLES_PAIRS, margintitle => joinpath("generated", fnmd))
end

makedocs(;
    modules=[Rimu,Rimu.RimuIO],
    format=Documenter.HTML(
        prettyurls = false,
        size_threshold=700_000, # 700 kB
        size_threshold_warn=200_000, # 200 kB
    ),
    pages=[
        "Guide" => "index.md",
        "Examples" => EXAMPLES_PAIRS[sortperm(EXAMPLES_NUMS)],
        "User documentation" => [
            "Exact Diagonalization" => "exactdiagonalization.md",
            "Projector Monte Carlo" => "projectormontecarlo.md",
            "StatsTools" => "statstools.md",
            "Using MPI" => "mpi.md",
        ],
        "Developer documentation" => [
            "Interfaces" => "interfaces.md",
            "Hamiltonians" => "hamiltonians.md",
            "Dict vectors" => "dictvectors.md",
            "BitString addresses" => "addresses.md",
            "Stochastic styles" => "stochasticstyles.md",
            "I/O" => "rimuio.md",
            "Random numbers" => "randomnumbers.md",
            "Documentation generation" => "documentation.md",
            "Code testing" => "testing.md",
        ],
        "API" => "API.md",
    ],
    sitename="Rimu.jl",
    authors="Joachim Brand <j.brand@massey.ac.nz>",
    checkdocs=:exports,
    doctest=false, # Doctests are done while testing.
    # warnonly = true, # should be diabled for a release
)

deploydocs(
    repo = "github.com/joachimbrand/Rimu.jl.git",
    push_preview = true,
)

# cleanup
foreach(fn -> rm(fn, force=true), filter(endswith(".arrow"), readdir(EXAMPLES_OUTPUT, join=true)))
foreach(fn -> rm(fn, force=true), filter(endswith(".arrow"), readdir(joinpath(@__DIR__, "build/generated"), join=true)))
