using Documenter, Rimu

makedocs(;
    modules=[Rimu],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://bitbucket.org/joachimbrand/Rimu.jl/blob/{commit}{path}#L{line}",
    sitename="Rimu.jl",
    authors="Joachim Brand <j.brand@massey.ac.nz>",
    assets=String[],
)
