using Documenter
using Rimu

DocMeta.setdocmeta!(
    Rimu, :DocTestSetup, :(using Rimu; using Rimu.RMPI; using Rimu.StatsTools; using Rimu.StatsTools: to_nt; using DataFrames); recursive=true
)
# Run with fix=true to fix docstrings
doctest(Rimu)
