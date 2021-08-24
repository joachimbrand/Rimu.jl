using Documenter
using Rimu

DocMeta.setdocmeta!(
    Rimu, :DocTestSetup, :(using Rimu; using Rimu.RMPI); recursive=true
)
# Run with fix=true to fix docstrings
doctest(Rimu)
