using Documenter
using Rimu

DocMeta.setdocmeta!(
    Rimu,
    :DocTestSetup,
    :(using Rimu; using Rimu.StatsTools; using DataFrames; using Random;
      using LinearAlgebra);
    recursive=true
)
# Run with fix=true to fix docstrings
doctest(Rimu)
