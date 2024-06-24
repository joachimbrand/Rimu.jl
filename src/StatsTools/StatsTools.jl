"""
Tools for the statistical analysis of Monte Carlo data.

### Exports:
- [`blocking_analysis`](@ref)
- [`blocking_analysis_data`](@ref)
- [`ratio_of_means`](@ref)
- [`growth_witness`](@ref)
- [`smoothen`](@ref)
- [`shift_estimator`](@ref)
- [`projected_energy`](@ref)
- [`variational_energy_estimator`](@ref)
- [`growth_estimator`](@ref)
- [`growth_estimator_analysis`](@ref)
- [`mixed_estimator`](@ref)
- [`mixed_estimator_analysis`](@ref)
- [`rayleigh_replica_estimator`](@ref)
- [`rayleigh_replica_estimator_analysis`](@ref)
- [`val_and_errs`](@ref)
- [`val`](@ref)
- [`mean_and_se`](@ref)
"""
module StatsTools

using DataFrames: DataFrames, DataFrame, metadata
using Distributions: Distributions, Chisq, Distribution, MvNormal, Normal,
    cquantile, var
using LinearAlgebra: LinearAlgebra, diag, norm
using MonteCarloMeasurements: MonteCarloMeasurements, AbstractParticles, pcov,
    pextrema, piterate, pmaximum, pmean, pmedian,
    pmiddle, pminimum, pquantile
using Random: Random
using SpecialFunctions: SpecialFunctions, erf
using Statistics: Statistics
using StrFormat: StrFormat, @f_str
using StrLiterals: StrLiterals

import ProgressLogging, Folds
import MacroTools
import Measurements

import Statistics: cov
import Measurements: measurement
import MonteCarloMeasurements: Particles
import Base: NamedTuple

export growth_witness, smoothen
export blocking_analysis, blocking_analysis_data, mean_and_se
export ratio_of_means
export replica_fidelity, med_and_errs, ratio_with_errs, to_measurement
export growth_estimator, mixed_estimator, rayleigh_replica_estimator, w_lin, w_exp, projected_energy, shift_estimator
export growth_estimator_analysis, mixed_estimator_analysis, rayleigh_replica_estimator_analysis
export variational_energy_estimator
export pmedian, pquantile, pmiddle, piterate, pextrema, pminimum, pmaximum, pmean, pcov
export val_and_errs, errs, val

include("growth_witness.jl")
include("variances.jl")
include("blocking.jl")
include("ratio_of_means.jl")
include("convenience.jl")
include("fidelity.jl")
include("reweighting.jl")
include("variational_energy_estimator.jl")

end  # module StatsTools
