module Rimu

using Arrow
using DataFrames
using DataStructures
using LinearAlgebra
using OrderedCollections # for LittleDict
using Parameters
using Reexport
using Setfield
using StaticArrays
using StatsBase
using ProgressLogging
using TerminalLoggers: TerminalLogger
using Logging: ConsoleLogger
import ConsoleProgressMonitor
import TOML

"""
    Rimu.PACKAGE_VERSION
Constant that contains the current `VersionNumber` of `Rimu`.
"""
const PACKAGE_VERSION = VersionNumber(TOML.parsefile(pkgdir(Rimu, "Project.toml"))["version"])

@doc """
    Rimu
**Random integrators for many-body quantum systems**

Welcome to `Rimu` version $PACKAGE_VERSION.
Read the documentation [online](https://joachimbrand.github.io/Rimu.jl/).
"""
Rimu

include("helpers.jl") # non MPI-dependent helper functions

include("Interfaces/Interfaces.jl")
@reexport using .Interfaces
include("BitStringAddresses/BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("Hamiltonians/Hamiltonians.jl")
@reexport using .Hamiltonians
include("StochasticStyles/StochasticStyles.jl")
@reexport using .StochasticStyles
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("RimuIO.jl")
@reexport using .RimuIO
include("StatsTools/StatsTools.jl")
@reexport using .StatsTools

export lomc!
export FciqmcRunStrategy, RunTillLastStep
export MemoryStrategy, NoMemory, DeltaMemory, ShiftMemory
export ShiftStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export DontUpdate, DoubleLogUpdate, DoubleLogUpdateAfterTargetWalkers
export ReportingStrategy, ReportDFAndInfo, ReportToFile
export ReplicaStrategy, NoStats, AllOverlaps
export PostStepStrategy, Projector, ProjectedEnergy, SignCoherence, WalkerLoneliness, Timer,
    SingleParticleDensity, single_particle_density
export TimeStepStrategy, ConstantTimeStep, OvershootControl
export threadedWorkingMemory, localpart, walkernumber
export smart_logger, default_logger

function __init__()
    # Turn on smart logging once at runtime. Turn off with `default_logger()`.
    smart_logger()
end

include("strategies_and_params/fciqmcrunstrategy.jl")
include("strategies_and_params/memorystrategy.jl")
include("strategies_and_params/poststepstrategy.jl")
include("strategies_and_params/replicastrategy.jl")
include("strategies_and_params/reportingstrategy.jl")
include("strategies_and_params/shiftstrategy.jl")
include("strategies_and_params/timestepstrategy.jl")
include("strategies_and_params/threadingstrategy.jl")
include("strategies_and_params/deprecated.jl")

include("apply_memory_noise.jl")
include("lomc.jl")                  # top level

include("RMPI/RMPI.jl")

end # module
