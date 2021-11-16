"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Arrow
using DataFrames
using DataStructures
using LinearAlgebra
using OrderedCollections # for LittleDict
using Parameters
using Reexport
using Setfield
using SplittablesBase
using StaticArrays
using StatsBase
using ThreadsX
using ProgressLogging
using TerminalLoggers: TerminalLogger
using Logging: ConsoleLogger
import ConsoleProgressMonitor

@reexport using Distributed

include("helpers.jl") # non MPI-dependent helper functions

include("ConsistentRNG.jl")
@reexport using .ConsistentRNG
include("Interfaces/Interfaces.jl")
@reexport using .Interfaces
include("StochasticStyles/StochasticStyles.jl")
@reexport using .StochasticStyles
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses/BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("Hamiltonians/Hamiltonians.jl")
@reexport using .Hamiltonians
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
export PostStepStrategy, Projector, ProjectedEnergy, SignCoherence, WalkerLoneliness, Timer
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

# Modules for parallel computing not exported by default for now
include("EmbarrassinglyDistributed.jl")

# analysis tool not reexported (to be deprecated)
include("Blocking.jl")

end # module
