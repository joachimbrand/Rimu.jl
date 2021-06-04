"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Arrow
using Reexport, Parameters, LinearAlgebra, DataFrames
using Setfield, StaticArrays
using SplittablesBase, ThreadsX
@reexport using Distributed
import MPI, DataStructures

include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses/BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("ConsistentRNG.jl")
@reexport using .ConsistentRNG
include("Hamiltonians/Hamiltonians.jl")
@reexport using .Hamiltonians
include("RimuIO.jl")
@reexport using .RimuIO

export lomc!
export FciqmcRunStrategy, RunTillLastStep
export MemoryStrategy, NoMemory, DeltaMemory, ShiftMemory
export ShiftUpdateStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export DontUpdate, DelayedLogUpdate, DelayedLogUpdateAfterTargetWalkers
export DoubleLogUpdate, DelayedDoubleLogUpdate, DoubleLogUpdateAfterTargetWalkers
export DelayedDoubleLogUpdateAfterTW
export DoubleLogUpdateAfterTargetWalkersSwitch
export HistoryLogUpdate
export ReportingStrategy, ReportDFAndInfo, ReportToFile
export ReplicaStrategy, NoStats, AllOverlaps
export PostStepStrategy, Projector, ProjectedEnergy, SignCoherence
export TimeStepStrategy, ConstantTimeStep, OvershootControl
export threadedWorkingMemory, localpart, walkernumber

include("report.jl")

include("strategies_and_params/fciqmcrunstrategy.jl")
include("strategies_and_params/memorystrategy.jl")
include("strategies_and_params/poststepstrategy.jl")
include("strategies_and_params/replicastrategy.jl")
include("strategies_and_params/reportingstrategy.jl")
include("strategies_and_params/shiftstrategy.jl")
include("strategies_and_params/timestepstrategy.jl")

include("helpers.jl")               # non MPI-dependent helper functions
include("fciqmc_col.jl")            # third level
include("apply_memory_noise.jl")
include("fciqmc_step.jl")           # second level
include("update_dvec.jl")
include("lomc.jl")                  # top level

include("RMPI/RMPI.jl")

# Modules for parallel computing not exported by default for now
include("EmbarrassinglyDistributed.jl")

# analysis tools not reexported
include("Blocking.jl")
include("StatsTools/StatsTools.jl")

end # module
