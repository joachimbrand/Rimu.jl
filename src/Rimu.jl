"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

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
using .RimuIO

export lomc!
export fciqmc!, FciqmcRunStrategy, RunTillLastStep
export MemoryStrategy, NoMemory, DeltaMemory, ShiftMemory
export ShiftUpdateStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export DontUpdate, DelayedLogUpdate, DelayedLogUpdateAfterTargetWalkers
export DoubleLogUpdate, DelayedDoubleLogUpdate, DoubleLogUpdateAfterTargetWalkers
export DelayedDoubleLogUpdateAfterTW
export DoubleLogUpdateAfterTargetWalkersSwitch
export HistoryLogUpdate
export ReportingStrategy, EveryTimeStep, EveryKthStep, ReportDFAndInfo
export TimeStepStrategy, ConstantTimeStep, OvershootControl
export threadedWorkingMemory, localpart, walkernumber
export RimuIO

include("strategies_and_params.jl") # type defs and helpers
include("helpers.jl")               # non MPI-dependent helper functions
include("fciqmc_col.jl")            # third level
include("apply_memory_noise.jl")
include("fciqmc_step.jl")           # second level
include("update_dvec.jl")
include("report.jl")
#include("lomc.jl")                  # top level
include("QMCState.jl")                  # top level
export lomc!

include("RMPI/RMPI.jl")
#using .RMPI

# Modules for parallel computing not exported by default for now
include("EmbarrassinglyDistributed.jl")

# analysis tools not reexported
include("Blocking.jl")
include("StatsTools/StatsTools.jl")

end # module
