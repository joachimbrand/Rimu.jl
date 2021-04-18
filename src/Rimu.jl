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

include("FastBufs.jl")
using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses/BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("ConsistentRNG.jl")
@reexport using .ConsistentRNG
include("Hamiltonians/Hamiltonians.jl")
@reexport using .Hamiltonians
include("Blocking.jl")
@reexport using .Blocking
include("RimuIO.jl")
using .RimuIO

export lomc!
export fciqmc!, FciqmcRunStrategy, RunTillLastStep
export MemoryStrategy, NoMemory, DeltaMemory, ShiftMemory
export ProjectStrategy, NoProjection, NoProjectionTwoNorm, ThresholdProject, ScaledThresholdProject
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
include("norm_project.jl")
include("lomc.jl")                  # top level

# Modules for parallel computing not exported by default for now
include("EmbarrassinglyDistributed.jl")
include("RMPI/RMPI.jl")
# @reexport using .RMPI

end # module
