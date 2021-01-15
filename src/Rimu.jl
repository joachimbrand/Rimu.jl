"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Reexport, Parameters, LinearAlgebra, DataFrames
using Setfield, StaticArrays
using SplittablesBase, ThreadsX

import MPI, DataStructures

include("FastBufs.jl")
using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("ConsistentRNG.jl")
@reexport using .ConsistentRNG
include("Hamiltonians.jl")
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
export StochasticStyle, IsStochastic, IsDeterministic, IsStochastic2Pop
# export IsSemistochastic # is not yet ready
export IsStochasticNonlinear, IsStochasticWithThreshold
export @setThreshold, @setDeterministic, setThreshold
export threadedWorkingMemory, localpart, walkernumber
export RimuIO

include("strategies_and_params.jl")
include("helpers.jl")
include("fciqmc.jl")

include("RMPI.jl")
# @reexport using .RMPI

export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
