"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Reexport, Parameters, LinearAlgebra, DataFrames
import MPI

include("FastBufs.jl")
using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("ConsistentRNG.jl")
using .ConsistentRNG
include("Hamiltonians.jl")
@reexport using .Hamiltonians
include("Blocking.jl")
@reexport using .Blocking

export fciqmc!, FciqmcRunStrategy, RunTillLastStep
export ShiftUpdateStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export DontUpdate, DelayedLogUpdate, DelayedLogUpdateAfterTargetWalkers
export HistoryLogUpdate
export ReportingStrategy, EveryTimeStep, EveryKthStep, ReportDFAndInfo
export TimeStepStrategy, ConstantTimeStep
export StochasticStyle, IsStochastic, IsDeterministic, IsSemistochastic
export IsStochasticNonlinear
export DistributeStrategy, MPIData, MPIDefault, MPIOSWin
export mpi_default, mpi_one_sided, fence, put, sbuffer, sbuffer!, targetrank
export localpart, free, mpi_no_exchange

include("strategies_and_params.jl")
include("mpi_helpers.jl")
include("fciqmc.jl")



export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
