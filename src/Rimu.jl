"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu
@assert VERSION ≥ v"1.3-" "Threaded version of `Rimu` requires Julia v1.3"

using Reexport, Parameters, LinearAlgebra, DataFrames

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
export TimeStepStrategy, ConstantTimeStep
export StochasticStyle, IsStochastic, IsDeterministic, IsSemistochastic

include("strategies_and_params.jl")
include("fciqmc.jl")



export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
