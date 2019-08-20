"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

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

export fciqmc!, FCIQMCParams
export ShiftUpdateStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export StochasticStyle, IsStochastic, IsDeterministic, IsSemistochastic


include("fciqmc.jl")



export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
