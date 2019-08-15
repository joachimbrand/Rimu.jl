"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Reexport, Parameters

include("FastBufs.jl")
using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("ConsistentRNG.jl")
# using .ConsistentRNG
include("Hamiltonians.jl")
@reexport using .Hamiltonians

export fciqmc!, FCIQMCParams,StochasticStyle, IsStochastic, IsDeterministic
export IsSemistochastic


include("fciqmc.jl")



export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
