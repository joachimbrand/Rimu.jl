"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Reexport

include("FastBufs.jl")
using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("Walkers.jl")
@reexport using Walkers
include("Hamiltonians.jl")
@reexport using Hamiltonians

export fciqmc!, FCIQMCParams,StochasticStyle, IsStochastic, IsDeterministic
export IsSemistochastic


include("fciqmc.jl")



export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
