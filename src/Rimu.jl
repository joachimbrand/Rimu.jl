"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Reexport

include("FastBufs.jl")
@reexport using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors


export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
