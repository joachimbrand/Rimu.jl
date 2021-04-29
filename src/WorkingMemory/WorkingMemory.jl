module WorkingMemory

using ..DictVectors
using ..Rimu

export AbstractWorkingMemory
export spawn!, setup_working_memory
export sort_into_targets!

include("abstract.jl")
include("dvecmemory.jl")
include("setup.jl")

end
