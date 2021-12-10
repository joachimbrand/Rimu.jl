```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using LinearAlgebra
using StaticArrays
using Setfield
using Parameters

using Base.Cartesian

export BitString
export num_bits, num_chunks, chunk_type, chunks, chunk_bits, top_chunk_bits

export AbstractFockAddress, SingleComponentFockAddress, BoseFS, BoseFS2C, FermiFS
export CompositeFS, FermiFS2C, time_reverse
export BoseFSIndex, FermiFSIndex
export num_particles, num_modes, num_components
export find_occupied_mode, find_mode, occupied_modes, is_occupied, num_occupied_modes
export excitation, onr, near_uniform, OccupiedModeMap
export @fs_str

include("bitstring.jl")
include("fockaddress.jl")
include("bosefs.jl")
include("fermifs.jl")
include("multicomponent.jl")

end
