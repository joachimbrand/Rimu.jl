```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using StaticArrays
using Setfield

using Base.Cartesian

export BitString
export num_bits, num_chunks, chunk_type, chunks, chunk_bits, top_chunk_bits
export find_occupied_mode, find_mode, move_particle

export AbstractFockAddress, SingleComponentFockAddress, BoseFS, BoseFS2C, FermiFS, CompositeFS
export num_particles, num_modes, num_components
export onr, near_uniform, occupied_modes, is_occupied, num_occupied_modes

include("bitstring.jl")
include("fockaddress.jl")
include("bosefs.jl")
include("fermifs.jl")
include("multicomponent.jl")

end
