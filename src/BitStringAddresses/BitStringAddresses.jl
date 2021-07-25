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

export AbstractFockAddress, BoseFS, BoseFS2C, num_particles, num_modes
export onr, nearUniform, nearUniformONR, occupied_orbitals

include("bitstring.jl")
include("bosefs.jl")

end
