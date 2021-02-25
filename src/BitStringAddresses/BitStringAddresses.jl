```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using StaticArrays
using Setfield

using Base.Cartesian

export AbstractFockAddress, BoseFS, BoseFS2C
export onr, nearUniform, nearUniformONR
export num_bits, num_chunks, chunks, num_particles, num_modes
export two_bit_mask, one_bit_mask, occupied_orbitals
export BitString

include("bitstring.jl")
include("bosefs.jl")

end
