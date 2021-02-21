```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using StaticArrays
using Setfield

using Base.Cartesian

import Base: isless, zero, iszero, show, ==, hash

export AbstractBitString, BSAdd64, BSAdd128, BitAdd
export AbstractFockAddress, BoseFS, BoseFS2C
export onr, nearUniform, nearUniformONR
export num_bits, num_chunks, chunks, chunk_size, num_particles, num_modes
export two_bit_mask, one_bit_mask, occupied_orbitals
export BitString, chunk_type

include("abstract.jl")
include("bitstring.jl")
include("bsadd.jl")
include("bitadd.jl")
include("bosefs.jl")

end
