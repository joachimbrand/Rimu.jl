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
export num_bits, num_chunks, chunks, num_particles, num_modes
export bitaddr, maxBSLength # deprecate

include("abstract.jl")
include("bsadd.jl")
include("bitadd.jl")
include("bosefs.jl")

end
