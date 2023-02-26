```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using LinearAlgebra
using StaticArrays
using SparseArrays
using Setfield
using Parameters

using Base.Cartesian

export AbstractFockAddress, SingleComponentFockAddress, BoseFS, BoseFS2C, FermiFS
export CompositeFS, FermiFS2C, time_reverse
export BoseFSIndex, FermiFSIndex
export BitString, SortedParticleList
export num_particles, num_modes, num_components
export find_occupied_mode, find_mode, occupied_modes, is_occupied, num_occupied_modes
export excitation, onr, near_uniform, OccupiedModeMap
export hopnextneighbour, bose_hubbard_interaction
export @fs_str

include("fockaddress.jl")
include("bitstring.jl")
include("sortedparticlelist.jl")
include("bosefs.jl")
include("fermifs.jl")
include("multicomponent.jl")

end
