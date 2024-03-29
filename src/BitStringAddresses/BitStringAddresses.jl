```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using LinearAlgebra: LinearAlgebra, I, dot
using Parameters: Parameters, @unpack
using Setfield: Setfield, @set, @set!, setindex
using SparseArrays: SparseArrays, SparseVector, nonzeros, rowvals, spzeros
using StaticArrays: StaticArrays, @MVector, FieldVector, MVector, SA, SVector

using Base.Cartesian

export AbstractFockAddress, SingleComponentFockAddress, BoseFS, BoseFS2C, FermiFS
export CompositeFS, FermiFS2C, time_reverse
export OccupationNumberFS
export BoseFSIndex, FermiFSIndex
export BitString, SortedParticleList
export num_particles, num_modes, num_components
export find_occupied_mode, find_mode, occupied_modes, is_occupied, num_occupied_modes
export excitation, near_uniform, OccupiedModeMap, OccupiedPairsMap
export onr, occupation_number_representation
export hopnextneighbour, bose_hubbard_interaction
export @fs_str

include("fockaddress.jl")
include("bitstring.jl")
include("sortedparticlelist.jl")
include("bosefs.jl")
include("fermifs.jl")
include("multicomponent.jl")
include("occupationnumberfs.jl")

end
