"""
    module ConsistentRNG
Provides a random number generator that is locally and independently seeded for
each worker. The initial state is save to file. If the appropriate file is
found in the working directory, then the random number generator is loaded from
this file.

Exports `cRand()`, `seedCRNG!()`, and `CRNG`.
"""
module ConsistentRNG
#__precompile__(false) # do not precompile

using RandomNumbers
import Random
using Distributed # only for info message

export cRand, CRNG, seedCRNG!

# pid = myid() # get process id number
# rngFileName = "rngP$pid.jld2"
# # file name for storing the initial state of the rng encodes the process number
#
# if isfile(rngFileName)
#       rdict = load(rngFileName)
#       const CRNG = copy(rdict["rng"])
#       println("CRNG loaded from file ",rngFileName)
#       @info "loaded random number generator from file " rngFileName CRNG
# else
#       const CRNG = Xorshifts.Xoroshiro128Plus()
#       save(rngFileName,"rng",CRNG)
#       println("New CRNG generated and saved to file ",rngFileName)
#       # @info "generated random number generator and saved to file " rngFileName CRNG
# end

"""
    CRNG
Defines the random number generator that should be used everywhere. For parallel
runs it should be seeded separately on each process. Currently we are using
'Xoroshiro128Plus' from 'RandomNumbers.jl', see the
[Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1).
In order to change the random number generator, edit 'ConsistentRNG.jl'.
"""
const CRNG = RandomNumbers.Xorshifts.Xoroshiro128Plus() # fast and good
# Alternatively:
# const CRNG = Random.MersenneTwister() # standard Julia RNG


"""
    r = cRand(args...)
Similar to 'rand(args)' but uses consistent random number generator 'CRNG'.
'cRand()' generates a single uniformly distributed random number in the interval [0,1).
Currently we are using `Xorshifts.Xoroshiro128Plus()` as random number generator
from the `RandomNumbers` package. The initial state was seeded separately
for each worker and saved to file."""
cRand(args...) = rand(CRNG, args...)

# """
#     seedcrng(seed::Tuple{UInt64,UInt64})
# Seed the consistent random number generator 'CRNG'. Similar to
# 'Random.seed!(CRNG,seed)' but without forwarding the sequence.
# """
# function seedcrng(seed::Tuple{UInt64,UInt64})
#     if seed == (0,0)
#         error("0 cannot be the seed")
#     end
#     CRNG.x = seed[1]
#     CRNG.y = seed[2]
#     @info "Seeded random number generator" myid() CRNG
#     return CRNG
# end
"""
    seedCRNG!(seed)
Seed the consistent random number generator 'CRNG'.
"""
function seedCRNG!(seed)
    Random.seed!(CRNG, seed)
    # @info "Seeded random number generator" myid() CRNG
    return CRNG
end

end
