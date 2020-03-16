"""
    module ConsistentRNG
Provides a an array random number generators with one for each thread.

Exports `cRand()` and `seedCRNG!()`. These are thread consistent.
"""
module ConsistentRNG
#__precompile__(false) # do not precompile

using RandomNumbers
import Random
# using Distributed # only for info message

export cRand, seedCRNG! # threadsafe

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

# """
#     CRNG
# Defines the random number generator that should be used everywhere. For parallel
# runs it should be seeded separately on each process. Currently we are using
# 'Xoroshiro128Plus' from 'RandomNumbers.jl', see the
# [Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1).
# In order to change the random number generator, edit 'ConsistentRNG.jl'.
# """
# const CRNG = RandomNumbers.Xorshifts.Xoroshiro128Plus() # fast and good
# # Alternatively:
# # const CRNG = Random.MersenneTwister() # standard Julia RNG

"""
    CRNG
Defines an array of random number generators suitable for threaded code.
For MPI or distributed
runs it should be seeded separately on each process with [`seedCRNG!`](@ref).
Currently we are using 'Xoshiro256StarStar' from 'RandomNumbers.jl', see the
[Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1)
and this [Blog post](https://nullprogram.com/blog/2017/09/21/).
In order to change the random number generator, edit 'ConsistentRNG.jl'.

```julia
rng = CRNG[Threads.threadid()]
rand(rng)
```
"""
# const CRNG = [RandomNumbers.Xorshifts.Xoroshiro128Plus() for i in 1:Threads.nthreads()]
const CRNG = Tuple(RandomNumbers.Xorshifts.Xoshiro256StarStar() for i in 1:Threads.nthreads())
# fast and good

@inline trng() = @inbounds CRNG[Threads.threadid()]

"""
    r = cRand(args...)
Similar to 'rand(args)' but uses consistent random number generator 'CRNG'.
'cRand()' generates a single uniformly distributed random number in the interval [0,1).
Currently we are using `Xorshifts.Xoroshiro128Plus()` as random number generator
from the `RandomNumbers` package. The initial state was seeded separately
for each worker and saved to file."""
@inline cRand(args...) = rand(trng(), args...)
# cRand(args...) = rand(CRNG, args...)
# #

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
# """
#     seedCRNG!(seed)
# Seed the consistent random number generator 'CRNG'.
# """
# function seedCRNG!(seed)
#     Random.seed!(CRNG, seed)
#     # @info "Seeded random number generator" myid() CRNG
#     return CRNG
# end

"""
    seedCRNG!(seed)
Seed the threaded consistent random number generators `CRNG`. If a single
number is given, this will be used to seed a Mersenne Twister random number
generator, which is then used to generate `UInt128` seeds for each rng in
the vector [`TCRNG`](@ref).
"""
function seedCRNG!(seeds::Vector)
    for (i,seed) in enumerate(seeds)
        Random.seed!(CRNG[i], seed)
    end
    return CRNG
end
function seedCRNG!(seed::Number)
    mtrng = RandomNumbers.MersenneTwisters.MT19937(seed)
    seedCRNG!(rand(mtrng,UInt128,length(CRNG)))
    return CRNG
end

end # module
