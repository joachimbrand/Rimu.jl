"""
    module ConsistentRNG
Provides a an array random number generators with one for each thread.

Exports [`cRand()`](@ref) and [`seedCRNG!()`](@ref). These are thread consistent.
"""
module ConsistentRNG
#__precompile__(false) # do not precompile

using RandomNumbers
import Random
# using Distributed # only for info message

export cRand, cRandn, seedCRNG!, trng, newChildRNG # threadsafe
export sync_cRandn

"""
Baseline random number generator used throughout.
Currently we are using 'Xoshiro256StarStar' from 'RandomNumbers.jl', see the
[Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1)
and this [Blog post](https://nullprogram.com/blog/2017/09/21/).
In order to change the random number generator, edit 'ConsistentRNG.jl'.
"""
const CRNG = RandomNumbers.Xorshifts.Xoshiro256StarStar

"""
    CRNGs[]
Defines an array of random number generators suitable for threaded code.
For MPI or distributed
runs it should be seeded separately on each process with [`seedCRNG!`](@ref).
Currently we are using 'Xoshiro256StarStar' from 'RandomNumbers.jl', see the
[Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1)
and this [Blog post](https://nullprogram.com/blog/2017/09/21/).
In order to change the random number generator, edit 'ConsistentRNG.jl'.

```julia
rng = CRNGs[][Threads.threadid()]
rand(rng)
```
"""
const CRNGs = Ref([CRNG(),])
# Define this as constant with the correct type for precompilation. Will be
# initialised at runtime in __init__()

"""
    trng()
Thread local random number generator.

```julia
rand(trng())
rand(trng(),UInt)
```
"""
@inline trng() = @inbounds CRNGs[][Threads.threadid()]

"""
    r = cRand(args...)
Similar to 'rand(args)' but uses consistent random number generator 'CRNGs[]'.
'cRand()' generates a single uniformly distributed random number in the interval [0,1).
Currently we are using 'Xoshiro256StarStar' from 'RandomNumbers.jl', see the
[Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1)
and this [Blog post](https://nullprogram.com/blog/2017/09/21/).
"""
@inline cRand(args...) = rand(trng(), args...)
# cRand(args...) = rand(CRNGs, args...)
# #

"""
    r = cRandn(args...)
Similar to 'randn(args)' but uses consistent random number generators 'CRNGs[]'.
'cRandn()' generates a single normally distributed random number.
Currently we are using 'Xoshiro256StarStar' from 'RandomNumbers.jl', see the
[Documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/benchmark/#Benchmark-1)
and this [Blog post](https://nullprogram.com/blog/2017/09/21/).
"""
@inline cRandn(args...) = randn(trng(), args...)

"""
    seedCRNG!([seed])
Seed the threaded consistent random number generators `CRNGs[]`. If a single
number is given, this will be used to seed a random sequence, which is hashed
and then used to generate seeds for each rng in
the vector [`CRNGs`](@ref). When no argument is given, each rng is seeded
randomly.
"""
function seedCRNG!(seeds::Vector)
    for (i,seed) in enumerate(seeds)
        Random.seed!(CRNGs[][i], seed)
    end
    return CRNGs[]
end
function seedCRNG!(seed::Number)
    rng=CRNG(seed)
    for i = 1:length(CRNGs[])
        @inbounds Random.seed!(CRNGs[][i], hash(rand(rng,UInt)))
    end
    return CRNGs[]
end
seedCRNG!() = map(Random.seed!, CRNGs[])

"""
    newChildRNG(parent_rng = trng())
Random number generator that is seeded deterministically from the
thread-consistent global rng [`trng()`](@ref). By scrambling with `hash()`,
a statistically independent pseudo-random sequence from the parent rng is
accessed.
"""
function newChildRNG(parent_rng = trng())
    return CRNG(hash(rand(parent_rng,UInt)))
end

"""
    sync_cRandn(v)
Generate one random number with [`cRandn()`](@ref) in a synchronous way.
Defaults to [`cRandn()`](@ref).
"""
sync_cRandn(v) = cRandn()

# generate random seeds for the RNGs when loading the module at runtime
function __init__()
    CRNGs[] = [CRNG() for i in 1:Threads.nthreads()]
end

end # module
