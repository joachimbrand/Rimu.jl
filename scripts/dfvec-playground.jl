# script for testing MPI functionality
using Rimu
import MPI
using Feather
import Humanize: datasize, timedelta

function main()
starttime = time()


# define the problem
Ĥ = BoseHubbardReal1D(
    n = 50,
    m = 50,
    u = 6.0,
    t = 1.0,
    AT = BSAdd128)

targetwalkers = 1_000
timesteps = 200

# prepare initial state and allocate memory
aIni = nearUniform(Ĥ)
svec = DFVec(Dict(aIni => (10,-1.0)), (targetwalkers*10))
# our buffer for the state vector and initial state
w = similar(svec) # working memory, preallocated

# seed random number generator (different seed on each rank)
Rimu.ConsistentRNG.seedCRNG!(17+19)


dvs = svec
size_est = Base.summarysize(w) + Base.summarysize(dvs)
println("Preparing fciqmc")
println("Size of primary data structures: ", datasize(size_est))


# set parameters, only single time step for compilation
params = RunTillLastStep(dτ = 0.001, laststep = 1)
s_strat = LogUpdateAfterTargetWalkers(targetwalkers = targetwalkers)
t_strat = ConstantTimeStep()
et = @elapsed df = fciqmc!(dvs, params, Ĥ, s_strat, t_strat, w)
println("fciqmc compiled in $et seconds")

params.laststep = timesteps # set actual number of time steps to run
# print info about what we are doing
println("Finding ground state for:")
println(Ĥ)
println("Strategies for run:")
println(params, s_strat, t_strat)

# run main calculation
println("Starting main calculation with $timesteps steps. Hang on ...")
et = @elapsed df = fciqmc!(dvs, params, df, Ĥ, s_strat, t_strat, w)

print("$timesteps fciqmc steps finished in $et seconds, or about ")
println(timedelta(Int(round(et))))

# write results to disk
Feather.write("fciqmcdata.feather", df)
println("written data to disk")

println("Finished in overall ", timedelta(Int(round(time()-starttime))))

end # main

main()
