# script for testing MPI functionality
starttime = time() # for recording overall run time

using Rimu
import MPI
using Feather
import Humanize: datasize, timedelta

# function main()
# Initialise MPI
# MPI.Init()
# comm = MPI.COMM_WORLD
id = 1 # MPI.Comm_rank(comm)
np = 1 # MPI.Comm_size(comm)

# define the problem
Ĥ = BoseHubbardReal1D(
    n = 8,
    m = 8,
    u = 6.0,
    t = 1.0,
    AT = BSAdd128)

targetwalkers = 2_000
timesteps = 1000



# prepare initial state and allocate memory
aIni = nearUniform(Ĥ)
svec = DVec(Dict(aIni => 10), (targetwalkers*2)÷np)
# our buffer for the state vector and initial state
w = similar(svec) # working memory, preallocated

pavec = FastDVec{BSAdd128,Int}(40)
for (add,elem) in Hops(Ĥ,nearUniform(Ĥ))
    pavec[add] = 1
end
pavec[aIni] = 1

# seed random number generator (different seed on each rank)
Rimu.ConsistentRNG.seedCRNG!(17+19*id)

# define parallel strategy
# dvs = mpi_one_sided(svec) # wrap with mpi strategy
dvs = svec
size_est = Base.summarysize(w) + Base.summarysize(dvs)
println("Preparing fciqmc")
println("Size of primary data structures per rank: ", datasize(size_est))


# set parameters, only single time step for compilation
params = RunTillLastStep(dτ = 0.001, laststep = 1)
s_strat = DelayedPartialNormUpdate(targetwalkers = targetwalkers, pavec = pavec)
#s_strat = DoubleLogUpdate(targetwalkers = targetwalkers)
#s_strat = LogUpdateAfterTargetWalkers(targetwalkers = targetwalkers)
r_strat = ReportDFAndInfo(k=1,i=100)#, writeinfo = dvs.isroot)
t_strat = ConstantTimeStep()
et = @elapsed df = fciqmc!(dvs, params, Ĥ, s_strat, r_strat, t_strat, w)
println("parallel fciqmc compiled in $et seconds")

params.laststep = timesteps # set actual number of time steps to run
# print info about what we are doing
println("Finding ground state for:")
println(Ĥ)
println("Strategies for run:")
println(params, s_strat, r_strat, t_strat)
# println("DistributeStrategy: ", dvs.s)

# run main calculation
println("Starting main calculation with $timesteps steps. Hang on ...")
et = @elapsed df = fciqmc!(dvs, params, df, Ĥ, s_strat, r_strat, t_strat, w)
actualtimesteps = size(df,1)-1
# params.laststep += 2000
# s_strat = PartialNormUpdate(targetwalkers = targetwalkers, ξ=0.0,pavec=pavec)#, a=3000)
# et += @elapsed df = fciqmc!(dvs, params, df, Ĥ, s_strat, r_strat, t_strat, w)
print("$actualtimesteps fciqmc steps finished in $et seconds, or about ")
println(timedelta(Int(round(et))))

# write results to disk
Feather.write("fciqmcdata.feather", df)
println("written data to disk")

# cleanup
# free(dvs) # MPI sync
# MPI.Finalize()
# return id # return the rank of the MPI process
# end # main

# rank = @time main()
# rank==0 && println("Finished in overall ", timedelta(Int(round(time()-starttime))))

include("../read_file_and_plot.jl")
ba_walker  = blocking(df[start_blocking:end,:norm])
qmcnorm = ba_walker[1,:mean]
normError = ba_walker[mtest(ba_walker),:std_err]
println(io, "Final walker number: $qmcnorm ± $normError")
