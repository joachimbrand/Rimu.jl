# script for testing MPI functionality
using Rimu
import MPI
using Feather

function main()
MPI.Init()
comm = MPI.COMM_WORLD
id = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

Ĥ = BoseHubbardReal1D(
    n = 9,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BSAdd64)

targetwalkers = 10000
timesteps = 10000


aIni = nearUniform(Ĥ)
svec = DVec(Dict(aIni => 2), (targetwalkers*3)÷np)
# our buffer for the state vector and initial state
w = similar(svec) # working memory, preallocated
dvs = mpi_one_sided(svec) # wrap with mpi strategy
# set default parameters, only single time step for compilation
params = RunTillLastStep(laststep = 1)
s_strat = LogUpdateAfterTargetWalkers(targetwalkers = targetwalkers)
t_strat = ConstantTimeStep()
et = @elapsed df = fciqmc!(dvs, params, Ĥ, s_strat, t_strat, w)
dvs.isroot && println("parallel fciqmc compiled in $et seconds")

params.laststep = timesteps # set actual number of time steps to run
et = @elapsed df = fciqmc!(dvs, params, df, Ĥ, s_strat, t_strat, w)
dvs.isroot && println("$timesteps fciqmc steps finished in $et seconds")

dvs.isroot && Feather.write("fciqmcdata.feather", df)
dvs.isroot && println("written data to disk")

# cleanup
free(dvs) # MPI sync
MPI.Finalize()
end # main

main()
