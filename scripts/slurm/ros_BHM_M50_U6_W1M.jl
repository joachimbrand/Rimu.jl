# script for testing MPI functionality
starttime = time() # for recording overall run time

using Rimu
import MPI
using Feather
import Humanize: datasize, timedelta

function main()
    # Initialise MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    id = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)

    # define the problem
    Ĥ = BoseHubbardReal1D(
        n = 50,
        m = 50,
        u = 6.0,
        t = 1.0,
        AT = BSAdd128)

    targetwalkers = 1_000_000
    timesteps = 4_000

    # prepare initial state and allocate memory
    aIni = nearUniform(Ĥ)
    svec = DVec(Dict(aIni => 10), (targetwalkers*2)÷np)
    # our buffer for the state vector and initial state
    w = similar(svec) # working memory, preallocated

    # seed random number generator (different seed on each rank)
    Rimu.ConsistentRNG.seedCRNG!(17+19*id)

    # define parallel strategy
    dvs = mpi_one_sided(svec) # wrap with mpi strategy
    size_est = Base.summarysize(w) + Base.summarysize(dvs)
    dvs.isroot && println("Preparing fciqmc")
    dvs.isroot && println("Size of primary data structures per rank: ", datasize(size_est))


    # set parameters, only single time step for compilation
    params = RunTillLastStep(dτ = 0.0001, laststep = 1)
    s_strat = LogUpdateAfterTargetWalkers(targetwalkers = targetwalkers)
    r_strat = ReportDFAndInfo(k=1,i=100, writeinfo = dvs.isroot)
    t_strat = ConstantTimeStep()
    et = @elapsed df = fciqmc!(dvs, params, Ĥ, s_strat, r_strat, t_strat, w)
    dvs.isroot && println("parallel fciqmc compiled in $et seconds")

    params.laststep = timesteps # set actual number of time steps to run
    # print info about what we are doing
    dvs.isroot && println("Finding ground state for:")
    dvs.isroot && println(Ĥ)
    dvs.isroot && println("Strategies for run:")
    dvs.isroot && println(params, s_strat, r_strat, t_strat)
    dvs.isroot && println("DistributeStrategy: ", dvs.s)

    # run main calculation
    dvs.isroot && println("Starting main calculation with $timesteps steps. Hang on ...")
    et = @elapsed df = fciqmc!(dvs, params, df, Ĥ, s_strat, r_strat, t_strat, w)
    actualtimesteps = size(df,1)-1

    dvs.isroot && print("$actualtimesteps fciqmc steps finished in $et seconds, or about ")
    dvs.isroot && println(timedelta(Int(round(et))))

    # write results to disk
    dvs.isroot && Feather.write("fciqmcdata.feather", df)
    dvs.isroot && println("written data to disk")

    # cleanup
    free(dvs) # MPI sync
    MPI.Finalize()
    return id # return the rank of the MPI process
end # main

rank = main()
rank==0 && println("Finished in overall ", timedelta(Int(round(time()-starttime))))
