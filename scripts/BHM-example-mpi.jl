# load all needed modules for a MPI run
using Rimu
import MPI
using Feather
import Humanize: datasize, timedelta

function main()

    # start a timer
    starttime = time()

    # define the problem
    # m: number of lattice sites; n: number of particles
    m = n = 6
    # generating a configuration that particles are evenly distributed
    aIni = nearUniform(BoseFS{n,m})
    # define the Hamiltonian
    Ĥ = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)

    # define Monte Carlo settings
    # number of walkers to use
    targetwalkers = 1_000
    # number of time steps before doing statistics
    steps_equilibrate = 1_000
    # number of time steps used for statistics, e.g. time-average of shift etc.
    steps_measure = 1_000


    # set the size of a time step
    dτ = 0.001
    # report every k-th step
    k = 1


    # Initialise MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    id = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)


    # prepare initial state and allocate memory
    # initial address
    # aIni = nearUniform(BoseFS{n,m})
    # initial number of walkers per rank
    nIni = 1
    # set the DVec size to be targetwalkers*10÷np
    svec = DVec(Dict(aIni => nIni), (targetwalkers*10)÷np)
    # our buffer for the state vector and initial state

    # working memory, preallocated
    w = similar(svec)

    # seed random number generator (different seed on each rank)
    Rimu.ConsistentRNG.seedCRNG!(17+19*id)

    # define parallel strategy
    dvs = mpi_one_sided(svec) # wrap with mpi strategy
    size_est = Base.summarysize(w) + Base.summarysize(dvs)
    dvs.isroot && println("Preparing fciqmc")
    dvs.isroot && println("Size of primary data structures per rank: ", datasize(size_est))
    dvs.isroot && println("StochasticStyle(svec) = ", StochasticStyle(svec))
    dvs.isroot && println("rand: ",Rimu.ConsistentRNG.cRand())



    # set parameters, only single time step for compilation
    params = RunTillLastStep(dτ = dτ, laststep = 1)
    # strategy for updating the shift
    s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
    # strategy for reporting info and setting projectors
    r_strat = ReportDFAndInfo(k = 1, i = 100, projector = UniformProjector())
    # strategy for updating dτ
    t_strat = ConstantTimeStep()



    # print info about what we are doing
    dvs.isroot && println("Finding ground state for:")
    dvs.isroot && println(Ĥ)
    dvs.isroot && println("Strategies for run:")
    dvs.isroot && println(params, s_strat)
    dvs.isroot && println(t_strat)
    dvs.isroot && println("DistributeStrategy: ", dvs.s)

    # start the main FCIQMC loop with a timer "et"
    et = @elapsed df = lomc!(Ĥ,svec;
                            params = params,
                            s_strat = s_strat,
                            r_strat = r_strat,
                            τ_strat = t_strat)

    dvs.isroot && println("parallel fciqmc compiled in $et seconds")



    # set actual number of time steps to run
    params.laststep = steps_equilibrate + steps_measure
    # run main calculation
    dvs.isroot && println("Starting main calculation with $(steps_equilibrate + steps_measure) steps. Hang on ...")
    et = @elapsed df = lomc!(Ĥ,dvs;
			     params = params,
			     s_strat = s_strat,
			     r_strat = r_strat,
			     τ_strat = t_strat,
			     wm = w)

    dvs.isroot && print("$(steps_equilibrate + steps_measure) fciqmc steps finished in $et seconds, or about ")
    dvs.isroot && println(timedelta(Int(round(et))))

    # write results to disk
    dvs.isroot && Feather.write("fciqmcdata.feather", df.df)
    dvs.isroot && println("written data to disk")

    # cleanup
    free(dvs) # MPI sync
    MPI.Finalize()
    dvs.isroot && println("Finished in overall ", timedelta(Int(round(time()-starttime))))
    return id # return the rank of the MPI process
end # main

# run everything in main()
rank = main()
