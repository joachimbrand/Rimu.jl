# simple sample script for running FCIQMC in parallel with MPI
# start on three MPI ranks with two threads each:
# $ mpirun -np 3 julia -t 2 script_mpi_minimum.jl
println("hello!")
using Rimu, Rimu.RMPI
mpi_seed_CRNGs!(17) # seed RNGs for reproducible results
@mpi_root @show VERSION

mpi_barrier() # optional - use for debugging and sanity checks
@info "after barrier 1" mpi_rank() mpi_size() Threads.nthreads() cRand(UInt16)

m = 10
n= 3
aIni = nearUniform(BoseFS{n,m})
ham = BoseHubbardReal1D(aIni; u = 1.0, t = 1.0)
ζ = 0.08
N = 1000
s_strat = DoubleLogUpdate(ζ = ζ, ξ = ζ^2/4, targetwalkers = N)

# the important step is to wrap the `DVec` in an `MPIData` type to enable
# exchanging walkers between MPI ranks
v = DVec2(aIni => 2; capacity = (s_strat.targetwalkers*2÷mpi_size()+100))
dv = MPIData(v; setup = RMPI.mpi_point_to_point)
@mpi_root @show dv.s

params = RunTillLastStep(dτ = 0.001, laststep = 500)

# use `localpart(dv)` to access the `DVec`
# here we set up a projector for projected energy calculation
r_strat = EveryTimeStep(projector = copytight(localpart(dv)))

wn = walkernumber(dv) # an MPI synchronising operation; must not appear after `@mpi_root`
# write a message only from the root rank
@mpi_root @info "running on $(mpi_size()) ranks with  $(mpi_size()) * 2  initial walkers" wn
@info "before 1st lomc!" mpi_rank() cRand(UInt16)

# compilation (optional)
nt0 = lomc!(ham,dv; params = deepcopy(params), s_strat, r_strat, laststep = 0, threading=false)
# run `lomc!()` passing in the `MPIData` object
@info "after 1st lomc!" mpi_rank() cRand(UInt16)

el = @elapsed nt = lomc!(ham,dv; params, s_strat, r_strat, threading=false)
@info "after 2nd lomc!" mpi_rank() cRand(UInt16)

@mpi_root @info "Parallel fciqmc completed in $el seconds."
@info "final walker number" mpi_rank() walkernumber(localpart(dv)) walkernumber(dv)

savefile = joinpath(@__DIR__,"mpi_df_ptp.arrow") # using the Arrow data format relying on Arrow.jl
@mpi_root @info """saving dataframe into file "$savefile" """
@mpi_root RimuIO.save_df(savefile, nt.df)
# load the data with
# > using Rimu
# > df = RimuIO.load_df(savefile)
