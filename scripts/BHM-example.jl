# load all needed modules
using Rimu
using Feather
using DataFrames


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

# prepare initial state and allocate memory
# initial address
aIni = nearUniform(Ĥ)
# initial number of walkers per rank
nIni = 1
# set the DVec size to be targetwalkers*10
svec = DVec(Dict(aIni => nIni), targetwalkers*10)

# passing dτ and total number of time steps into params
params = RunTillLastStep(dτ = dτ, laststep = steps_equilibrate + steps_measure)
# strategy for updating the shift
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
# strategy for reporting info and setting projectors
r_strat = ReportDFAndInfo(k = k, i = 100, projector = UniformProjector())
# strategy for updating dτ
t_strat = ConstantTimeStep()

# print info about what we are doing
println("Finding ground state for:")
println(Ĥ)
println("Strategies for run:")
println(params, s_strat)
println(t_strat)

# start the main FCIQMC loop with a timer "et"
et = @elapsed df = lomc!(Ĥ,svec;
                        params = params,
                        laststep = steps_equilibrate + steps_measure,
                        s_strat = s_strat,
                        r_strat = r_strat,
                        τ_strat = t_strat)

# saving output data
println("Writing data to disk...")
Feather.write("fciqmcdata.feather", df.df)

# some quick stats
(qmcShift,qmcShiftErr,qmcEnergy,qmcEnergyErr) = autoblock(df.df,start=steps_equilibrate)
println("Energy from $steps_measure steps with $targetwalkers walkers:
Shift: $qmcShift ± $qmcShiftErr
E_proj:$qmcEnergy ± $qmcEnergyErr")

# Finished !
println("Finished!")
