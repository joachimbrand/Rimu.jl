# # Example: 1D Bose-Hubbard Model

# This is an example calculation finding the ground state of
# a 1D Bose-Hubbard chain with 6 particles in 6 lattice site.
# The Julia run-able script is in [`scripts/BHM-example.jl`](../../scripts/BHM-example.jl).

# Firstly, we load all needed modules.
# `Rimu` for FCIQMC calculation (obviously);
# `Feather` for saving output data in a `DataFrame` using `DataFrames`:

using Rimu
using Feather
using DataFrames


# Now we define the physical problem:
# Setting the number of lattice sites `m = 6`;
# and the number of particles `n = 6`:
m = 3; n = 3
# Generating a configuration that particles are evenly distributed:
aIni = nearUniform(BoseFS{n,m})
aIni2c = BoseFS2C(aIni,BoseFS((0,1,0)))
aIniMom = BoseFS((0,5,0))
aIni2cMom = BoseFS2C(aIniMom,BoseFS((0,1,0)))
# aIni2c_big = BoseFS2C(BoseFS(ones(Int,10)),BoseFS(ones(Int,10)))
# The Hamiltonian is defined based on the configuration `aIni`,
# with additional onsite interaction strength `u = 6.0`
# and the hopping strength `t = 1.0`:
# Ĥ = BoseHubbardReal1D(aIni; u = 1.0, t = 1.0)
Ĥ2c = BoseHubbard2CReal1D(aIni2c; ua = 1.0, ub = 0.0, ta = 1.0, tb = 0.0, v= 0.0)
Ĥ2cMom = BoseHubbard2CMom1D(aIni2cMom; ua = 1.0, ub = 0.0, ta = 1.0, tb = 0.0, v= 0.0)

# Now let's setup the Monte Carlo settings.
# The number of walkers to use in this Monte Carlo run:
targetwalkers = 5_000
# The number of time steps before doing statistics,
# i.e. letting the walkers to sample Hilbert and to equilibrate:
steps_equilibrate = 2_000
# And the number of time steps used for getting statistics,
# e.g. time-average of shift, projected energy, walker numbers, etc.:
steps_measure = 2_000


# Set the size of a time step
dτ = 0.001
# and we report QMC data every k-th step,
# setting `k = 1` means we record QMC data every step:
k = 1

# Now we prepare initial state and allocate memory.
# The initial address is defined above as `aIni = nearUniform(Ĥ)`.
# Define the initial number of walkers per rank:
nIni = 10
# Putting the `nIni` number of walkers into the initial address `aIni`,
# and set the DVec size to be targetwalkers*10:
# svec = DVec(Dict(aIni => nIni), targetwalkers*10)
svec2c = DVec(Dict(aIni2c => nIni), targetwalkers*10)
svec2cMom = DVec(Dict(aIni2cMom => nIni), targetwalkers*10)
# Let's plant a seed for the random number generator to get consistent result:
Rimu.ConsistentRNG.seedCRNG!(17)

# Now let's setup all the FCIQMC strategies.

# Passing dτ and total number of time steps into params:
params = RunTillLastStep(step = 0, dτ = dτ, laststep = steps_equilibrate + steps_measure)
# Strategy for updating the shift:
s_strat = DoubleLogUpdate(targetwalkers = targetwalkers, ζ = 0.08)
# Strategy for reporting info and setting projectors:
# r_strat = ReportDFAndInfo(k = k, i = 100, projector = copytight(svec))
r_strat_2c = ReportDFAndInfo(k = k, i = 100, projector = copytight(svec2c))
r_strat_2cMom = ReportDFAndInfo(k = k, i = 100, projector = copytight(svec2cMom))
# Strategy for updating dτ:
t_strat = ConstantTimeStep()

# Print out info about what we are doing:
# println("Finding ground state for:")
# println(Ĥ)
# println("Strategies for run:")
# println(params, s_strat)
# println(t_strat)


# Finally, we can start the main FCIQMC loop:
# @time df = lomc!(Ĥ,svec;
#             params = params,
#             laststep = steps_equilibrate + steps_measure,
#             s_strat = s_strat,
#             r_strat = r_strat,
#             τ_strat = t_strat);
#
# params = RunTillLastStep(step = 0, dτ = dτ, laststep = steps_equilibrate + steps_measure)
@time df2c = lomc!(Ĥ2c,svec2c;
            params = params,
            laststep = steps_equilibrate + steps_measure,
            s_strat = s_strat,
            r_strat = r_strat_2c,
            τ_strat = t_strat);

params = RunTillLastStep(step = 0, dτ = dτ, laststep = steps_equilibrate + steps_measure)
@time df2cMom = lomc!(Ĥ2cMom,svec2cMom;
            params = params,
            laststep = steps_equilibrate + steps_measure,
            s_strat = s_strat,
            r_strat = r_strat_2cMom,
            τ_strat = t_strat);
# println("Writing data to disk...")
# Saving output data stored in `df.df` into a `.feather` file which can be read in later:
# Feather.write("fciqmcdata.feather", df.df)

# Now do some quick statistics:
# println(Ĥ)
# (qmcShift,qmcShiftErr,qmcEnergy,qmcEnergyErr) = autoblock(df.df,start=steps_equilibrate)
# println("Energy from $steps_measure steps with $targetwalkers walkers:
# Shift: $qmcShift ± $qmcShiftErr
# E_proj:$qmcEnergy ± $qmcEnergyErr")

println(Ĥ2c)
(qmcShift,qmcShiftErr,qmcEnergy,qmcEnergyErr) = autoblock(df2c.df,start=steps_equilibrate)
println("Energy from $steps_measure steps with $targetwalkers walkers:
Shift: $qmcShift ± $qmcShiftErr
E_proj:$qmcEnergy ± $qmcEnergyErr")

println(Ĥ2cMom)
(qmcShift,qmcShiftErr,qmcEnergy,qmcEnergyErr) = autoblock(df2cMom.df,start=steps_equilibrate)
println("Energy from $steps_measure steps with $targetwalkers walkers:
Shift: $qmcShift ± $qmcShiftErr
E_proj:$qmcEnergy ± $qmcEnergyErr")

# Finished !
println("Finished!")

using LinearAlgebra

# smat, adds = Hamiltonians.build_sparse_matrix_from_LO(Ĥ,aIni)
# eig = eigen(Matrix(smat))

smat2c, adds2c = Hamiltonians.build_sparse_matrix_from_LO(Ĥ2c,aIni2c)
eig2c = eigen(Matrix(smat2c))

smat2cMom, adds2cMom = Hamiltonians.build_sparse_matrix_from_LO(Ĥ2cMom,aIni2cMom)
eig2cMom = eigen(Matrix(smat2cMom))

eig2c.values == eig2cMom.values

# figure()
# subplot(211)
# plot(df.df.steps,df.df.norm,label="BHM-Real1D")
# plot(df2c.df.steps,df2c.df.norm,label="BHM2C-Real1D")
# ylabel(L"N_w")
# legend()
# title("M=6, NA=5, NB=1, ta=tb=6, ua=ub=1, v=12")
# subplot(212)
# plot(df.df.steps,df.df.shift,label="BHM-Real1D")
# plot(df2c.df.steps,df2c.df.shift,label="BHM2C-Real1D")
# eigvalues = -4.021502406906465
# eigvalues2c = -3.3252936086867373
# plot(df2c.df.steps,ones(length(df2c.df.steps))*eigvalues, "--g", label="BHM-Real1 exact")
# plot(df2c.df.steps,ones(length(df2c.df.steps))*eigvalues2c, "--b", label="BHM2C-Real1 exact")
# ylabel("shift")
# legend()
