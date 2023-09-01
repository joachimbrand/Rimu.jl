# # Example 4: Impurity Yrast States

# This is an example calculation of the lowest energy eigenstates at given total momentum (yrast states) of a mobile
# impurity coupled with a one-dimensional Bose gas. We will be using MPI parallelisation
# as such calculations are typically expensive. This script is designed to be run effectively
# on a high performance computing (HPC) unit. 

# The aim of this example is to showcase how a two-component Hamiltonian in momentum-space can be set up, 
# as well as how a multi-stage FCIQMC can be run. Furthermore, this momentum-space setup will incur the 
# sign problem, hence the initiator approach for FCIQMC will be used.

# A detailed description of the physical system and the physics behind the model can be found 
# at our published paper (open access) "Polaron-Depleton Transition in the Yrast Excitations of a One-Dimensional 
# Bose Gas with a Mobile Impurity", M. Yang, M. Čufar, E. Pahl, J. Brand, 
# [*Condens. Matter* **7**, 15 (2022)](https://www.mdpi.com/2410-3896/7/1/15).

# A runnable script for this example is located 
# [here](https://github.com/joachimbrand/Rimu.jl/blob/develop/scripts/Impurity-example.jl).
# Run it with `mpirun -np [# of CPUs] julia Impurity-example.jl`.

# ## Initial setup and parameters

# Firstly, we load all needed modules `Rimu` and `Rimu.RMPI` for parallel FCIQMC calculation.

using Rimu
using Rimu.RMPI

# Let's define a function for constructing the starting vector based on
# the total momentum of the coupled system `P`, the number of modes `m` and the 
# number of non-impurity bosons `n`. The maximum allowed total momentum equals to the total
# number of particles including the impurity, hence `n+1`. Apart from the zero and the maximum
# total momentum states, we will have more than one configurations in the starting vector
# to reflect various possible excitation options based on intuitions in physics.
function init_dv(P,m,n)
    aIni = BoseFS2C(BoseFS([n; zeros(Int, m-1)]), BoseFS([1; zeros(Int, m-1)]))
    dv = InitiatorDVec(aIni=>1.0;style=IsDynamicSemistochastic())
    empty!(dv)
    c = (m+1)÷2 # finding the zero-momentum mode
    if P == 0 # zero total momentum
        bfs1 = zeros(Int, m);bfs2 = zeros(Int, m)
        bfs1[c] = n; bfs2[c] = 1
        dv[BoseFS2C(BoseFS(bfs1),BoseFS(bfs2))]+=1.0
    elseif P == (n+1) # maximum total momentum
        bfs1 = zeros(Int, m);bfs2 = zeros(Int, m)
        bfs1[c+1] = n; bfs2[c+1] = 1
        dv[BoseFS2C(BoseFS(bfs1),BoseFS(bfs2))]+=1.0
    else
        bfs1 = zeros(Int, m);bfs2 = zeros(Int, m)
        bfs1[c] = n-(P-1); bfs1[c+1] = P-1; bfs2[c+1] = 1 # move impurity to c+1
        dv[BoseFS2C(BoseFS(bfs1),BoseFS(bfs2))]+=1.0
        
        bfs1 = zeros(Int, m);bfs2 = zeros(Int, m)
        bfs1[c] = n-P; bfs1[c+1] = P; bfs2[c] = 1 # move bosons to c+1
        dv[BoseFS2C(BoseFS(bfs1),BoseFS(bfs2))]+=1.0
        
        bfs1 = zeros(Int, m);bfs2 = zeros(Int, m)
        bfs1[c] = n; bfs2[c+P] = 1 # move impurity to c+P
        dv[BoseFS2C(BoseFS(bfs1),BoseFS(bfs2))]+=1.0
        
        if (n-1) >= P >(n÷2)
            bfs1 = zeros(Int, m);bfs2 = zeros(Int, m)
            bfs1[c] = n-(P+1); bfs1[c+1] = P+1; bfs2[c-1] = 1 # move impurity to c-1 and a boson to c+1
            dv[BoseFS2C(BoseFS(bfs1),BoseFS(bfs2))]+=1.0
        end
    end
    return dv
end

# Note that the `dv` will be constructed with `InitiatorDVec()`, meaning 
# that the initiator-FCIQMC algorithm will be used.

# Now let's first do some MPI sanity checks and print some information:
mpi_barrier() # optional, use for debugging and sanity checks
@info "After barrier 1" mpi_rank() mpi_size() Threads.nthreads()

# Now we specify parameters for constructing a two-component Hamiltonian
P = 3 # total momentum
m = 8 # number of modes
na= 4 # number of non-impurity bosons
γ = 0.2 # boson-boson coupling strength, dimensionless quantity
η = 0.5 # impurity-boson coupling strength, dimensionless quantity
T = m^2/2 # normalised hopping strength
U = m*γ*na/(γ*na/(m*π^2) + 1) # converting γ to U
V = m*η*na/(η*na/(m*π^2) + 1) # converting η to V
# Here we use an initial address `aIni` for constructing the Hamiltonian, but 
# it will not be used in the starting vector.
aIni = BoseFS2C(BoseFS([na; zeros(Int, m-1)]), BoseFS([1; zeros(Int, m-1)]))
ham = BoseHubbardMom1D2C(aIni;ta=T,tb=T,ua=U,v=V,dispersion=continuum_dispersion)



# Now we can setup the Monte Carlo parameters
steps_warmup = 10_000 # number of QMC steps running with a dummy Hamiltonian, see Stage 1
steps_equilibrate = 10_000 # number of QMC steps running with the real `ham`
steps_final = 10_000 # number of QMC steps running with G2 correlators, very slow, be caution!
tw = 1_000 # number of walkers, be sure to use a larger enough number to eliminate biases

# Specifying the shift strategy:
s_strat = DoubleLogUpdateAfterTargetWalkers(targetwalkers = tw)

# Wrapping `dv` for MPI:
dv = MPIData(init_dv(P,m,na));

# Let's have a look of the starting vector, in this particular case, all 4 different ways of 
# distributing total momenta `P` with `init_dv()` are triggered:
@mpi_root @show dv

# ## Stage 1: Running with the "dummy" Hamiltonian

# Here we are constructing a secondary Hamiltonian `ham2` with equal boson-boson and impurity coupling
# strength. We use this Hamiltonian to further generate a batter staring vector. From previous experiences 
# calculating impurity problems, this setup can significantly speed up the convergence and help FCIQMC to 
# sample the important part of the Hilbert space, especially useful when `η` is very small.
η2 = γ
V2 = m*η2*na/(η2*na/(m*π^2) + 1)
ham2 = BoseHubbardMom1D2C(aIni;ta=T,tb=T,ua=U,v=V2,dispersion=continuum_dispersion)

# We use very small time-step size and high starting value of shift
params = RunTillLastStep(step = 0, dτ = 0.00001, laststep = steps_warmup,shift = 200.0)
# As we only use the secondary Hamiltonian `ham2` to generate a staring vector, we don't have to
# save any data in this stage. Progress messages are suppressed with `io=devnull`.
r_strat = ReportDFAndInfo(reporting_interval = 1_000, info_interval = 1_000, writeinfo = is_mpi_root(), io = devnull)

# Now we run FCIQMC with `lomc!()` and track the elapsed time. 
# Both `df` and `state` will be overwritten later with the "real" data.
el = @elapsed df, state = lomc!(ham2, dv; params, s_strat, r_strat,)
@mpi_root @info "Initial fciqmc completed in $(el) seconds."

# ## Stage 2: Running with the real Hamiltonian with replica but no observables

# We are ready to run the real Hamiltonian, here we redefine some variables for saving outputs.
# We save the Monte Carlo data every 1000 steps.
# Progress messages are suppressed with `io=devnull`, for a real job one should remove the 
# line to invoke the default `io` and reenable the output messages.
r_strat = ReportToFile(
    save_if = is_mpi_root(),
    filename = "mpi_df_$(η)_$(P).arrow",
    chunk_size = 1000,
    return_df = true, # change it to `false` when running the real job
    io=devnull # remove this line when running the real job
    )

# We will turn on the replica, but without operators for a fast equilibration.
el2 = @elapsed df, state = lomc!(ham,dv; params, s_strat, r_strat, replica = AllOverlaps(2, nothing), laststep = (steps_equilibrate+steps_warmup))
@mpi_root @info "Replica fciqmc completed in $(el2) seconds."

# ## Stage 3: Running with the real Hamiltonian with replica and observables

# We now at the last stage of the calculation, doing replica FCIQMC with a serious of 
# G2 correlators with distance `d` from `0` to `m`. See [`G2Correlator`](@ref).
# Here we save data every 1000 steps, but using a smaller `chunk_size` like 10 or even 1
# is highly recommended as replica FCIQMC with many observables being calculated are very 
# expensive and you don't want to loose too much data if the job stops before finishes.
# Progress messages are suppressed with `io=devnull`, for a real job one should remove the 
# line to invoke the default `io` and reenable the output messages.
r_strat = ReportToFile(
    save_if = is_mpi_root(),
    filename = "mpi_df_g2_$(η)_$(P).arrow",
    chunk_size = 1000,
    return_df = true, # change it to `false` when running the real job
    io = devnull # remove this line when running the real job
    )

# Setting up a tuple of G2 correlators:
g = Tuple(G2Correlator.(0:m))
# By default, for a two-component system the cross-component G2 operators are set up.
# If you want to calculate the correlations within a single component, `G2Correlator(d,:first)` 
# or `G2Correlator(d,:second)` can be called based on your choice.

# Carry over information from the previous stage and set up a new `QMCState`:
new_state = Rimu.QMCState(
    state.hamiltonian, state.replicas, Ref(Int(state.maxlength)),
    r_strat, state.s_strat, state.τ_strat, state.post_step, AllOverlaps(2, g)
    )
# The final stage 
el3 = @elapsed df2, state2 = lomc!(new_state; laststep = (steps_equilibrate+steps_warmup+steps_final))
@mpi_root @info "Replica fciqmc with G2 completed in $(el3) seconds."
println("MPI run finished!")

# ## Post-calculation analysis

# Typically, one should not include any analyses when using MPI, as they will be calculated multiple
# time unless you put the `@mpi_root` macro everywhere. Even so, all other MPI ranks apart from the root
# will be idling and wasting CPU hours on a HPC unit.
# But here, let's have a look of the calculated G2 correlations:
@mpi_root println("Two-body correlator from 2 replicas:")
@mpi_root for d in 0:m
    r = rayleigh_replica_estimator(df2; op_name = "Op$(d+1)", skip = 5_000)
    println("   G2($d) = $(r.f) ± $(r.σ_f)")
end
# A symmetry between `G2(d)` and `G2(m-d)` can be observed above, which is the expected outcome due 
# the periodic boundary conditions. 

# Finished !

using Test #src
r = rayleigh_replica_estimator(df2; op_name = "Op1", skip = 5_000) #src
@test r.f ≈ 0.6294961872457038 rtol = 0.01 #src