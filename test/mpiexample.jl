# script for testing MPI functionality
using Rimu
import MPI
using LinearAlgebra

# include("../src/mpi_helpers.jl")
# function Rimu.fciqmc_step!(Ĥ, dv::MPIData{D,S}, shift, dτ, w::D) where {D,S}
#     v = localpart(dv)
#     @assert w ≢ v "`w` and `v` must not be the same object"
#     empty!(w)
#     stats = zeros(valtype(v), 5) # pre-allocate array for stats
#     for (add, num) in pairs(v)
#         res = Rimu.fciqmc_col!(w, Ĥ, add, num, shift, dτ)
#         ismissing(res) || (stats .+= res) # just add all stats together
#     end
#     sort_into_targets!(dv, w)
#     MPI.Allreduce!(stats, +, dv.comm) # add stats of all ranks
#     return dv, w, stats
#     # returns the structure with the correctly distributed end
#     # result `dv` and cumulative `stats` as an array on all ranks
#     # stats == (spawns, deaths, clones, antiparticles, annihilations)
# end # fciqmc_step!

function mytypecheck(a::T) where T <: Union{Int,Nothing}
    if T == Nothing
        return nothing
    else
        return a^2
    end
end

function mytypecheck2(a::T) where T <: Union{Real,Nothing}
    T == Nothing ? nothing : a^2
end
function mytypecheck4(a::T) where T
    T == Nothing && begin
        println("nothing detected")
        return nothing
    end
    return a^2
end

function fciqmc_for_bm(vs, ham, w, steps, s, comm)
    MPI.Barrier(comm)
    pa = RunTillLastStep(step = 0, laststep = steps)
    return fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), w)
end

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
aIni = nearUniform(Ĥ)
svec = DVec(Dict(aIni => 2), Ĥ(:dim)÷np) # our buffer for the state vector

aIrank = targetrank(aIni, np)
# aIrank ≠ id && empty!(svec) # empty on all except correct rank

w = similar(svec) # this one is a buffer for working
dv = mpi_one_sided(svec) # wrap state vector with mpi strategy
# dv = mpi_default(svec) # wrap state vector with mpi strategy
Rimu.ConsistentRNG.seedCRNG!(17+id) # for now seed the RNG with the rank of the process
cstats = zeros(Int,5)

id == dv.root && println("Initial state:")

len = length(localpart(dv))
lenarray = MPI.Gather(len, dv.root, comm)
id == dv.root && println("Number of configurations on ranks: ", lenarray, ", sum = ", sum(lenarray))
nwalkers = sum(abs.(values(dv.data)))
nw_array = MPI.Gather(nwalkers, dv.root, comm)
id == dv.root && println("Number of walkers on ranks: ", nw_array, ", sum = ", sum(nw_array))

for n in 1:30
    dv, w, stats = Rimu.fciqmc_step!(Ĥ, dv, 0.0, 0.01, w)
    id == dv.root && println("Step $n, stats = ",stats)
    cstats .+= stats
    id == dv.root && println("Step $n, cstats = ",cstats)

    len = length(localpart(dv))
    lenarray = MPI.Gather(len, dv.root, comm)
    id == dv.root && println("Number of configurations on ranks: ", lenarray, ", sum = ", sum(lenarray))
    tlen = length(dv)
    id == dv.root && println("length(dv) = ",tlen)
    nwalkers = sum(abs.(values(dv.data)))
    nw_array = MPI.Gather(nwalkers, dv.root, comm)
    id == dv.root && println("Number of walkers on ranks: ", nw_array, ", sum = ", sum(nw_array))
    n1 = norm(dv,1)
    n2 = norm(dv,2)
    nInf = norm(dv,Inf)
    id == dv.root && println("1-norm = $n1, 2-norm = $n2, Inf-norm = $nInf")
end

@time free(dv) # MPI sync

vs = DVec(Dict(aIni => 2), Ĥ(:dim)÷np) # our buffer for the state vector
s = LogUpdateAfterTargetWalkers(targetwalkers = 1000)
MPI.Barrier(comm)
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

# run fciqmc on a process independently without MPI communication
dvs = similar(vs, Ĥ(:dim)) # increase space
pa = RunTillLastStep(laststep = 1)
et = @elapsed df = fciqmc!(dvs, pa, Ĥ, s)
id==0 && println("serial fciqmc compiled in $et seconds")
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

dvs = copy(vs) # just a copy
pa = RunTillLastStep(laststep = 100)
et = @elapsed df = fciqmc!(dvs, pa, Ĥ, s)
id==0 && println("serial fciqmc finished in $et seconds")
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

id==0 && println(df)

w = similar(vs)
steps = 100
ets = zeros(Float64,5)
for i in 1:length(ets)
    ets[i] = @elapsed fciqmc_for_bm(dvs, Ĥ, w, steps, s, comm)
end
id==0 && println("serial fciqmc benchmarks in seconds: ", ets)

println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

# now run with parallel communications
id==0 && println("starting mpi_one_sided fciqmc")
dvs = mpi_one_sided(copy(vs)) # mpi wrapping
pa = RunTillLastStep(laststep = 1)
et = @elapsed df = fciqmc!(dvs, pa, Ĥ, s)
dvs.isroot && println("parallel fciqmc compiled in $et seconds")
@time free(dvs) # MPI sync

dvs = mpi_one_sided(copy(vs)) # mpi wrapping
pa = RunTillLastStep(laststep = 100)
et = @elapsed df = fciqmc!(dvs, pa, Ĥ, s)
dvs.isroot && println("parallel fciqmc finished in $et seconds")
# dvs.isroot && println(df)
MPI.Barrier(comm)
dvs.isroot && println("starting benchmarks")
pa = RunTillLastStep(step = 0, laststep = 20)
ts = 1 # @elapsed
fciqmc!(dvs, pa, Ĥ, s, EveryTimeStep(), ConstantTimeStep(), w)
dvs.isroot && println("r ", ts)
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

ts1 = 2 # @elapsed
df = fciqmc_for_bm(dvs, Ĥ, w, 100, s, comm)
dvs.isroot && println("r1 ", ts1)
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

ts2 = 3 # @elapsed
df = fciqmc_for_bm(dvs, Ĥ, w, 100, s, comm)
dvs.isroot && println("r2 ", ts2)
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

dvs.isroot && println(df)
dvs.isroot && println("starting benchmarks")
println("$(id): arrived at barrier; main")
MPI.Barrier(comm)

# w = similar(vs)
steps = 100
ets = zeros(Float64,5)
for i in 1:length(ets)
    dvs.isroot && println("step $i")
    ets[i] = @elapsed fciqmc_for_bm(dvs, Ĥ, w, 100, s, comm)
end
id==0 && println("mpi one sided fciqmc benchmarks in seconds: ", ets)

@time free(dvs) # MPI sync


# now run with parallel communications
id==0 && println("starting mpi_default fciqmc")
dvs = mpi_default(copy(vs)) # mpi wrapping
pa = RunTillLastStep(laststep = 1)
et = @elapsed df = fciqmc!(dvs, pa, Ĥ, s)
dvs.isroot && println("mpi_default fciqmc compiled in $et seconds")
dvs = mpi_default(copy(vs)) # mpi wrapping
pa = RunTillLastStep(laststep = 100)
et = @elapsed df = fciqmc!(dvs, pa, Ĥ, s)
dvs.isroot && println("mpi_default fciqmc finished in $et seconds")
dvs.isroot && println(df)

ets = zeros(Float64,5)
for i in 1:length(ets)
    ets[i] = @elapsed fciqmc_for_bm(dvs, Ĥ, w, steps, s, comm)
end
id==0 && println("mpi_default fciqmc benchmarks in seconds: ", ets)



MPI.Barrier(comm)


MPI.Finalize()
end # main()

main()
