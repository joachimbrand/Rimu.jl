# script for testing MPI functionality
import MPI
using Rimu
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

MPI.Finalize()
end # main()

main()
