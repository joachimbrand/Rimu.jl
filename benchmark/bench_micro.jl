"""
Microbenchmarks.
"""
module BenchMicro

using Rimu
using Rimu.Hamiltonians: kinetic_energy, numberoccupiedsites
using BenchmarkTools
using Random
using StaticArrays

suite = BenchmarkGroup()
suite["diagonal_element"] = BenchmarkGroup()
suite["get_offdiagonal"] = BenchmarkGroup()
suite["kinetic_energy"] = BenchmarkGroup()
suite["fciqmc_col"] = BenchmarkGroup()
for k in ("diagonal_element", "get_offdiagonal", "kinetic_energy", "fciqmc_col")
    for bits in ("16bit", "64bit", "128bit", "256bit")
        suite[k][bits] = BenchmarkGroup()
        for type in ("Real1D", "Mom1D")
            suite[k][bits][type] = BenchmarkGroup()
        end
    end
end
suite["numberoccupiedsites"] = BenchmarkGroup()
for bits in ("16bit", "64bit", "128bit", "256bit")
    suite["numberoccupiedsites"][bits] = BenchmarkGroup()
end

"""
Generate random address.
"""
function Base.rand(::Type{<:BoseFS{N,M}}) where {N,M}
    onr = zeros(MVector{M,Int})
    for _ in 1:N
        onr[rand(1:M)] += 1
    end
    return BoseFS{N,M}(SVector(onr))
end

Random.seed!(1337)
addr1 = rand(BoseFS{8,8})
Hr1 = BoseHubbardReal1D(addr1; u=6.0, t=1.0)
Hm1 = HubbardMom1D(addr1; u=6.0, t=1.0)
suite["numberoccupiedsites"]["16bit"] = @benchmarkable numberoccupiedsites($addr1)

suite["diagonal_element"]["16bit"]["Real1D"] = @benchmarkable diagonal_element($Hr1, $addr1)
suite["get_offdiagonal"]["16bit"]["Real1D"] = @benchmarkable get_offdiagonal($Hr1, $addr1, 10)

suite["diagonal_element"]["16bit"]["Mom1D"] = @benchmarkable diagonal_element($Hm1, $addr1)
suite["get_offdiagonal"]["16bit"]["Mom1D"] = @benchmarkable get_offdiagonal($Hm1, $addr1, 10)
suite["kinetic_energy"]["16bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm1, $addr1)

Random.seed!(1337)
addr2 = rand(BoseFS{32,32})
Hr2 = BoseHubbardReal1D(addr2; u=6.0, t=1.0)
Hm2 = HubbardMom1D(addr2; u=6.0, t=1.0)
suite["numberoccupiedsites"]["64bit"] = @benchmarkable numberoccupiedsites($addr2)

suite["diagonal_element"]["64bit"]["Real1D"] = @benchmarkable diagonal_element($Hr2, $addr2)
suite["get_offdiagonal"]["64bit"]["Real1D"] = @benchmarkable get_offdiagonal($Hr2, $addr2, 10)

suite["diagonal_element"]["64bit"]["Mom1D"] = @benchmarkable diagonal_element($Hm2, $addr2)
suite["get_offdiagonal"]["64bit"]["Mom1D"] = @benchmarkable get_offdiagonal($Hm2, $addr2, 10)
suite["kinetic_energy"]["64bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm2, $addr2)

Random.seed!(1337)
addr3 = rand(BoseFS{64,64})
Hr3 = BoseHubbardReal1D(addr3; u=6.0, t=1.0)
Hm3 = HubbardMom1D(addr3; u=6.0, t=1.0)
suite["numberoccupiedsites"]["128bit"] = @benchmarkable numberoccupiedsites($addr3)

suite["diagonal_element"]["128bit"]["Real1D"] = @benchmarkable diagonal_element($Hr3, $addr3)
suite["get_offdiagonal"]["128bit"]["Real1D"] = @benchmarkable get_offdiagonal($Hr3, $addr3, 10)

suite["diagonal_element"]["128bit"]["Mom1D"] = @benchmarkable diagonal_element($Hm3, $addr3)
suite["get_offdiagonal"]["128bit"]["Mom1D"] = @benchmarkable get_offdiagonal($Hm3, $addr3, 10)
suite["kinetic_energy"]["128bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm3, $addr3)

Random.seed!(1337)
addr4 = rand(BoseFS{256,256})
Hr4 = BoseHubbardReal1D(addr4; u=6.0, t=1.0)
Hm4 = HubbardMom1D(addr4; u=6.0, t=1.0)
suite["numberoccupiedsites"]["256bit"] = @benchmarkable numberoccupiedsites($addr4)

suite["diagonal_element"]["256bit"]["Real1D"] = @benchmarkable diagonal_element($Hr4, $addr4)
suite["get_offdiagonal"]["256bit"]["Real1D"] = @benchmarkable get_offdiagonal($Hr4, $addr4, 10)

suite["diagonal_element"]["256bit"]["Mom1D"] = @benchmarkable diagonal_element($Hm4, $addr4)
suite["get_offdiagonal"]["256bit"]["Mom1D"] = @benchmarkable get_offdiagonal($Hm4, $addr4, 10)
suite["kinetic_energy"]["256bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm4, $addr4)

end
BenchMicro.suite
