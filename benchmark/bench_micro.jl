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
suite["diagME"] = BenchmarkGroup()
suite["hop"] = BenchmarkGroup()
suite["kinetic_energy"] = BenchmarkGroup()
suite["fciqmc_col"] = BenchmarkGroup()
for k in ("diagME", "hop", "kinetic_energy", "fciqmc_col")
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

suite["diagME"]["16bit"]["Real1D"] = @benchmarkable diagME($Hr1, $addr1)
suite["hop"]["16bit"]["Real1D"] = @benchmarkable hop($Hr1, $addr1, 10)

suite["diagME"]["16bit"]["Mom1D"] = @benchmarkable diagME($Hm1, $addr1)
suite["hop"]["16bit"]["Mom1D"] = @benchmarkable hop($Hm1, $addr1, 10)
suite["kinetic_energy"]["16bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm1, $addr1)

Random.seed!(1337)
addr2 = rand(BoseFS{32,32})
Hr2 = BoseHubbardReal1D(addr2; u=6.0, t=1.0)
Hm2 = HubbardMom1D(addr2; u=6.0, t=1.0)
suite["numberoccupiedsites"]["64bit"] = @benchmarkable numberoccupiedsites($addr2)

suite["diagME"]["64bit"]["Real1D"] = @benchmarkable diagME($Hr2, $addr2)
suite["hop"]["64bit"]["Real1D"] = @benchmarkable hop($Hr2, $addr2, 10)

suite["diagME"]["64bit"]["Mom1D"] = @benchmarkable diagME($Hm2, $addr2)
suite["hop"]["64bit"]["Mom1D"] = @benchmarkable hop($Hm2, $addr2, 10)
suite["kinetic_energy"]["64bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm2, $addr2)

Random.seed!(1337)
addr3 = rand(BoseFS{64,64})
Hr3 = BoseHubbardReal1D(addr3; u=6.0, t=1.0)
Hm3 = HubbardMom1D(addr3; u=6.0, t=1.0)
suite["numberoccupiedsites"]["128bit"] = @benchmarkable numberoccupiedsites($addr3)

suite["diagME"]["128bit"]["Real1D"] = @benchmarkable diagME($Hr3, $addr3)
suite["hop"]["128bit"]["Real1D"] = @benchmarkable hop($Hr3, $addr3, 10)

suite["diagME"]["128bit"]["Mom1D"] = @benchmarkable diagME($Hm3, $addr3)
suite["hop"]["128bit"]["Mom1D"] = @benchmarkable hop($Hm3, $addr3, 10)
suite["kinetic_energy"]["128bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm3, $addr3)

Random.seed!(1337)
addr4 = rand(BoseFS{256,256})
Hr4 = BoseHubbardReal1D(addr4; u=6.0, t=1.0)
Hm4 = HubbardMom1D(addr4; u=6.0, t=1.0)
suite["numberoccupiedsites"]["256bit"] = @benchmarkable numberoccupiedsites($addr4)

suite["diagME"]["256bit"]["Real1D"] = @benchmarkable diagME($Hr4, $addr4)
suite["hop"]["256bit"]["Real1D"] = @benchmarkable hop($Hr4, $addr4, 10)

suite["diagME"]["256bit"]["Mom1D"] = @benchmarkable diagME($Hm4, $addr4)
suite["hop"]["256bit"]["Mom1D"] = @benchmarkable hop($Hm4, $addr4, 10)
suite["kinetic_energy"]["256bit"]["Mom1D"] = @benchmarkable kinetic_energy($Hm4, $addr4)

end
BenchMicro.suite
