using KrylovKit
using Rimu
using Test

@testset "Krylov eigsolve with BoseFS{6,6}" begin
    ham = BoseHubbardReal1D(nearUniform(BoseFS{6,6}); u=6.0, t=1.0)

    a_init = nearUniform(ham)
    c_init = DVec(a_init => 1.0)

    all_results = eigsolve(ham, c_init, 1, :SR; issymmetric = true)
    energy = all_results[1][1]

    @test energy ≈ -4.0215 atol=0.0001
end
