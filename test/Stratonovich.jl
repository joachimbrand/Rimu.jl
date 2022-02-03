using Rimu
using Test
using Rimu.Stratonovich
using Rimu.StochasticStyles: diagonal_step!, compress!, ThresholdCompression
using Rimu.DictVectors: deposit!, value, InitiatorValue

@testset "Stratonovich" begin
    add = fs"|1 1 1 1⟩"
    svec = InitiatorDVec(
        add => 2;
        style=IsDynamicSemistochastic(),
        initiator = StratonovichCorrection()
    )
    deposit!(svec, add, 3.3, (fs"|1 0 2 1⟩", 1.0))
    @test svec[fs"|1 1 1 1⟩"] == 5.3
    @test_throws AssertionError deposit!(svec, fs"|1 1 1 1⟩", 3.3, (fs"|1 1 1 1⟩", 1.0))

    H = HubbardReal1D(add)
    dv = DVec(svec) # copy into regular DVec
    diagonal_step!(svec,H, add, 2.0, 0.01, 2.0)
    val = storage(svec)[add]
    @test value(svec.initiator, val) ==  svec[add] == 7.34
    @test val.unsafe == 0.02
    # check that we get the same as with a DVec
    diagonal_step!(dv, H, add, 2.0, 0.01, 2.0)
    @test dv == svec

    compress!(ThresholdCompression(7.345),svec)
    # Stratonovich corrected value (7.35) is larger than threshold; no compression
    @test storage(svec)[add] == InitiatorValue{Float64}(7.34, 0.02, 0.0)
    compress!(ThresholdCompression(7.355),svec)
    # and now it is smaller, so compression is triggered and `unsafe` removed
    val = storage(svec)[add]
    @test iszero(val.unsafe)

    # test that `compress!` removes zero entries
    storage(svec)[add] = 0
    @test length(svec) == 1
    compress!(ThresholdCompression(1),svec)
    @test length(svec) == 0    
end
