using Rimu
using Rimu.WorkingMemory: InitiatorValue, value
using Test

@testset "InitiatorValue" begin
    iv1 = InitiatorValue{Int}(; safe=1)
    @test value(iv1) === 1

    iv2 = InitiatorValue{Int}(; unsafe=2)
    @test value(iv2) === 0

    iv3 = InitiatorValue{Int}(; initiator=3)
    @test value(iv3) === 3

    @test zero(iv1) == InitiatorValue{Int}()
    @test value(zero(iv1)) == 0
    @test zero(iv1) + iv1 == iv1
    @test zero(iv1) + iv2 == iv2
    @test zero(iv1) + iv3 == iv3
    @test value(iv1 + iv2) == 1
    @test value(iv1 + iv3) == 4
    @test value(iv2 + iv3) == 5
end
