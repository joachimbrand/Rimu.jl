using Rimu
using Test

@testset "extension not loaded" begin
    # Can only test this when KrylovKit is not loaded
    ext = Base.get_extension(Rimu, :KrylovKitExt)
    if ext === nothing
        @test_throws ErrorException KrylovKitMatrix()
    end
end

using KrylovKit

@testset "ExactDiagonalizationProblem" begin
    p = ExactDiagonalizationProblem(HubbardReal1D(BoseFS(1,2,3)); which=:LM)
    @test eval(Meta.parse(repr(p))) == p
    solver = init(p, KrylovKitMatrix(); howmany=2)
    @test dimension(solver.basissetrep) == dimension(p.h) == size(solver.basissetrep.sm)[1]

    p2 = ExactDiagonalizationProblem(
        HubbardReal1D(BoseFS(1, 2, 3)), DVec(BoseFS(1, 2, 3) => 2.3); which=:LM
    )
    @test eval(Meta.parse(repr(p2))) == p2
    s2 = init(p2, KrylovKitMatrix())
end
