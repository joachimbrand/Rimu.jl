using Rimu
using Test
import Random

using Rimu.DictVectors: FrozenDVec
using OrderedCollections: freeze

@testset "QMCProblem" begin
    h = HubbardReal1D(BoseFS(1,3))
    p = QMCProblem(h)
    @test p.hamiltonian == h
    sp = only(p.initial_shift_parameters)
    @test sp.shift == diagonal_element(h, starting_address(h))
    @test sp.pnorm == walkernumber(only(p.starting_vectors))
    @test p.maxlength == 2 * p.shift_strategy.targetwalkers + 100
    @test Rimu.num_replicas(p) == 1

    simulation = init(p)
    @test simulation.qmc_state.hamiltonian == h

    ps = QMCProblem(h; initial_shift_parameters=sp)
    @test ps.initial_shift_parameters == (sp,)

    p = QMCProblem(h; n_replicas = 3)
    @test Rimu.num_replicas(p) == 3
    dv = p.starting_vectors[1]
    @test pairs(dv) == [starting_address(h) => 1.0]
    @test p.starting_vectors == QMCProblem(h; start_at=dv, n_replicas = 3).starting_vectors
    @test Rimu.num_replicas(init(p)) == 3

    @test_throws ArgumentError QMCProblem(h; start_at=[BoseFS(1, 3), BoseFS(2, 3)])
    @test_throws ArgumentError QMCProblem(h; start_at=[dv, dv, dv])
    p = QMCProblem(h; start_at=[BoseFS(1, 3)=>1, BoseFS(2, 2)=>3])
    @test p.starting_vectors isa Tuple{FrozenDVec}
    @test_throws ArgumentError QMCProblem(h; start_at=(1,2,3))
    @test_throws ArgumentError QMCProblem(h; shift=(1, 2, 3))
    @test QMCProblem(h; shift=2).initial_shift_parameters[1].shift == 2
    @test QMCProblem(h; shift=(2,)).initial_shift_parameters[1].shift == 2
end

@testset "QMCSimulation" begin
    h = HubbardReal1D(BoseFS(1, 3))
    p = QMCProblem(h) # generates random_seed
    @test p.random_seed isa UInt64

    # default gives reproducible random numbers
    sm = init(p) # seeds RNG
    r = rand(Int)
    init(p) # re-seeds RNG with same seed
    @test r == rand(Int)

    # but QMCProblem will re-seed
    Random.seed!(127)
    p = QMCProblem(h)
    sm = init(p)
    r = rand(Int)
    Random.seed!(127)
    p = QMCProblem(h)
    sm = init(p)
    @test r ≠ rand(Int)

    # unless seeding in QMCProblem is disabled
    Random.seed!(127)
    p = QMCProblem(h; seed=false)
    @test isnothing(p.random_seed)
    sm = init(p)
    r = rand(Int)
    Random.seed!(127)
    p = QMCProblem(h; seed=false)
    sm = init(p)
    @test r == rand(Int)

    @test sm.modified[] == false == sm.aborted[] == sm.success[]

end
