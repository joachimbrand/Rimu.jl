using Rimu
using Test
import Random

using Rimu: is_finalized
using Rimu.DictVectors: FrozenDVec
using OrderedCollections: freeze

@testset "ProjectorMonteCarloProblem" begin
    h = HubbardReal1D(BoseFS(1,3))
    p = ProjectorMonteCarloProblem(h; threading=true)
    @test p.hamiltonian == h
    sp = only(p.initial_shift_parameters)
    @test sp.shift == diagonal_element(h, starting_address(h))
    @test sp.pnorm == walkernumber(only(p.starting_vectors))
    @test sp.pnorm isa Float64
    @test p.maxlength == 2 * p.shift_strategy.targetwalkers + 100
    @test Rimu.num_replicas(p) == 1
    @test startswith(sprint(show, p), "ProjectorMonteCarloProblem with 1 replica(s)")

    simulation = init(p)
    @test simulation.state.hamiltonian == h
    @test only(state_vectors(simulation)) isa PDVec

    ps = ProjectorMonteCarloProblem(h; initial_shift_parameters=sp, threading=false)
    @test ps.initial_shift_parameters == (sp,)
    @test only(ps.starting_vectors) isa FrozenDVec
    sm = init(ps)
    @test only(state_vectors(sm)) isa DVec

    p = ProjectorMonteCarloProblem(h; n_replicas = 3, threading=false, initiator=Initiator())
    @test Rimu.num_replicas(p) == 3
    dv = p.starting_vectors[1]
    @test pairs(dv) == [starting_address(h) => 1.0]
    @test p.starting_vectors == ProjectorMonteCarloProblem(h; start_at=dv, n_replicas = 3).starting_vectors
    sm = init(p)
    @test Rimu.num_replicas(sm) == 3
    @test first(state_vectors(sm)) isa InitiatorDVec


    @test_throws ArgumentError ProjectorMonteCarloProblem(h; start_at=[BoseFS(1, 3), BoseFS(2, 3)])
    @test_throws ArgumentError ProjectorMonteCarloProblem(h; start_at=[dv, dv, dv])
    p = ProjectorMonteCarloProblem(h; start_at=[BoseFS(1, 3)=>1, BoseFS(2, 2)=>3])
    @test p.starting_vectors isa Tuple{FrozenDVec}
    @test_throws ArgumentError ProjectorMonteCarloProblem(h; start_at=(1,2,3))
    @test_throws ArgumentError ProjectorMonteCarloProblem(h; shift=(1, 2, 3))
    @test ProjectorMonteCarloProblem(h; shift=2).initial_shift_parameters[1].shift == 2
    @test ProjectorMonteCarloProblem(h; shift=(2,)).initial_shift_parameters[1].shift == 2

    # passing PDVec to ProjectorMonteCarloProblem
    dv = PDVec(starting_address(h)=>3; style=IsDynamicSemistochastic())
    p = ProjectorMonteCarloProblem(h; n_replicas=3, start_at=dv)
    sm = init(p)
    @test first(state_vectors(sm)) == dv
    @test first(state_vectors(sm)) !== dv
    @test first(single_states(sm)).pv !== dv

    # copy_vectors = false
    dv1 = deepcopy(dv)
    dv2 = deepcopy(dv)
    p = ProjectorMonteCarloProblem(h; n_replicas=2, start_at = (dv1, dv2))
    sm = init(p; copy_vectors=false)
    @test state_vectors(sm)[1] === dv1
    @test state_vectors(sm)[2] === dv2
    @test_throws BoundsError sm.state.spectral_states[3]
end

@testset "PMCSimulation" begin
    h = HubbardReal1D(BoseFS(1, 3))
    p = ProjectorMonteCarloProblem(h) # generates random_seed
    @test p.random_seed isa UInt64

    # default gives reproducible random numbers
    sm = init(p) # seeds RNG
    r = rand(Int)
    init(p) # re-seeds RNG with same seed
    @test r == rand(Int)

    # but ProjectorMonteCarloProblem will re-seed
    Random.seed!(127)
    p = ProjectorMonteCarloProblem(h)
    sm = init(p)
    r = rand(Int)
    Random.seed!(127)
    p = ProjectorMonteCarloProblem(h)
    sm = init(p)
    @test r ≠ rand(Int)

    # unless seeding in ProjectorMonteCarloProblem is disabled
    Random.seed!(127)
    p = ProjectorMonteCarloProblem(h; random_seed=false)
    @test isnothing(p.random_seed)
    sm = init(p)
    r = rand(Int)
    Random.seed!(127)
    p = ProjectorMonteCarloProblem(h; random_seed=false)
    sm = init(p)
    @test r == rand(Int)

    # or if the seed is provided
    p = ProjectorMonteCarloProblem(h; random_seed=123)
    @test p.random_seed == 123
    sm = init(p)
    r = rand(Int)
    p = ProjectorMonteCarloProblem(h; random_seed=123)
    sm = init(p)
    @test r == rand(Int)

    @test sm.modified[] == false == sm.aborted[] == sm.success[]
    @test size(DataFrame(sm)) == (0, 0)
end

using Rimu: num_replicas, num_spectral_states
@testset "step! and solve!" begin
    h = HubbardReal1D(BoseFS(1, 3))
    p = ProjectorMonteCarloProblem(h; threading=true)
    sm = init(p)
    @test sm.modified == false == sm.aborted == sm.success
    @test is_finalized(sm.report) == false
    @test sprint(show, sm) == "PMCSimulation with 1 replica(s) and 1 spectral state(s).\n  Algorithm:   FCIQMC()\n  Hamiltonian: HubbardReal1D(BoseFS{4,2}(1, 3); u=1.0, t=1.0)\n  Step:        0 / 100\n  modified = false, aborted = false, success = false"

    @test step!(sm) isa Rimu.PMCSimulation
    @test sm.modified == true
    @test is_finalized(sm.report) == false
    @test size(DataFrame(sm))[1] == sm.state.step[]

    @test solve!(sm) isa Rimu.PMCSimulation
    @test sm.modified == true
    @test sm.success == true
    @test is_finalized(sm.report) == true
    @test size(DataFrame(sm))[1] == sm.state.step[]
    @test num_replicas(sm) == num_replicas(p) == num_replicas(sm.state)
    @test num_spectral_states(sm) == num_spectral_states(p) == num_spectral_states(sm.state)
    @test size(state_vectors(sm)) == (num_replicas(sm), num_spectral_states(sm))
    @test size(single_states(sm)) == (num_replicas(sm), num_spectral_states(sm))

    df, state = sm # deconstruction for backward compatibility
    @test df == DataFrame(sm) == sm.df
    @test state == sm.state

    # Tables.jl interface
    @test Tables.istable(sm)
    @test map(NamedTuple, Tables.rows(sm)) == map(NamedTuple, Tables.rows(df))

    # continue simulation
    @test sm.state.step[] == 100
    solve!(sm; last_step=200)
    @test sm.state.step[] == 200
    @test sm.success == true == parse(Bool, (Rimu.get_metadata(sm.report, "success")))

    # time out
    p = ProjectorMonteCarloProblem(h; last_step=500, walltime=1e-3)
    sm = init(p)
    @test_logs (:warn, Regex("(Walltime)")) solve!(sm)
    @test sm.success == false
    @test sm.aborted == true
    @test sm.message == "Walltime limit reached."

    sm2 = solve!(sm; walltime=1.0)
    @test sm2 === sm
    @test sm.success == true
    @test sm.state.step[] == 500 == size(sm.df)[1]

    # continue simulation and change strategies
    sm3 = solve!(sm;
        last_step = 600,
        post_step_strategy = Rimu.Timer(),
        metadata = Dict(:test => 1)
    )
    @test sm3 === sm
    @test sm.success == true
    @test sm.state.step[] == 600
    @test size(sm.df)[1] == 100 # the report was emptied
    @test parse(Int, Rimu.get_metadata(sm.report, "test")) == 1
    @test Rimu.get_metadata(sm.report, "post_step_strategy") == "(Rimu.Timer(),)"

    # continue simulation and change replica strategy
    @test_throws ArgumentError solve!(sm; replica_strategy = NoStats(3))

    p = ProjectorMonteCarloProblem(h; last_step=100, replica_strategy=NoStats(3))
    sm = init(p)
    @test solve!(sm) === sm
    @test solve!(sm; last_step=200, replica_strategy=AllOverlaps(3)) === sm
    @test size(sm.df)[1] == 100 # the report was emptied
    @test solve!(sm; last_step=300, reporting_strategy=ReportDFAndInfo()) === sm
    @test size(sm.df)[1] == 200 # the report was not emptied
end
