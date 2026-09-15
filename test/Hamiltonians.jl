using KrylovKit
using LinearAlgebra
using Random
using Rimu
using Test
using DataFrames
using Suppressor
using StaticArrays
using Rimu.Hamiltonians: TransformUndoer, AbstractOffdiagonals, ScaledOrShiftedHamiltonian
using Rimu.InterfaceTests: test_observable_interface, test_operator_interface,
    test_hamiltonian_interface, test_hamiltonian_structure
using Rimu.Interfaces: LOStructure, IsHermitian, IsDiagonal, AdjointKnown,
    AdjointUnknown

function exact_energy(ham)
    dv = DVec(starting_address(ham) => 1.0)
    all_results = eigsolve(ham, dv, 1, :SR; issymmetric = LOStructure(ham) == IsHermitian())
    return all_results[1][1]
end

@testset "Hamiltonian interface tests" begin
    for H in [
        HubbardReal1D(BoseFS((1, 2, 3, 4)); u=1.0, t=2.0),
        HubbardReal1D(BoseFS(1, 2, 3, 4);t=1.0im),
        HubbardReal1D(BoseFS(1, 2, 3, 4);u=1.0im),
        HubbardReal1DEP(BoseFS((1, 2, 3, 4)); u=1.0, t=2.0, v_ho=3.0),
        HubbardReal1DEP(BoseFS(1, 2, 3, 4); t=1.0im),
        HubbardReal1DEP(BoseFS(1, 2, 3, 4); u=1.0im),
        HubbardMom1D(BoseFS((6, 0, 0, 4)); t=1.0, u=0.5),
        HubbardMom1D(BoseFS{missing}(6, 0, 0, 4); t=1.0, u=0.5),
        HubbardMom1D(BoseFS{missing}(6, 0, 0, 4); t=1.0, u=0.5),
        HubbardMom1D(BoseFS((6, 0, 0, 4)); t=1.0, u=0.5 + im),
        ExtendedHubbardReal1D(BoseFS((1, 0, 0, 0, 1)); u=1.0, v=2.0, t=3.0),
        ExtendedHubbardReal1D(BoseFS(1, 0, 2, 1); u=1 + 0.5im),
        ExtendedHubbardReal1D(BoseFS(1, 0, 2, 1); t=1 + 0.5im),
        ExtendedHubbardReal1D(BoseFS(1, 0, 2, 1); t=2.0, power=3),
        ExtendedHubbardMom1D(BoseFS((1, 0, 0, 0, 1)); u=1.0, v=2.0, t=3.0),
        ExtendedHubbardMom1D(BoseFS(1, 0, 2, 1); u=1 + 0.5im),
        ExtendedHubbardMom1D(BoseFS(1, 0, 2, 1); t=1 + 0.5im),
        ExtendedHubbardMom1D(BoseFS{missing}(1,2,0,0); u=1.0, v=2.0, t=3.0),
        ExtendedHubbardMom1D(BoseFS{missing}(1,2,0,0); u=1.0, v=2.0, t=3.0),
        ExtendedHubbardMom1D(FermiFS(1,1,0,0); u=1.0, v=2.0, t=3.0),
        HubbardRealSpace(BoseFS((1, 2, 3)); u=[1], t=[3], w=[1]),
        HubbardRealSpace(FermiFS((1, 1, 1, 1, 1, 0, 0, 0)); u=[0], t=[3]),
        HubbardRealSpace(FermiFS((1, 1, 1, 1, 1, 0, 0, 0)); u=[0], t=[3*im]),
        HubbardRealSpace(
            CompositeFS(
                FermiFS((1, 1, 1, 1, 1, 0, 0, 0)),
                FermiFS((1, 1, 1, 1, 0, 0, 0, 0)),
            ); t=[1, 2], u=[0 3; 3 0], w=[1 0.5; 0.5 1]
        ),
        HubbardRealSpace(
            BoseFS((1, 2, 3, 4, 5, 6));
            potential=reshape(float.([1, 2, 3, 4, 5, 5]), (6, 1))
        ),
        GutzwillerSampling(HubbardReal1D(BoseFS((1, 2, 3)); u=6 + 2im); g=0.3),
        GutzwillerSampling(Transcorrelated1D(FermiFS2C((0, 0, 1, 1), (0, 1, 1, 0))); g=0.1),

        GuidingVectorSampling(HubbardReal1D(BoseFS(1, 2, 3); u=6 + 2im), DVec(BoseFS(1,2,3) => 1.0)),
        GuidingVectorSampling(Transcorrelated1D(FermiFS2C((0, 0, 1, 1), (0, 1, 1, 0))), DVec(FermiFS2C((0, 0, 1, 1), (0, 1, 1, 0)) => 1.0)),

        MatrixHamiltonian(Float64[1 2; 2 0]),
        GutzwillerSampling(MatrixHamiltonian([1.0 2.0; 2.0 0.0]); g=0.3),
        TransformUndoer(
            GutzwillerSampling(MatrixHamiltonian([1.0 2.0; 2.0 0.0]); g=0.3)
        ),
        Transcorrelated1D(FermiFS2C((0, 0, 1, 1, 0), (0, 1, 1, 0, 0)); t=2),
        Transcorrelated1D(CompositeFS(FermiFS((0, 0, 1, 0)), FermiFS((0, 1, 1, 0))); v=3, v_ho=1),
        HubbardMom1DEP(BoseFS((0, 0, 5, 0, 0))),
        HubbardMom1DEP(CompositeFS(FermiFS((0, 1, 1, 0, 0)), FermiFS((0, 0, 1, 0, 0))), v_ho=5),
        ParitySymmetry(HubbardRealSpace(CompositeFS(BoseFS((1, 2, 0)), FermiFS((0, 1, 0))))),
        TimeReversalSymmetry(HubbardMom1D(FermiFS2C((1, 0, 1), (0, 1, 1)))),
        TimeReversalSymmetry(HubbardMom1D(FermiFS2C{missing}((1, 0, 1), (0, 1, 1)))),
        Stoquastic(HubbardMom1D(BoseFS((0, 5, 0)))),
        momentum(HubbardMom1D(BoseFS((0, 5, 0)))),
        HOCartesianContactInteractions(BoseFS((2, 0, 0, 0))),
        HOCartesianEnergyConservedPerDim(BoseFS((2, 0, 0, 0))),
        HOCartesianCentralImpurity(BoseFS((1, 0, 0, 0, 0))),
        FroehlichPolaron1D(BoseFS{missing}(1, 1, 1)),
        FroehlichPolaron1D(BoseFS{missing}(1, 1, 1); momentum_cutoff=10.0),
        FroehlichPolaron1D{Float32}(BoseFS{missing}(1, 1, 1); momentum_cutoff=10.0),
        momentum(HubbardMom1D(BoseFS(0, 1, 5, 1, 0))),
        # HamiltonianProduct
        HubbardReal1D(BoseFS(2,0,0); u=1.0im) * ExtendedHubbardReal1D(BoseFS(2,0,0)),
        FroehlichPolaron(BoseFS{missing}(0,0,0,0)),
        FroehlichPolaron(BoseFS{missing}(0, 0, 0, 0); twist=[0.1]),
        # HamiltonianSum
        HubbardReal1D(BoseFS(2,0,0); u=1.0im) + ExtendedHubbardReal1D(BoseFS(2,0,0)),
        # FCIQMC transition operator
        (@inferred I + 0.01 * (-5.0*I + HubbardRealSpace(BoseFS(1,1,1,1)))),
        2 * HubbardReal1D(BoseFS(2, 0, 0); u=1.0im), # scale
        2 * HubbardReal1D(BoseFS(2, 0, 0)), # scale with real factor
        HubbardReal1D(BoseFS(2, 0, 0); u=1.0im) + 2I, # ScaledOrShiftedHamiltonian
        2 * (HubbardReal1D(BoseFS(2, 0, 0)) + 3.0I), # scaled and shifted
        3.0I - HubbardReal1D(BoseFS(2, 0, 0)), # subtract a Hamiltonian
        - HubbardReal1D(BoseFS(2, 0, 0)), # unary minus
        (2 + 2im) * (HubbardReal1D(BoseFS(2, 0, 0)) + (3.0 + 1.0im)I), # complex
        ]
        test_hamiltonian_interface(H)
        # Check that the result of show can be pasted into the REPL. Does not work with
        # GuidingVectorSampling because it includes a DVec.
        if !(H isa GuidingVectorSampling)
            @test eval(Meta.parse(repr(H))) == H
        end
    end
end

@testset "Operator interface test" begin
    # this is only needed for AbstractOperators that are not AbstractHamiltonians
    # and are not tested in the Hamiltonian interface tests
    for (op, addr) in [
        (G2RealSpace(CubicGrid((2, 2), (true, true)), 1, 1), BoseFS{10,4}(1, 2, 3, 4)),
        (G2RealSpace(CubicGrid(4); sum_components=true),
            CompositeFS(BoseFS(4, 1 => 1), BoseFS(4, 1 => 1), BoseFS(4, 1 => 1))),
        (ParticleNumberOperator(), BoseFS(1, 2, 3)),
        (ParticleNumberOperator(), FermiFS2C((1, 0, 1), (0, 1, 1))),
        (G2RealCorrelator(3), FermiFS2C((1, 0, 1, 1), (0, 1, 1, 0))),
        (SuperfluidCorrelator(3), BoseFS(1, 2, 3, 1)),
        (StringCorrelator(3), BoseFS(1, 0, 3, 1)),
        (DensityMatrixDiagonal(3), FermiFS(1, 0, 1)),
        (SingleParticleExcitation(2, 3), BoseFS(1, 2, 3, 4)),
        (TwoParticleExcitation(3, 2, 1, 4), BoseFS(1, 2, 3, 4)),
        (Momentum(1), BoseFS(1, 2, 3, 4)),
        (G2MomCorrelator(3), BoseFS(1, 2, 0, 3, 0, 4, 0, 1)),
        (IdentityOperator(), BoseFS(1, 2, 0, 3, 0, 4, 0, 1)),
    ]
        test_operator_interface(op, addr)
        # Check that the result of show can be pasted into the REPL
        @test eval(Meta.parse(repr(op))) == op
    end
end

@testset "Observable interface test" begin
    for (op, addr) in [
        (SignCorrelator(), BoseFS(1, 2, 0, 3, 0, 4, 0, 1)),
        (SignCorrelator{Float64}(), FermiFS(1, 1, 0, 1, 0, 1, 0, 1)),
    ]
        test_observable_interface(op, addr)
        @test eval(Meta.parse(repr(op))) == op
    end
end

using Rimu.Hamiltonians: momentum_transfer_excitation

@testset "momentum_transfer_excitation" begin
    @testset "BoseFS" begin
        add1 = BoseFS((0,1,1,0))
        add2 = BoseFS((1,0,0,1))
        for i in 1:4
            ex = momentum_transfer_excitation(add1, i, occupied_mode_map(add1); fold=true)
            @test ex[1] == add2
            @test ex[2] == 1

            ex = momentum_transfer_excitation(add1, i, occupied_mode_map(add1); fold=false)
            @test ex[1] == add2
            @test ex[2] == 1
        end

        add3 = BoseFS((1,1,0,0))
        for i in 1:4
            ex = momentum_transfer_excitation(add3, i, occupied_mode_map(add3); fold=true)
            @test ex[2] == 1

            ex = momentum_transfer_excitation(add3, i, occupied_mode_map(add3); fold=false)
            @test ex[2] == 0
        end

        add4 = BoseFS((0,3,0))
        add5 = BoseFS((1,1,1))
        for i in 1:2
            ex = momentum_transfer_excitation(add4, i, occupied_mode_map(add4); fold=false)
            @test ex[1] == add5
            @test ex[2] ≈ √6

            ex = momentum_transfer_excitation(add4, i, occupied_mode_map(add4); fold=true)
            @test ex[1] == add5
            @test ex[2] ≈ √6
        end
    end
    @testset "FermiFS" begin
        add1 = FermiFS((0,0,1,0))
        add2 = FermiFS((0,1,0,0))
        occ1 = occupied_mode_map(add1)
        occ2 = occupied_mode_map(add2)
        for i in 1:3
            ex = momentum_transfer_excitation(add1, add2, i, occ1, occ2; fold=true)
            @test ex[3] == 1

            ex = momentum_transfer_excitation(add1, add2, i, occ1, occ2; fold=false)
            @test ex[3] == 1
        end

        add3 = FermiFS((1,0,0,0))
        add4 = FermiFS((0,1,0,0))
        occ3 = occupied_mode_map(add3)
        occ4 = occupied_mode_map(add4)
        for i in 1:3
            ex = momentum_transfer_excitation(add3, add4, i, occ3, occ4; fold=true)
            @test ex[3] == 1
        end
        num_nonzero = 0
        for i in 1:3
            ex = momentum_transfer_excitation(add3, add4, i, occ3, occ4; fold=false)
            num_nonzero += ex[3] == 1
        end
        @test num_nonzero == 1
    end
end

using Rimu.Hamiltonians: Directions, Displacements

@testset "CubicGrid" begin
    @testset "construtors and basic properties" begin
        @test PeriodicBoundaries(3, 3) == CubicGrid(3, 3)
        @test HardwallBoundaries(3, 4, 5) == CubicGrid((3, 4, 5), (false, false, false))
        @test LadderBoundaries(2, 3, 4) == CubicGrid((2, 3, 4), (false, true, true))

        for (dims, fold) in (
            ((4,), (false,)), ((2, 5), (true, false)), ((5, 6, 7), (true, true, false))
        )
            geom = CubicGrid(dims, fold)
            @test size(geom) == dims
            @test length(geom) == prod(dims)
            @test Rimu.Hamiltonians.fold(geom) == fold
            @test eval(Meta.parse(repr(geom))) == geom
            @test dimension(geom) == length(size(geom))
        end
    end

    @testset "getindex" begin
        g = CubicGrid((2,3,4), (false,true,false))
        for i in 1:length(g)
            v = SVector(Tuple(CartesianIndices((2,3,4))[i])...)
            @test g[i] == v
            @test g[v] == i
        end

        @test g[SVector(3, 1, 1)] == 0
        @test g[SVector(1,4,1)] == 1
        @test g[SVector(2,0,4)] == 24
        @test g[SVector(2,3,0)] == 0
    end

    @testset "Directions" begin
        @test Directions(1) == [[1], [-1]]
        @test Directions(CubicGrid(2,3)) == [[1,0], [0,1], [-1,0], [0,-1]]
        @test Directions(3) == [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
        @test_throws BoundsError Directions(3)[0]
        @test_throws BoundsError Directions(2)[5]
        @test_throws BoundsError Directions(1)[15]
    end

    @testset "Displacements" begin
        @test collect(Displacements(CubicGrid(3), center=false)) == [[0], [1], [2]]
        @test collect(Displacements(CubicGrid(3), center=true)) == [[-1], [0], [1]]

        @test collect(Displacements(CubicGrid(2,2))) == [[0,0], [1,0], [0,1], [1,1]]
        @test collect(Displacements(CubicGrid(2,3))) == [[0,0], [1,0], [0,1], [1,1], [0,2], [1,2]]
        @test collect(Displacements(CubicGrid(2,3); center=true)) == [
            [0,-1], [1,-1], [0,0], [1,0], [0,1], [1,1]
        ]
    end
end

@testset "Hubbard models with [t|u] = 0" begin
    bs1 = BoseFS((0,1,0))
    bs2 = BoseFS((3,3,3))
    HMt0 = HubbardMom1D(bs1; t=0)
    HMu0 = HubbardMom1D(bs1; u=0)
    HRt0 = HubbardReal1D(bs2; t=0)
    HRu0 = HubbardReal1D(bs2; u=0)

    @test diagonal_element(HMt0, bs1) == 0
    @test diagonal_element(HRu0, bs2) == 0
    @test all(iszero, e for (_, e) in offdiagonals(HMu0, bs1))
    @test all(iszero, e for (_, e) in offdiagonals(HRt0, bs2))

    t = 1
    bs3 = BoseFS((0,3,0))
    HM3Cu0 =HubbardMom1D(bs3; u=0, t, dispersion=continuum_dispersion)
    HM3Hu0 =HubbardMom1D(bs3; u=0, t, dispersion=hubbard_dispersion)
    @test HubbardMom1D(bs3; u=0, t) == HM3Hu0
    @test diagonal_element(HM3Cu0, bs3) == 0
    @test 2t*num_particles(bs3) + diagonal_element(HM3Hu0, bs3) == 0

    HM2Cu0 =HubbardMom1D(bs2; u=0, t, dispersion=continuum_dispersion)
    HM2Hu0 =HubbardMom1D(bs2; u=0, t, dispersion=hubbard_dispersion)
    @test diagonal_element(HM2Cu0, bs2) > 2t*num_particles(bs2)+diagonal_element(HM2Hu0,bs2)
    @test diagonal_element(HM2Cu0, bs2) ≈ 6*t*(2pi/num_modes_check_equal(bs2))^2

    HM3Ct0 =HubbardMom1D(bs3; t=0, dispersion=continuum_dispersion)
    HM3Ht0 =HubbardMom1D(bs3; t=0, dispersion=hubbard_dispersion)
    @test offdiagonals(HM3Ht0,bs3) == offdiagonals(HM3Ht0,bs3)
end

@testset "1C model properties" begin
    addr = near_uniform(BoseFS{100,100})

    for Hamiltonian in (HubbardReal1D, HubbardMom1D)
        @testset "$Hamiltonian" begin
            H = Hamiltonian(addr; t=1.0, u=2.0)
            @test H.t == 1.0
            @test H.u == 2.0
            @test LOStructure(H) == IsHermitian()
            @test starting_address(H) == addr
            @test eval(Meta.parse(repr(H))) == H
        end
    end
end

@testset "HubbardRealSpace" begin
    @testset "Constructor" begin
        bose = BoseFS((1, 2, 3, 4, 5, 6))
        @test_throws MethodError HubbardRealSpace(BoseFS{10,10})
        @test_throws ArgumentError HubbardRealSpace(bose; geometry=PeriodicBoundaries(3,3))
        @test_throws ArgumentError HubbardRealSpace(
            bose; geometry=PeriodicBoundaries(3,2), t=[1, 2],
        )
        @test_throws ArgumentError HubbardRealSpace(
            bose; geometry=PeriodicBoundaries(3,2), u=[1 1; 1 1],
        )
        @test_throws InexactError HubbardRealSpace(
            bose; geometry=PeriodicBoundaries(3,2), u=[1.0im], t=[1.0im]
        )
        @test_throws ArgumentError HubbardRealSpace(
            bose; potential=[1,2,3,4,5,5]
        )

        comp = CompositeFS(bose, bose)
        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), t=[1, 2], u=[1 2; 3 4],
        )
        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), t=[1, 2], w=[1 2; 3 4],
        )
        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), t=[1, 2], u=[2 2; 2 2; 2 2],
        )
        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), v=[1 1; 1 1; 1 1],
        )
        @test_throws ArgumentError HubbardRealSpace(
            comp; t=[1 2]
        )

        @test_logs (:warn,) HubbardRealSpace(FermiFS((1,0)), u=[2])
        @test_logs (:warn,) HubbardRealSpace(
            CompositeFS(BoseFS((1,1)), FermiFS((1,0))); u=[2 2; 2 2]
        )
        @test_logs (:warn,) HubbardRealSpace(
            bose; v=1, potential=reshape(float.([1, 2, 3, 4, 5, 5]), (6, 1))
        )

        H = HubbardRealSpace(comp, t=[1,2], u=[1 2; 2 3])
        @test eval(Meta.parse(repr(H))) == H
    end
    @testset "Offdiagonals" begin
        f = near_uniform(FermiFS{3,12})

        H = HubbardRealSpace(f, geometry=PeriodicBoundaries(3, 4))
        od_values = last.(offdiagonals(H * f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 6

        H = HubbardRealSpace(f, geometry=PeriodicBoundaries(4, 3))
        od_values = last.(offdiagonals(H * f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 8

        H = HubbardRealSpace(f, geometry=HardwallBoundaries(3, 4))
        od_values = last.(offdiagonals(H * f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 3

        H = HubbardRealSpace(f, geometry=HardwallBoundaries(4, 3))
        od_values = last.(offdiagonals(H * f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 4
    end
    @testset "1D Bosons (single)" begin
        H1 = HubbardReal1D(BoseFS((1, 1, 1, 1, 1, 0)); u=2, t=3)
        H2 = HubbardRealSpace(BoseFS((1, 1, 1, 1, 1, 0)); u=[2], t=[3])

        @test exact_energy(H1) == exact_energy(H2)
    end
    @testset "1D Bosons (2-component)" begin
        add2 = CompositeFS(
            BoseFS((1, 1, 1, 0, 0, 0)),
            BoseFS((1, 0, 0, 0, 0, 0)),
        )
        H2 = HubbardRealSpace(add2, t=[1,4], u=[2 3; 3 0])

        add3 = CompositeFS(
            BoseFS((1, 1, 1, 0, 0, 0)),
            FermiFS((1, 0, 0, 0, 0, 0)),
        )
        H3 = HubbardRealSpace(add3, t=[1,4], u=[2 3; 3 0])

        add4 = CompositeFS(
            BoseFS((1, 0, 0, 0, 0, 0)),
            BoseFS((1, 1, 1, 0, 0, 0)),
        )
        H4 = HubbardRealSpace(add4, t=[4,1], u=[0 3; 3 2])

        add5 = CompositeFS(
            FermiFS((1, 0, 0, 0, 0, 0)),
            BoseFS((1, 1, 1, 0, 0, 0)),
        )
        H5 = HubbardRealSpace(add5, t=[4,1], u=[0 3; 3 2])

        E2 = exact_energy(H2)
        E3 = exact_energy(H3)
        E4 = exact_energy(H4)
        E5 = exact_energy(H5)

        @test E2 ≈ E3 rtol=0.0001
        @test E3 ≈ E4 rtol=0.0001
        @test E4 ≈ E5 rtol=0.0001
    end
    @testset "1D Fermions" begin
        H1 = HubbardRealSpace(FermiFS((1, 1, 1, 0, 0, 0)), t=[3.5])

        # Kinetic energies [+1, -1, -2, -1, +1, +2] can be multiplied by t to get the exact
        # energy.
        @test exact_energy(H1) ≈ -14 rtol=0.0001

        # Not interacting, we can sum the parts together.
        H2 = HubbardRealSpace(
            CompositeFS(FermiFS((1, 1, 1, 1, 0, 0)), FermiFS((1, 1, 0, 0, 0, 0))),
            t=[1, 2], u=[0 0; 0 0],
        )

        @test exact_energy(H2) ≈ -3 + -6 rtol=0.0001

        # Repulsive interactions increase energy.
        H3 = HubbardRealSpace(
            CompositeFS(FermiFS((1, 1, 1, 1, 0, 0)), FermiFS((1, 1, 0, 0, 0, 0))),
            t=[1, 2], u=[0 1; 1 0],
        )
        @test exact_energy(H3) > -9

        # Attractive interactions reduce energy.
        H4 = HubbardRealSpace(
            CompositeFS(FermiFS((1, 1, 1, 1, 0, 0)), FermiFS((1, 1, 0, 0, 0, 0))),
            t=[1, 2], u=[0 -1; -1 0],
        )
        @test exact_energy(H4) < -9
    end
    @testset "1D trap" begin
        H1 = HubbardReal1DEP(BoseFS((1,2,3,4)); u=2, t=3, v_ho=4)
        H2 = HubbardRealSpace(BoseFS((1,2,3,4)); u=[2], t=[3], v=[4])

        @test exact_energy(H1) ≈ exact_energy(H2)
        # composite
        add3 = CompositeFS(
            BoseFS((1, 1, 1, 0, 0, 0)),
            BoseFS((1, 0, 0, 0, 0, 0)),
        )
        H3 = HubbardRealSpace(add3, v=[1,4], u=[2 3; 3 0])

        add4 = CompositeFS(
            BoseFS((1, 0, 0, 0, 0, 0)),
            BoseFS((1, 1, 1, 0, 0, 0)),
        )
        H4 = HubbardRealSpace(add4, v=[4,1], u=[0 3; 3 2])

        E3 = exact_energy(H3)
        E4 = exact_energy(H4)
        @test E3 ≈ E4 rtol=0.0001

        ranges = (range(-2; length=6),)
        x_sq = map(x -> Tuple(x) .^ 2, CartesianIndices(ranges))
        pot_vec = zeros(6, 1)
        pot_vec[:, 1] .= vec(map(x -> sum(x), x_sq))
        addr = BoseFS(1, 1, 1, 0, 0, 0)
        H5 = HubbardRealSpace(addr; v=1)
        H6 = HubbardRealSpace(addr; potential=pot_vec, v = 0.0)
        @test exact_energy(H5) ≈ exact_energy(H6)
    end
    @testset "2D Fermions" begin
        @testset "2 × 2" begin
            p22 = PeriodicBoundaries(2, 2)
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{1, 4}), geometry=p22, t=[2])
            ) ≈ -8 rtol=0.001
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{2, 4}), geometry=p22, t=[2])
            ) ≈ -8 rtol=0.001
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{3, 4}), geometry=p22, t=[2])
            ) ≈ -8 rtol=0.001
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{4, 4}), geometry=p22, t=[2])
            ) ≈ 0 rtol=0.001
        end
        @testset "4 × 4" begin
            p44 = PeriodicBoundaries(4, 4)
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{1, 16}), geometry=p44)
            ) ≈ -4 rtol=0.001
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{2, 16}), geometry=p44)
            ) ≈ -6 rtol=0.001
            @test exact_energy(
                HubbardRealSpace(near_uniform(FermiFS{3, 16}), geometry=p44)
            ) ≈ -8 rtol=0.001
            # Note: a vector with only near_uniform is orthogonal to the ground state, so
            # KrylovKit will give the wrong energy here.
            @test exact_energy(
                HubbardRealSpace(FermiFS((1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0)), geometry=p44)
            ) ≈ -10 rtol=0.001
        end
        @testset "Two-component" begin
            H1 = HubbardRealSpace(
                CompositeFS(near_uniform(FermiFS{3,9}), near_uniform(FermiFS{2,9}));
                t=[1,2],
                u=[0 0; 0 0],
                geometry=PeriodicBoundaries(3, 3),
            )
            @test exact_energy(H1) ≈ -16 rtol=0.001

            H2 = HubbardRealSpace(
                CompositeFS(near_uniform(FermiFS{3,9}), near_uniform(FermiFS{2,9}));
                t=[1,2],
                u=[0 1; 1 0],
                geometry=PeriodicBoundaries(3, 3),
            )
            @test exact_energy(H2) > -16

            H3 = HubbardRealSpace(
                CompositeFS(near_uniform(FermiFS{3,9}), near_uniform(FermiFS{2,9}));
                t=[1,2],
                u=[0 -1; -1 0],
                geometry=PeriodicBoundaries(3, 3),
            )
            @test exact_energy(H3) < -16
        end
        @testset "Hardwall" begin
            geom1 = HardwallBoundaries(2, 3)
            geom2 = HardwallBoundaries(3, 2)
            bose = BoseFS((1, 1, 1, 0, 0, 0))
            fermi = FermiFS((1, 0, 0, 0, 1, 0))

            H1 = HubbardRealSpace(bose, geometry=geom1)
            H2 = HubbardRealSpace(bose, geometry=geom2)
            @test exact_energy(H1) ≈ exact_energy(H2)

            H1 = HubbardRealSpace(fermi, geometry=geom1)
            H2 = HubbardRealSpace(fermi, geometry=geom2)
            @test exact_energy(H1) ≈ exact_energy(H2)
        end
    end
    @testset "Complex hopping" begin
        address = FermiFS(1, 0, 1, 0)
        for H in (
            HubbardRealSpace(address; t=[2.0 + 3im], geometry=CubicGrid(2, 2)),
            HubbardRealSpace(address; w=[6], t=[2.0 + 3im], geometry=CubicGrid(4)),
            HubbardRealSpace(address; t=[im 2im], geometry=CubicGrid(2, 2)),
        )
            @test eltype(H) ≡ ComplexF64
            @test LOStructure(H) ≡ IsHermitian()
            test_hamiltonian_structure(H)
        end
    end
    @testset "Nearest neighbour interaction" begin
        addr = near_uniform(BoseFS{4,4})
        H1 = HubbardRealSpace(addr; geometry=PeriodicBoundaries(4), w=[2.0])
        H2 = ExtendedHubbardReal1D(addr; v=2.0)
        @test Matrix(H1) == Matrix(H2)
        addr = near_uniform(FermiFS{2,4})
        H1 = HubbardRealSpace(addr; geometry=PeriodicBoundaries(4), w=[-1.0])
        H2 = ExtendedHubbardReal1D(addr; v=-1.0)
        @test Matrix(H1) == Matrix(H2)

        addr = BoseFS(1,1,0, 0,0,0, 0,0,0)
        H1 = HubbardRealSpace(addr; geometry=PeriodicBoundaries(3, 3), w=[2])
        H2 = HubbardRealSpace(addr; geometry=HardwallBoundaries(3, 3), w=[2])
        @test diagonal_element(H1 * addr) == 2
        @test diagonal_element(H2 * BoseFS(1,0,1, 0,0,0, 0,0,0)) == 0

        @test diagonal_element(H1 * BoseFS(1,0,0, 0,0,0, 1,0,0)) == 2
        @test diagonal_element(H2 * BoseFS(1,0,0, 0,0,0, 1,0,0)) == 0
    end
    @testset "Per-dimension hopping" begin
        addr = BoseFS(1,0,0, 0,0,0, 0,0,0)
        H = HubbardRealSpace(addr; geometry=PeriodicBoundaries(3, 3), t=[3 4])

        offdiags = DVec(offdiagonals(H * addr))
        @test offdiags[BoseFS(0,1,0, 0,0,0, 0,0,0)] == -3
        @test offdiags[BoseFS(0,0,1, 0,0,0, 0,0,0)] == -3
        @test offdiags[BoseFS(0,0,0, 1,0,0, 0,0,0)] == -4
        @test offdiags[BoseFS(0,0,0, 0,0,0, 1,0,0)] == -4
    end
end

@testset "Importance sampling" begin
    @testset "Gutzwiller" begin
        @testset "Gutzwiller transformation" begin
            for H in (
                HubbardMom1D(BoseFS((2,2,2)), u=6),
                ExtendedHubbardReal1D(BoseFS(1,1,1,1,1,1), u=6, t=2.0),
                ExtendedHubbardMom1D(BoseFS(0,0,1,1,1,0,0), u=3),
            )
                h_matrix = sparse(H; sort=true)

                # GutzwillerSampling with parameter zero is exactly equal to the original H
                G = GutzwillerSampling(H, 0.0)

                @test sparse(G; sort=true) == h_matrix
                @test starting_address(G) == starting_address(H)

                @test LOStructure(G) isa AdjointKnown
                @test LOStructure(TransformUndoer(G,G)) isa AdjointKnown

                @test eval(Meta.parse(repr(G))) == G
                @test eval(Meta.parse(repr(G'))) == G'

                g = 0.1
                g_matrix = sparse(GutzwillerSampling(G, g); sort=true)
                for i in axes(g_matrix, 1), j in axes(g_matrix, 2)
                    value = exp(h_matrix[i,i] * -g) * h_matrix[i,j] * exp(h_matrix[j,j] * g)
                    @test g_matrix[i, j] ≈ value
                end
            end
        end

        @testset "Gutzwiller observables" begin
            for H in (
                HubbardReal1D(BoseFS((2,2,2)), u=6),
                HubbardMom1D(BoseFS((2,2,2)), u=6),
                ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
                ExtendedHubbardMom1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
            )
                # energy
                g = rand()
                x = rand()
                G = GutzwillerSampling(H, g)
                address = starting_address(H)
                dv = DVec(address => x)
                # transforming the Hamiltonian again should be consistent
                fsq = TransformUndoer(G)
                fHf = TransformUndoer(G, H)
                Ebare = dot(dv, H, dv)/dot(dv, dv)
                Egutz = dot(dv, G, dv)/dot(dv, dv)
                Etrans = dot(dv, fHf, dv)/dot(dv, fsq, dv)
                @test Ebare ≈ Egutz ≈ Etrans

                # general operators
                m = num_modes_check_equal(address)
                g2vals = map(d -> dot(dv, G2RealCorrelator(d), dv)/dot(dv, dv), 0:m-1)
                g2transformed = map(
                    d -> dot(dv, TransformUndoer(G,G2RealCorrelator(d)), dv)/dot(dv, fsq, dv),
                    0:m-1
                )
                @test all(g2vals ≈ g2transformed)
            end
        end
    end

    @testset "GuidingVector" begin
        H = HubbardMom1D(BoseFS((2,2,2)), u=6)
        v = DVec(
            BoseFS(0, 0, 6) => 0.0770580680636451,
            BoseFS(6, 0, 0) => 0.0770580680636451,
            BoseFS(1, 1, 4) => 0.3825802976327182,
            BoseFS(4, 1, 1) => 0.3825802976327182,
            BoseFS(0, 6, 0) => 0.04322440994245527,
            BoseFS(3, 3, 0) => 0.2565124277520772,
            BoseFS(3, 0, 3) => 0.3460652270329457,
            BoseFS(0, 3, 3) => 0.2565124277520772,
            BoseFS(1, 4, 1) => 0.28562685053740633,
            BoseFS(2, 2, 2) => 0.6004825560434165;
        )
        h_matrix = sparse(H; sort=true)
        @testset "GuidingVector transformation" begin
            @testset "With empty vector" begin
                G = GuidingVectorSampling(H, empty(v), 0.2)

                @test starting_address(G) == starting_address(H)
                @test LOStructure(G) isa AdjointKnown
                @test LOStructure(TransformUndoer(G,G)) isa AdjointKnown

                @test h_matrix == sparse(G; sort=true)
            end

            @testset "With non-empty vector" begin
                G = GuidingVectorSampling(H, v, 0.2)

                @test starting_address(G) == starting_address(H)
                @test LOStructure(G) isa AdjointKnown
                @test LOStructure(TransformUndoer(G,G)) isa AdjointKnown
                @test G == GuidingVectorSampling(H; vector = v, eps = 0.2) # call signature

                bsr = BasisSetRepresentation(G; sort=true)
                g_matrix = bsr.sparse_matrix
                basis = bsr.basis

                for i in axes(g_matrix, 1), j in axes(g_matrix, 2)
                    top = ifelse(v[basis[i]] < 0.2, 0.2, v[basis[i]])
                    bot = ifelse(v[basis[j]] < 0.2, 0.2, v[basis[j]])

                    weight = top / bot
                    @test g_matrix[i, j] == h_matrix[i, j] * weight
                end
            end
        end

        @testset "Guiding vector observables" begin
            for H in (
                HubbardReal1D(BoseFS((2,2,2)), u=6),
                HubbardMom1D(BoseFS((2,2,2)), u=6),
                ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
                ExtendedHubbardMom1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
            )
                # energy
                x = rand()
                G = GuidingVectorSampling(H, v, 0.2)
                address = starting_address(H)
                dv = DVec(address => x)
                # transforming the Hamiltonian again should be consistent
                fsq = TransformUndoer(G)
                fHf = TransformUndoer(G, H)
                Ebare = dot(dv, H, dv)/dot(dv, dv)
                Egutz = dot(dv, G, dv)/dot(dv, dv)
                Etrans = dot(dv, fHf, dv)/dot(dv, fsq, dv)
                @test Ebare ≈ Egutz ≈ Etrans

                # general operators
                m = num_modes_check_equal(address)
                g2vals = map(d -> dot(dv, G2RealCorrelator(d), dv)/dot(dv, dv), 0:m-1)
                g2transformed = map(d -> dot(dv, TransformUndoer(G,G2RealCorrelator(d)), dv)/dot(dv, fsq, dv), 0:m-1)
                @test all(g2vals ≈ g2transformed)
            end
        end
    end

    @testset "adjoints" begin
        M = MatrixHamiltonian(rand(Complex{Float64}, (20, 20)))
        @test Matrix(M; sort=true) == M.m
        @test Matrix(M'; sort=true) == M.m'

        @testset "Gutzwiller adjoint" begin
            @test Matrix(GutzwillerSampling(M, 0.2)') == Matrix(GutzwillerSampling(M, 0.2))'
            @test LOStructure(GutzwillerSampling(M, 0.2)) isa AdjointKnown
            @test LOStructure(
                GutzwillerSampling(HubbardReal1D(BoseFS((1,2)),t=0+2im), 0.2)
            ) isa AdjointKnown
        end
        @testset "GuidingVector adjoint" begin
            v = DVec(starting_address(M) => 10; capacity=10)
            @test Matrix(GuidingVectorSampling(M, v, 0.2)') ≈
                Matrix(GuidingVectorSampling(M, v, 0.2))'
            @test LOStructure(GuidingVectorSampling(M, v, 0.2)) isa AdjointKnown
            @test LOStructure(GuidingVectorSampling(
                HubbardReal1D(BoseFS((1,2)),t=0+2im),
                DVec(BoseFS((1,2)) => 1.1; capacity=10),
                0.2,
            )) isa AdjointKnown
        end
    end

    @testset "supported transformations" begin
        # supported
        H = HubbardMom1D(BoseFS((2,2,2)), u=6)
        v = DVec(starting_address(H) => 1.)
        for G in (
            GutzwillerSampling(H,g=1),
            GuidingVectorSampling(H, v, 0.2),
        )
            # test supported constructor
            @test !isa(try TransformUndoer(G) catch e e end, Exception)
            @test !isa(try TransformUndoer(G,H) catch e e end, Exception)
        end
        # unsupported
        for H in (
            HubbardMom1D(BoseFS((2,2,2)), u=6),
            ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
            ExtendedHubbardMom1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
        )
            @test_throws ArgumentError TransformUndoer(H)
            @test_throws ArgumentError TransformUndoer(H, H)
        end
    end
end

@testset "MatrixHamiltonian" begin
    # generate matrix
    ham = HubbardReal1D(BoseFS((1, 1, 1, 1)))
    dim = dimension(ham)
    @test dim ≤ dimension(Int, starting_address(ham)) == dimension(starting_address(ham))
    bsr = BasisSetRepresentation(ham, starting_address(ham))
    sparse_matrix, basis = sparse(bsr), bsr.basis
    @test dim == length(basis)

    # run ProjectorMonteCarloProblem in deterministic mode with Hamiltonian and DVec
    v = DVec(k=>1.0 for k in basis; style=IsDeterministic()) # corresponds to `ones(dim)`
    a = solve(ProjectorMonteCarloProblem(ham; start_at=v)).df

    # MatrixHamiltonian
    @test_throws ArgumentError MatrixHamiltonian([1 2 3; 4 5 6])
    @test_throws ArgumentError MatrixHamiltonian(sparse_matrix, starting_address = dim+1)
    # adjoint nonhermitian
    nonhermitian = MatrixHamiltonian([1 2; 4 5])
    @test LOStructure(nonhermitian) == AdjointKnown()
    @test get_offdiagonal(nonhermitian,2,1)[2] == get_offdiagonal(nonhermitian',1,1)[2]

    # wrap sparse matrix as MatrixHamiltonian
    mh =  MatrixHamiltonian(sparse_matrix)
    # adjoint IsHermitian
    @test LOStructure(mh) == IsHermitian()
    @test mh' == mh

    @test starting_address(mh) == 1
    @test dimension(mh) == dim

    # ProjectorMonteCarloProblem with MatrixHamiltonian
    # float walkernumber triggers IsDeterministic algorithm
    sim = solve(ProjectorMonteCarloProblem(mh; start_at=DVec(pairs(ones(dim)))))
    @test StochasticStyle(only(state_vectors(sim))) == IsDeterministic()
    d = DataFrame(sim)
    @test d.shift ≈ a.shift
    # integer walkernumber triggers IsStochasticInteger algorithm
    sim = solve(
        ProjectorMonteCarloProblem(mh; start_at=DVec(pairs(ones(Int, dim))), random_seed=18)
    )
    @test StochasticStyle(only(state_vectors(sim))) == IsStochasticInteger()
    e = DataFrame(sim)
    @test ≈(e.shift[end], a.shift[end], atol=0.3)
    # wrap full matrix as MatrixHamiltonian
    fmh =  MatrixHamiltonian(Matrix(sparse_matrix))
    sim = solve(
        ProjectorMonteCarloProblem(fmh; start_at=DVec(pairs(ones(dim))), random_seed=15)
    )
    @test StochasticStyle(only(state_vectors(sim))) == IsDeterministic()
    f = DataFrame(sim)
    @test f.shift ≈ a.shift
end

using Rimu.Hamiltonians: circshift_dot

@testset "Correlation functions" begin
    @testset "circshift_dot" begin
        for i in 1:10
            A = rand(3, 4, 5)
            B = rand(3, 4, 5)
            inds = (rand(0:3), rand(0:4), rand(0:5))
            @test circshift_dot(A, B, inds) ≈ dot(A, circshift(B, inds))
        end
    end

    @testset "G2RealCorrelator" begin
        m = 6
        n1 = 4
        n2 = m
        add1 = BoseFS((n1,0,0,0,0,0))
        add2 = near_uniform(BoseFS{n2,m})

        # localised state
        @test diagonal_element(G2RealCorrelator(0), add1) == n1 * (n1 - 1) / m
        @test diagonal_element(G2RealCorrelator(1), add1) == 0.0

        # constant density state
        @test diagonal_element(G2RealCorrelator(0), add2) == (n2/m) * ((n2/m) - 1)
        @test diagonal_element(G2RealCorrelator(1), add2) == (n2/m)^2

        # local-local
        comp = CompositeFS(add1,add1)
        @test diagonal_element(G2RealCorrelator(0), comp) == 2n1 * (2n1 - 1) / m
        @test diagonal_element(G2RealCorrelator(1), comp) == 0.0

        # local-uniform (assuming unit filling)
        comp = CompositeFS(add1,add2)
        @test diagonal_element(G2RealCorrelator(0), comp) == (n1 + 1) * n1 / m
        @test diagonal_element(G2RealCorrelator(1), comp) == (2 * (n1 + 1) + (m - 2)) / m

        # uniform-uniform
        comp = CompositeFS(add2,add2)
        @test diagonal_element(G2RealCorrelator(0), comp) == (2n2 / m) * (2 * (n2 / m) - 1)
        @test diagonal_element(G2RealCorrelator(1), comp) == (2n2 / m)^2

        # offdiagonals
        @test num_offdiagonals(G2RealCorrelator(0), add1) == 0
        @test num_offdiagonals(G2RealCorrelator(0), comp) == 0

        # Test show method
        d = 5
        output = @capture_out print(G2RealCorrelator(d))
        @test output == "G2RealCorrelator($d)"
    end

    @testset "G2RealSpace" begin
        @testset "1D G2RealCorrelator comparison" begin
            @testset "constructors" begin
                g2_1 = G2RealSpace(CubicGrid(2, 2, 3), 1, 3)
                g2_2 = G2RealSpace(CubicGrid(2, 2), 2)
                g2_3 = G2RealSpace(CubicGrid(2, 2); sum_components=true)
                @test g2_1 isa G2RealSpace{1,3}
                @test g2_2 isa G2RealSpace{2,2}
                @test g2_3 isa G2RealSpace{0,0}

                @test eval(Meta.parse(repr(g2_1))) == g2_1
                @test eval(Meta.parse(repr(g2_2))) == g2_2
                @test eval(Meta.parse(repr(g2_3))) == g2_3

                @test scalartype(g2_1) == Float64
                @test scalartype(g2_2) == Float64
                @test scalartype(g2_3) == Float64

                @test_throws ArgumentError G2RealSpace(CubicGrid(3), 1, 0)
                @test_throws ArgumentError G2RealSpace(CubicGrid(2, 2), 0, 0)
                @test_throws ArgumentError G2RealSpace(CubicGrid(1, 2, 3), -1, 2)
                @test_throws ArgumentError G2RealSpace(CubicGrid(12), 3; sum_components=true)
            end
            @testset "single components" begin
                addr = near_uniform(BoseFS{6,6})
                H = HubbardReal1D(addr)
                v = normalize!(H * (H * (H * DVec(addr => 1.0))))

                g2 = dot(v, G2RealSpace(CubicGrid(6)), v)
                for d in 0:5
                    @test g2[d + 1] ≈ dot(v, G2RealCorrelator(d), v)
                end
            end

            @testset "sum of components" begin
                addr = CompositeFS(BoseFS(4, 1=>1), BoseFS(4, 1=>1), BoseFS(4, 1=>1))
                H = HubbardRealSpace(addr)
                v = normalize!(H * (H * (H * DVec(addr => 1.0))))

                g2 = dot(v, G2RealSpace(CubicGrid(4); sum_components=true), v)
                for d in 0:3
                    @test g2[d + 1] ≈ dot(v, G2RealCorrelator(d), v)
                end

                g2_nosum = dot(v, G2RealSpace(CubicGrid(4)), v)
                @test iszero(g2_nosum)
            end

            @testset "adjoint" begin
                G2 = G2RealSpace(CubicGrid(6))
                @test LOStructure(G2) isa IsDiagonal
                @test eltype(eltype(G2)) <: Real
                @test G2' == G2
            end
        end

        @testset "2-component, 2D" begin
            c1 = BoseFS(1, 0, 0, 0, 0, 0)
            c2 = BoseFS(1, 2, 3, 4, 5, 6)
            addr = CompositeFS(c1, c2)
            geom = CubicGrid((3, 2))

            g2_11 = diagonal_element(G2RealSpace(geom, 1, 1), addr)
            g2_12 = diagonal_element(G2RealSpace(geom, 1, 2), addr)
            g2_21 = diagonal_element(G2RealSpace(geom, 2, 1), addr)
            g2_22 = diagonal_element(G2RealSpace(geom, 2, 2), addr)

            # Sum is N1 * (N2 - δ_12) / M
            @test sum(g2_11) == 0
            @test sum(g2_12) == 3.5
            @test sum(g2_21) == 3.5
            @test sum(g2_22) == 70

            # Swapping components flips axes
            @test g2_11 == [0 0; 0 0; 0 0]
            @test g2_12 == [1 4; 2 5; 3 6] ./ 6
            @test g2_21 == [1 4; 3 6; 2 5] ./ 6
            @test g2_22[1, 1] == sum(n -> n * (n-1), onr(c2)) / 6
            @test g2_22[3, 2] == dot(onr(c2, geom), circshift(onr(c2, geom), (2, 1))) / 6
        end

        @testset "G2 is symmetric for translationally invariant ground states" begin
            addr = near_uniform(BoseFS{3,18})
            geom = CubicGrid((2,3,3), (false, true, true))
            H = HubbardRealSpace(addr; geometry=geom)
            bsr = BasisSetRepresentation(H)
            v0 = PDVec(zip(bsr.basis, eigen(Matrix(bsr)).vectors[:,1]))

            g2 = dot(v0, G2RealSpace(geom), v0)

            @test sum(g2) ≈ 3 * 2 / 18
            @test g2[:,2,:] ≈ g2[:,3,:]
            @test g2[:,:,2] ≈ g2[:,:,3]
            @test minimum(g2) == first(g2)
        end
        @testset "G2 operator vector products" begin
            addr = BoseFS(2, 0, 1, 1)
            geom = CubicGrid(2, 2)
            g2 = G2RealSpace(geom)
            dv = DVec(addr => 2.0)
            @test dot(dv, g2, dv) isa SMatrix
            w = empty(dv, eltype(g2); style=IsDeterministic())
            mul!(w, g2, dv) # operator vector product
            @test dot(dv, w) == dot(dv, g2, dv) == conj.(dot(w, dv))
        end
    end

    @testset "SuperfluidCorrelator" begin
        m = 6
        n1 = 4
        n2 = m
        add1 = BoseFS((n1,0,0,0,0,0))
        add2 = near_uniform(BoseFS{n2,m})

        # localised state
        @test @inferred diagonal_element(SuperfluidCorrelator(0), add1) == n1/m
        @test @inferred diagonal_element(SuperfluidCorrelator(1), add1) == 0.

            # constant density state
            @test diagonal_element(SuperfluidCorrelator(0), add2) == n2/m
        @test diagonal_element(SuperfluidCorrelator(1), add2) == 0.

            # offdiagonals
            @test num_offdiagonals(SuperfluidCorrelator(0), add1) == 1
        @test num_offdiagonals(SuperfluidCorrelator(0), add2) == 6

        # get_offdiagonal
        @test get_offdiagonal(SuperfluidCorrelator(0), add1, 1) == (add1, n1/m)
        @test get_offdiagonal(SuperfluidCorrelator(1), add1, 1) == (BoseFS((3,1,0,0,0,0)), sqrt(n1)/m)
        @test get_offdiagonal(SuperfluidCorrelator(0), add2, 1) == (add2, 1/m)
        @test get_offdiagonal(SuperfluidCorrelator(1), add2, 1) == (BoseFS((0,2,1,1,1,1)), sqrt(2)/m)

        # Test show method
        d = 5
        output = @capture_out print(SuperfluidCorrelator(d))
        @test output == "SuperfluidCorrelator($d)"
    end

    @testset "StringCorrelator" begin
        m = 6
        n1 = 4
        n2 = m

        # unital refers to n̄=1
        non_unital_localised_state = BoseFS((n1,0,0,0,0,0))
        non_unital_uniform_state = near_uniform(non_unital_localised_state)

        localised_state = BoseFS((n2,0,0,0,0,0))
        uniform_state = near_uniform(BoseFS{n2,m})

        S0 = StringCorrelator(0)
        S1 = StringCorrelator(1)
        S2 = StringCorrelator(2)
        SI = StringCorrelator(3; address=uniform_state)
        SC = StringCorrelator(3; type=ComplexF64)

        @test num_offdiagonals(S0, localised_state) == 0

        # non unital localised state
        @test @inferred diagonal_element(S0, non_unital_localised_state) ≈ 20/9
        @test @inferred diagonal_element(S1, non_unital_localised_state) ≈ (-4/9)*exp(im * -2pi/3)

        # non unital near uniform state
        @test @inferred diagonal_element(S0, non_unital_uniform_state) ≈ 2/9

        # constant density localised state
        @test @inferred diagonal_element(S0, localised_state) == 5.
            @test @inferred diagonal_element(S1, localised_state) ≈ 1
        @test @inferred diagonal_element(S2, localised_state) ≈ -1

        # constant density uniform state
        @test @inferred diagonal_element(S0, uniform_state) == 0
        @test @inferred diagonal_element(S2, uniform_state) == 0

        # Test return type for integer, and non-integer filling
        @test @inferred diagonal_element(S0, localised_state) isa Float64
        @test @inferred diagonal_element(S1, non_unital_localised_state) isa ComplexF64
        @test @inferred diagonal_element(SI, uniform_state) isa Float64
        @test @inferred diagonal_element(SC, uniform_state) isa ComplexF64

        # Test show method
        d = 5
        output = @capture_out print(StringCorrelator(d))
        @test output == "StringCorrelator($d; type=ComplexF64)"

    end

    @testset "Momentum" begin
        @test diagonal_element(Momentum(), BoseFS((0,0,2,1,3))) ≡ 2.0
        @test diagonal_element(Momentum(fold=false), BoseFS((0,0,2,1,3))) ≡ 7.0
        @test diagonal_element(Momentum(1), BoseFS((1,0,0,0))) ≡ -1.0
        @test_throws MethodError diagonal_element(Momentum(2), BoseFS((0,1,0)))

        for address in (FermiFS2C((1,0,0,1), (0,0,1,0)),)
            @test diagonal_element(Momentum(1), address) + diagonal_element(Momentum(2), address) ≡
                diagonal_element(Momentum(0), address)
        end

        @test num_offdiagonals(Momentum(), BoseFS((0,1,0))) == 0
        @test LOStructure(Momentum(2; fold=true)) == IsDiagonal()
        @test Momentum(1)' === Momentum(1)
    end

    @testset "DensityMatrixDiagonal" begin
        @test diagonal_element(DensityMatrixDiagonal(5), FermiFS((0,1,0,1,0,1,0))) == 0
        @test diagonal_element(DensityMatrixDiagonal(2; component=1), BoseFS((1,5,1,0))) == 5

        for address in (
            CompositeFS(BoseFS((1,2,3,4,5)), BoseFS((5,4,3,2,1))),
            )
            for i in 1:5
                @test diagonal_element(DensityMatrixDiagonal(i, component=1), address) == i
                @test diagonal_element(DensityMatrixDiagonal(i, component=2), address) == 6 - i
                @test diagonal_element(DensityMatrixDiagonal(i), address) == 6
            end
        end

        @test num_offdiagonals(DensityMatrixDiagonal(1), BoseFS((0,1,0))) == 0
        @test LOStructure(DensityMatrixDiagonal(2)) == IsDiagonal()
        @test DensityMatrixDiagonal(15)' === DensityMatrixDiagonal(15)

        @test allows_address_type(DensityMatrixDiagonal(1), BoseFS(0,1,0))
        @test allows_address_type(DensityMatrixDiagonal(2; component=1), BoseFS(0, 1, 0))
        @test allows_address_type(DensityMatrixDiagonal(2; component=2), BoseFS(0, 1, 0)) == false
        @test allows_address_type(DensityMatrixDiagonal(2), CompositeFS(BoseFS(0, 1, 0), BoseFS(0, 1, 0)))
        csf = CompositeFS(BoseFS(0, 1, 0), BoseFS(0, 1))
        @test allows_address_type(DensityMatrixDiagonal(2; component=2), csf) == true
        @test allows_address_type(DensityMatrixDiagonal(2), csf) == false
    end

    @testset "Reduced Density Matrix" begin
        addr_bose = BoseFS(6,5,4,3,2,1)
        addr_fermi = FermiFS(1,0,1,0,1,0)
        for i in 1:6
            naddr_fermi1, value_f1 = excitation(addr_fermi,find_mode(addr_fermi,(1,)),find_mode(addr_fermi,(i,)))
            naddr_fermi2, value_f2 = excitation(addr_fermi,find_mode(addr_fermi,(2,)),find_mode(addr_fermi,(i,)))
            @test diagonal_element(SingleParticleExcitation(i, i), addr_bose) == 7-i
            @test diagonal_element(SingleParticleExcitation(i, i), addr_fermi) == i%2
            @test get_offdiagonal(SingleParticleExcitation(2,i),addr_fermi,1) == (naddr_fermi2,value_f2)
            if i == 1
                @test diagonal_element(TwoParticleExcitation(1,i,1,i), addr_bose) == 5*(7-i)
                @test diagonal_element(TwoParticleExcitation(1,i,1,i), addr_fermi) == 0.0
                @test get_offdiagonal(SingleParticleExcitation(1,i),addr_fermi,1) == (naddr_fermi1,1)
            else
                naddr_bose1, value_b1 = excitation(addr_bose,find_mode(addr_bose,(1,)),find_mode(addr_bose,(i,)))
                naddr_fermi11, value_f11 = excitation(addr_fermi,find_mode(addr_fermi,(1,2,)),find_mode(addr_fermi,(i,1,)))
                @test diagonal_element(TwoParticleExcitation(1,i,1,i), addr_bose) == 6*(7-i)
                @test diagonal_element(TwoParticleExcitation(1,i,1,i), addr_fermi) == i%2
                @test get_offdiagonal(SingleParticleExcitation(1,i),addr_bose,1) == (naddr_bose1,value_b1)
                @test get_offdiagonal(SingleParticleExcitation(1,i),addr_fermi,1) == (naddr_fermi1,0.0)
                @test get_offdiagonal(TwoParticleExcitation(1,2,1,i),addr_fermi,1) == (naddr_fermi11,value_f11)
                if i!=2
                    naddr_bose11, value_b11 = excitation(addr_bose,find_mode(addr_bose,(1,2,)),find_mode(addr_bose,(i,1,)))
                    @test get_offdiagonal(TwoParticleExcitation(1,2,1,i),addr_bose,1) == (naddr_bose11,value_b11)
                end
            end
        end
        @test num_offdiagonals(SingleParticleExcitation(1,2), addr_bose) == 1
        @test LOStructure(SingleParticleExcitation(1,2)) == AdjointUnknown()
        @test num_offdiagonals(TwoParticleExcitation(1,2,2,1), addr_bose) == 0
        @test LOStructure(TwoParticleExcitation(1,2,2,1)) == AdjointUnknown()
    end
end

@testset "HubbardReal1DEP" begin
    for M in [3,4]
        is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
        js = shift_lattice(is) # shifted such that js[1] = 0
        @test js[1] == 0
        @test shift_lattice_inv(js) == is
    end
    m = 100 # number of lattice sites, i.e. L in units of the lattice parameter alpha
    n = 1 # number of particles
    addr = BoseFS(Tuple(i == 1 ? n : 0 for i in 1:m)) # at the bottom of potential
    l0 = 10 # harmonic oscillator length in units of alpha; 1 << l0 << m
    v_ho = 0.5/l0^2 # energies now in units of hbar omega
    t = 0.5*l0^2 # energies now in units of hbar omega
    h = HubbardReal1DEP(addr; t, v_ho)
    # all particles at the bottom of potential well
    @test diagonal_element(h, addr) == 0 == h.ep⋅onr(addr)
    energies = eigvals(Matrix(h)) .+ 2n*t # shifted by bottom of Hubbard dispersion
    @test energies[1:3] ≈ 0.5:1.0:2.5 atol=0.005 # first few eigenvalues
    # # Here is a quick plot script that shows eigenvalues to deviate around n = 10
    # using Plots
    # r = 1:15
    # scatter(r .-1, energies[r], label="Hubbard with ho potential", legend=:bottomright)
    # plot!(n->n+0.5, r .-1, label="n + 1/2")
    # ylabel!("Energy")
    # xlabel!("ho quantum number n")
    # title!("Harmonic oscillator in Hubbard, M = $m, l_0 = $l0")
end

@testset "HubbardMom1D(FermiFS2C)" begin
    @testset "Two fermions vs two bosons" begin
        bose = HubbardMom1D(BoseFS((0,0,2,0,0)))
        fermi = HubbardMom1D(CompositeFS(FermiFS((0,0,1,0,0)), FermiFS((0,0,1,0,0))))

        @test exact_energy(bose) ≈ exact_energy(fermi)
    end
    @testset "Comparison with HubbardRealSpace" begin
        c = CompositeFS(FermiFS((0,1,1,1,0)), FermiFS((0,0,1,0,0)))
        h_real = HubbardRealSpace(c; u=[0 0.5; 0.5 0])
        h_mom = HubbardMom1D(c; u=0.5)

        @test exact_energy(h_real) ≈ exact_energy(h_mom)
    end
end

@testset "HubbardMom1DEP" begin
    @testset "Comparison with real space" begin
        h_real = HubbardReal1DEP(BoseFS((1,1,1,1,1)); v_ho=2, t=2, u=1.2)
        h_mom = HubbardMom1DEP(BoseFS((0,0,5,0,0)); v_ho=2, t=2, u=1.2)

        @test exact_energy(h_real) ≈ exact_energy(h_mom)
    end
    @testset "no potential/fermions" begin
        c = CompositeFS(FermiFS((0,1,0,1,0,0)), FermiFS((0,0,1,0,0,0)))
        h_real = HubbardMom1D(c, u=2)
        h_mom = HubbardMom1DEP(c, v_ho=0, u=2)

        @test Matrix(h_real) == Matrix(h_mom)
    end
    @testset "Two fermions vs two bosons" begin
        for dispersion in (continuum_dispersion, hubbard_dispersion)
            bose = HubbardMom1DEP(BoseFS((0,0,2,0,0)); v_ho=1.5, dispersion)
            fermi = HubbardMom1DEP(
                CompositeFS(FermiFS((0,0,1,0,0)), FermiFS((0,0,1,0,0)));
                v_ho=1.5, dispersion
            )
            @test exact_energy(bose) ≈ exact_energy(fermi)
        end
    end
end

@testset "FroehlichPolaron1D" begin
    addr1 = BoseFS{missing}(1,1,1)

    # test momentum_cutoff and mode_cutoff when initialising
    addr2 = BoseFS{missing}(1,2,3)
    @test_throws ArgumentError FroehlichPolaron1D(addr2; mode_cutoff=1.0)
    @test_throws ArgumentError FroehlichPolaron1D(addr2; momentum_cutoff=10.0)
    @test_throws ArgumentError FroehlichPolaron1D(BoseFS{missing}(3,2,1); momentum_cutoff=10.0)

    addr3 = BoseFS{missing}(1,2,3,4)
    f2 = FroehlichPolaron1D(addr2)
    f3 = FroehlichPolaron1D(addr3; mode_cutoff=20.0)

    @test starting_address(f2) == f2.addr == addr2
    @test f2 == @test_logs (:warn,) FroehlichPolaron1D(addr2; mass=1)
    @test f2 == @test_logs (:warn,) FroehlichPolaron1D{Float64}(addr2; mass=1)
    @test_throws ArgumentError FroehlichPolaron1D{Int}(addr2)
    @test_throws ArgumentError FroehlichPolaron1D(addr2; l=-1)

    # passing alpha
    f_alpha = FroehlichPolaron1D(addr2; alpha=2.0, l=3, two_m=4, omega=5.0)
    @test f_alpha.v^2 ≈ 2 * 2 * 5.0^2/(3 * sqrt(4 * 5.0))
    # v^2 = 2 * alpha * omega^2 / (l * sqrt(2 m omega))

    # test ks vector
    step = (2π/3)
    ks2 = (3/1)*range(-π*(1+1/3) +  step; step=step, length=3)
    @test Vector(f2.ks) == ks2
    step = (2π/4)
    ks3 = (4/1)*range(-π+step; step=step, length=4)
    @test Vector(f3.ks) == ks3

    # test num_offdiagonals
    @test num_offdiagonals(f2, addr1) == 2*3

    # test diagonal_element
    f2_diag = f2.omega*6 + (1/f2.two_m) * (f2.p - dot(f2.ks, onr(addr2)))^2
    @test diagonal_element(f2, addr2) == f2_diag

    # test offdiagonal element
    f2_offdiag = (BoseFS{missing}(1,3,3), -f2.v*sqrt(3))
    @test get_offdiagonal(f2, addr2, 2) == f2_offdiag

    f3_offdiag = (BoseFS{missing}(1,2,3,3), -f3.v*sqrt(4))
    @test get_offdiagonal(f3, addr3, 8) == f3_offdiag

    # test mode_cutoff
    @test get_offdiagonal(f2, BoseFS{missing}(10,3,4), 1)[2] ≠ 0.0
    @test get_offdiagonal(f3, BoseFS{missing}(1,3,20,10), 3)[2] == 0.0

    # test momentum_cutoff
    # addr2 has momentum 12.56
    addr4 = BoseFS{missing}(1,2,1)
    f4 = FroehlichPolaron1D(addr4; momentum_cutoff=10.0)
    @test get_offdiagonal(f4, addr2, 3)[2] == 0.0

    m = 5; l = 6
    addr5 = BoseFS{missing,5}()
    mom_unit = 2π/l
    momentum_cutoff = 1.5 * mom_unit
    f5 = FroehlichPolaron1D(addr5; l, mode_cutoff=1, momentum_cutoff)
    basis5 = build_basis(f5)
    mom_vec = map(o -> dot(o, f5.ks), onr.(basis5))
    @test all(abs.(mom_vec) .≤ momentum_cutoff) ==  true
    @test length(basis5) == 20

    # with and without momentum cutoff
    f6 = FroehlichPolaron1D(addr5; v=10, mode_cutoff=1)
    f7 = FroehlichPolaron1D(addr5; v=10, mode_cutoff=1, momentum_cutoff=100)
    @test get_offdiagonal(f6, addr5, 1) == get_offdiagonal(f7, addr5, 1)
end

@testset "FroehlichPolaron" begin
    addr1 = BoseFS{missing}(1,1,1)

    # test momentum_cutoff, mode_cutoff and dimention when initialising
    addr2 = BoseFS{missing}(1,2,3)
    @test_throws ArgumentError FroehlichPolaron(addr2; mode_cutoff=1.0)
    @test_throws ArgumentError FroehlichPolaron(addr2; momentum_cutoff=10.0)
    f = FroehlichPolaron(addr2; two_m=3)
    @test f == @test_logs (:warn,) FroehlichPolaron(addr2; mass=3)
    @test f == @test_logs (:warn,) FroehlichPolaron{Float64}(addr2; mass=3)
    @test_throws ArgumentError FroehlichPolaron(BoseFS{missing}(3, 2, 1); momentum_cutoff=10.0)
    @test_throws ArgumentError FroehlichPolaron(BoseFS{missing}(3,2,1); D=2)
    @test_throws ArgumentError FroehlichPolaron(BoseFS{missing}(3,2,1,0); D=2, v=1)
    @test_logs (:warn,) FroehlichPolaron(BoseFS{missing}(3,2,1,0); D=1, v=1)

    addr3 = BoseFS{missing}(1,2,3,4)
    f2 = FroehlichPolaron(addr2)
    @test maximum_mode_occupation(f2) == 255
    f3 = FroehlichPolaron(addr3; mode_cutoff=20.0)
    @test maximum_mode_occupation(f3) == 20

    @test starting_address(f2) == f2.address == addr2

    # test ks vector
    step = (2π/3)
    ks2 = [(3/1) * [k] for k in range(-π*(1+1/3) + step; step=step, length=3)]
    @test Vector(f2.ks) == ks2
    step = (2π/4)
    ks3 = [(4/1)* [k] for k in range(-π+step; step=step, length=4)]
    @test Vector(f3.ks) == ks3

    # test a two-dimensional square lattice
    addr_2d = BoseFS{missing}(0, 0, 0, 0)
    f_2d = FroehlichPolaron(addr_2d; D=2, alpha=2, two_m=4, omega=3, l=2, p=[1, -2])
    @test Vector(f_2d.ks) == [[0, 0], [π, 0], [0, π], [π, π]]

    addr_2d_occupied = BoseFS{missing}(1, 2, 0, 0)
    f_2d_diag = f_2d.omega * 3 + norm(f_2d.p - f_2d.ks[1] - 2f_2d.ks[2])^2 / f_2d.two_m
    @test diagonal_element(f_2d * addr_2d_occupied) == f_2d_diag

    offd_2d = offdiagonals(operator_column(f_2d, addr_2d))
    @test last(offd_2d[1]) == 0
    @test first(offd_2d[2]) == BoseFS{missing}(0, 1, 0, 0)
    @test last(offd_2d[2]) ≈ -sqrt(
        2π * f_2d.alpha * f_2d.omega^2 /
        (sqrt(f_2d.two_m * f_2d.omega) * f_2d.l^2 * norm(f_2d.ks[2]))
    )
    @test abs(last(offd_2d[4])) < abs(last(offd_2d[2]))

    f_2d_cutoff = FroehlichPolaron(addr_2d; D=2, l=2, momentum_cutoff=π / 2)
    @test last(offdiagonals(operator_column(f_2d_cutoff, addr_2d))[2]) == 0

    f_2d_twist = FroehlichPolaron(addr_2d; D=2, l=2, twist=[1 / 2, 1 / 4])
    @test Vector(f_2d_twist.ks) == [
        π * [1 / 2, 1 / 4], π * [3 / 2, 1 / 4],
        π * [1 / 2, 5 / 4], π * [3 / 2, 5 / 4],
    ]

    # test num_offdiagonals
    @test num_offdiagonals(operator_column(f2, addr1)) == 2*3

    # test diagonal_element
    f2_diag = f2.omega * 6 + norm(f2.p - sum(f2.ks .* onr(addr2)))^2 / f2.two_m
    @test diagonal_element(f2*addr2) == f2_diag

    # test offdiagonal element
    offd = offdiagonals(operator_column(f2,addr2))
    @test (offd[2]) == (BoseFS{missing}(1,3,3) => -(2* f2.alpha /(f2.l[1]))^0.5 *sqrt(3))

    f3_offdiag = (BoseFS{missing}(1,2,3,3) => -(2* f2.alpha /(f2.l[1]))^0.5*sqrt(4))
    offdf3 = offdiagonals(operator_column(f3,addr3))
    @test (offdf3[8]) == f3_offdiag

    # test mode_cutoff
    offd =offdiagonals(operator_column(f2,BoseFS{missing}(10,3,4)))
    @test offd[1][2] ≠ 0.0
    offdf3 =offdiagonals(operator_column(f3,BoseFS{missing}(1,3,20,10)))
    @test offdf3[3][2] == 0.0

    # test momentum_cutoff
    # addr2 has momentum 12.56
    addr4 = BoseFS{missing}(1,2,1)
    f4 = FroehlichPolaron(addr4; momentum_cutoff=10.0)
    offdf4 = offdiagonals(operator_column(f4,addr2))
    @test offdf4[3][2] == 0.0

    # test basis building with momentum cutoff for D=2
    m = 9; l = 6
    addr5 = BoseFS{missing,m}()
    mom_unit = 2π/l
    momentum_cutoff = 1.5 * mom_unit
    f5 = FroehlichPolaron(addr5; D=2, l, mode_cutoff=1, momentum_cutoff)
    basis5 = build_basis(f5)
    mom_vec = map(o -> sum(f5.ks .* o), onr.(basis5))
    @test all(norm.(mom_vec) .≤ momentum_cutoff) == true
    @test length(basis5) == 152

    # with and without momentum cutoff
    f6 = FroehlichPolaron(addr5;  mode_cutoff=1)
    f7 = FroehlichPolaron(addr5;  mode_cutoff=1, momentum_cutoff=100)
    offdf6 = offdiagonals(operator_column(f6, addr5))
    offdf7 = offdiagonals(operator_column(f7, addr5))
    @test offdf6[1] == offdf7[1]
end

"""
    compare_to_bethe(g, nf, m)

Compare transcorrelated numbers to numbers you get form Bethe ansatz.
"""
function compare_to_bethe(g, nf, m; hamiltonian=Transcorrelated1D, kwargs...)
    if nf == 2
        f1 = f2 = FermiFS([i == cld(m, 2) ? 1 : 0 for i in 1:m])
        exact = g == 10 ? 5.2187287509452015 : g == -10 ? -25.640329369393125 : error()
    elseif nf == 3
        f1 = FermiFS([i == cld(m, 2) || i == cld(m, 2) + 1 ? 1 : 0 for i in 1:m])
        f2 = FermiFS([i == cld(m, 2) ? 1 : 0 for i in 1:m])
        exact = g == -10 ? -15.151863462651115 : error()
    elseif nf == 6
        f1 = f2 = FermiFS([abs(i - cld(m, 2)) ≤ 1 ? 1 : 0 for i in 1:m])
        exact = g == 10 ? 148.90448481827905 : g == -10 ? -43.819879567678 : error()
    else
        error()
    end

    t = m^2/2
    v = t*2/m*g
    c = CompositeFS(f1,f2)
    if hamiltonian == Transcorrelated1D
        ham = Transcorrelated1D(c; t, v, kwargs...)
    elseif hamiltonian == HubbardMom1D
        ham = HubbardMom1D(c; t, u=v, dispersion=continuum_dispersion, kwargs...)
    else
        error()
    end
    energy = eigen(Matrix(ham)).values[1]
    return abs(energy - exact)
end

@testset "Transcorrelated1D" begin
    @testset "Bethe ansatz energies" begin
        @test compare_to_bethe(10, 2, 7) < 0.03
        @test compare_to_bethe(-10, 2, 7) ≤ 0.02
        @test compare_to_bethe(-10, 3, 7) ≤ 0.06
        @test compare_to_bethe(10, 6, 7) < 1.5
        @test compare_to_bethe(-10, 6, 7) < 0.4

        @test compare_to_bethe(-10, 3, 6) < compare_to_bethe(-10, 3, 7)
    end
    @testset "very high cutoff" begin
        # When setting a high cutoff, the differences between
        # Transcorrelated and HubbardMom1D become small.
        f1 = FermiFS((0,1,0,1,0))
        f2 = FermiFS((0,0,1,0,0))
        c = CompositeFS(f1, f2)
        h_trans_cut = Transcorrelated1D(c; cutoff=100_000, v=15)
        h_trans = Transcorrelated1D(c; v=15)
        h_mom = HubbardMom1D(c; u=15, dispersion=continuum_dispersion)

        @test exact_energy(h_trans) ≉ exact_energy(h_mom)
        @test exact_energy(h_trans_cut) ≈ exact_energy(h_mom)

        normal_error = compare_to_bethe(10, 6, 7)
        cutoff_error = compare_to_bethe(10, 6, 7; cutoff=3)
        @test normal_error < cutoff_error < 2 * normal_error
    end
    @testset "no three body term" begin
        f1 = FermiFS((0,1,0,1,0))
        f2 = FermiFS((0,0,1,0,1))
        c = CompositeFS(f1, f2)
        h_trans = Transcorrelated1D(c)
        h_trans_no3b = Transcorrelated1D(c; three_body_term=false)

        @test length(offdiagonals(h_trans, c)) > length(offdiagonals(h_trans_no3b, c))

        @test compare_to_bethe(-10, 3, 20) <
            compare_to_bethe(-10, 3, 20; three_body_term=false, cutoff=4) <
            compare_to_bethe(-10, 3, 20; three_body_term=false, cutoff=3) <
            compare_to_bethe(-10, 3, 20; hamiltonian=HubbardMom1D)
    end
    @testset "non-interacting with potential" begin
        f1 = FermiFS((0,0,1,0,1,0))
        f2 = FermiFS((0,0,0,1,0,1))
        c = CompositeFS(f1, f2)
        h_trans = Transcorrelated1D(c; v=0, v_ho=4)
        h_mom = HubbardMom1DEP(c; u=0, dispersion=continuum_dispersion, v_ho=4)

        @test exact_energy(h_trans) ≈ exact_energy(h_mom)
    end
    @testset "matrix size / folding" begin
        f1 = FermiFS((0,1,0,1,0,0)) # using even number of sites: folding changes things
        f2 = FermiFS((0,0,1,0,1,0))
        c = CompositeFS(f1, f2)
        h_trans = Transcorrelated1D(c; v=-3)
        h_mom = HubbardMom1D(c; u=-3, dispersion=continuum_dispersion)

        @test size(sparse(h_trans))[1] < size(sparse(h_mom))[1]
    end
end

@testset "ParitySymmetry" begin
    @test_throws ArgumentError ParitySymmetry(HubbardMom1D(BoseFS((1, 1))))
    @test_throws ArgumentError ParitySymmetry(HubbardMom1D(BoseFS((1, 1, 1))); even=false)

    @testset "HubbardMom1D" begin
        ham = HubbardMom1D(BoseFS((1, 0, 1, 2, 0)))
        even = ParitySymmetry(ham; odd=false)
        odd = ParitySymmetry(ham; even=false)

        ham_m = Matrix(ham)
        even_m = Matrix(even)
        odd_m = Matrix(odd)

        @test sort(vcat(eigvals(even_m), eigvals(odd_m))) ≈ eigvals(ham_m)
        @test issymmetric(even_m)
        @test issymmetric(odd_m)
    end
    @testset "2-particle HubbardMom1DEP" begin
        ham = HubbardMom1DEP(BoseFS((0,0,1,1,0)))
        even = ParitySymmetry(ham)
        odd = ParitySymmetry(ham; even=false)

        h_eigs = eigvals(Matrix(ham))
        p_eigs = sort!(vcat(eigvals(Matrix(even)), eigvals(Matrix(odd))))

        @test starting_address(even) == reverse(starting_address(ham))
        @test h_eigs ≈ p_eigs
    end
    @testset "Multicomponent" begin
        ham = HubbardRealSpace(
            CompositeFS(FermiFS((1,1,0)), FermiFS((1,0,0)), BoseFS((0,0,2)))
        )
        even_b = BasisSetRepresentation(ParitySymmetry(ham))
        odd_b = BasisSetRepresentation(ParitySymmetry(ham; odd=true))

        for address in even_b.basis
            @test address == min(address, reverse(address))
        end
        for address in odd_b.basis
            @test address == min(address, reverse(address))
            @test address ≠ reverse(address)
        end

        ham_m = Matrix(ham)
        even_m = Matrix(even_b)
        odd_m = Matrix(odd_b)

        @test size(ham_m, 1) == size(even_m, 1) + size(odd_m, 1)
        @test sort(real.(vcat(eigvals(even_m), eigvals(odd_m)))) ≈ real.(eigvals(ham_m))
        @test issymmetric(even_m)
        @test issymmetric(odd_m)
    end
    @testset "Even Hamiltonian" begin
        # This Hamiltonian only has even addresses.
        ham = HubbardMom1D(BoseFS((0,0,0,2,0,0,0)); u=3)
        even_b = BasisSetRepresentation(ParitySymmetry(ham))

        ham_m = Matrix(ham)
        even_m = Matrix(even_b)

        @test ham_m == even_m
        @test issymmetric(even_m)
    end
end

@testset "TimeReversalSymmetry" begin
    @test_throws ArgumentError TimeReversalSymmetry(HubbardMom1D(BoseFS((1, 1))))
    @test_throws ArgumentError begin
        TimeReversalSymmetry(HubbardRealSpace(CompositeFS(FermiFS((1, 1)),BoseFS((2,1)))))
    end
    @test_throws ArgumentError TimeReversalSymmetry(HubbardMom1D(FermiFS2C((1,0,1),(1,0,1)));odd=true)
    @test_throws ArgumentError TimeReversalSymmetry(HubbardMom1D(FermiFS2C((1,0,1),(1,0,1)); u=2+3im))

    @testset "HubbardMom1D" begin
        ham = HubbardMom1D(FermiFS2C((1,0,1),(0,1,1)))
        even = TimeReversalSymmetry(ham; odd=false)
        odd = TimeReversalSymmetry(ham; even=false)

        ham_m = Matrix(ham)
        even_m = Matrix(even)
        odd_m = Matrix(odd)

        @test sort(vcat(eigvals(even_m), eigvals(odd_m))) ≈ eigvals(ham_m)
        @test issymmetric(even_m)
        @test issymmetric(odd_m)
    end
end

@testset "Stoquastic" begin
    ham = HubbardMom1D(BoseFS((0,5,0))) # a Hamiltonian that has a sign problem
    sham = Stoquastic(ham) # sign problem removed, but smaller ground state eigenvalue
    stoquastic_gap = eigvals(Matrix(ham))[1] - eigvals(Matrix(sham))[1]
    @test stoquastic_gap > 0
    tc_ham = Transcorrelated1D(FermiFS2C((1,1,0),(1,0,1)))
    @test LOStructure(Stoquastic(tc_ham)) == AdjointUnknown()
    @test LOStructure(Stoquastic(G2RealCorrelator(2))) == IsDiagonal()
end

@testset "Harmonic oscillator in Cartesian basis" begin
    @testset "HOCartesianContactInteractions" begin
        # argument checks
        @test_throws ArgumentError HOCartesianContactInteractions(BoseFS(4, 1=>1); S = (5,))
        @test_throws ArgumentError HOCartesianContactInteractions(BoseFS(4, 1=>1); S = (4,), η = (2,3))

        N = 3
        D = 2
        M = 4
        S = ntuple(_ -> M + 1, D)
        addr = BoseFS(prod(S), 1 => N)
        H = HOCartesianContactInteractions(addr; S)
        E0 = Hamiltonians.noninteracting_energy(H, addr)
        @test N*D/2 == E0
        @test diagonal_element(H, BoseFS(prod(S), (1,2,3) .=> 1)) ≈ 6.4177817256162255

        block_df = get_all_blocks(H, max_energy = E0 + M)
        @test length(block_df[:,:block_E0]) == 9
        @test Int.(block_df[:,:block_E0]) == [3,4,5,6,7,4,5,6,7]
        @test block_df[:,:block_size] == [1,1,4,7,16,1,2,7,12]

        # interaction matrix elements
        @test count(H.vtable .== 0) == 312
        @test sum(H.vtable) ≈ 11.220010295489221

        # offdiagonals interface
        @test num_offdiagonals(H, addr) == dimension(H) - 1

        h = offdiagonals(H, addr)
        @test Base.eltype(h) == Tuple{typeof(addr),eltype(H)}
        @test Base.IteratorSize(h) == Base.SizeUnknown()
        @test_throws ErrorException getindex(h,1)
        @test_throws ErrorException size(h)
        @test_throws ErrorException length(h)

        next_state = (1,1,3)
        @test iterate(h) == ((addr,0.0), next_state)
        @test isnothing(iterate(h, next_state))

        # block_by_level = false
        H = HOCartesianContactInteractions(addr; S, block_by_level = false)
        all_offs = collect(offdiagonals(H, addr))
        @test length(all_offs) == 169
        @test sum(o -> o[2], all_offs) ≈ 0.3151984121740107

        # aspect ratio
        S = (4,2,2)
        addr = BoseFS(prod(S), 1 => 1)
        H = HOCartesianContactInteractions(addr; S)
        @test H.aspect == (1,3,3)
        @test H.aspect1 == (1.0,3.0,3.0)
        H = HOCartesianContactInteractions(addr; S, η = (1,2,3))
        @test H.aspect == (1,3,3)
        @test H.aspect1 == (1.0,2.0,3.0)
        H = HOCartesianContactInteractions(addr; S, η = 2)
        @test H.aspect == (1,3,3)
        @test H.aspect1 == (1.0,2.0,2.0)

        S = (4,4)
        H = HOCartesianContactInteractions(addr; S)
        b1 = Hamiltonians.find_Ebounds(3, 2, S, Hamiltonians.box_to_aspect(S))
        b2 = Hamiltonians.find_Ebounds(3, 2, S, H.aspect)
        @test b1 == b2
        @test !(b1 === b2)

        @test eval(Meta.parse(repr(H))) == H
    end

    @testset "HOCartesianEnergyConservedPerDim" begin
        # argument checks
        # @test_logs (:warn,) HOCartesianEnergyConservedPerDim(BoseFS(12, 1=>1); S = (3,4))
        @test_throws ArgumentError HOCartesianEnergyConservedPerDim(BoseFS(4, 1=>1); S = (5,))
        @test_throws ArgumentError HOCartesianEnergyConservedPerDim(BoseFS(4, 1=>1); S = (4,), η = (2,3))

        N = 3
        D = 2
        M = 4
        S = ntuple(_ -> M + 1, D)
        addr = BoseFS(prod(S), 1 => N)
        H = HOCartesianEnergyConservedPerDim(addr; S)
        E0 = Hamiltonians.noninteracting_energy(H, addr)
        @test N*D/2 == E0

        block_df = get_all_blocks(H, max_energy = E0 + M)
        @test length(block_df[:,:block_E0]) == 15
        @test Int.(block_df[:,:block_E0]) == [3,4,5,6,7,4,5,6,7,5,6,7,6,7,7]
        @test block_df[:,:block_size] == [1,1,2,3,4,1,2,4,6,2,4,8,3,6,4]

        # interaction matrix elements
        @test count(H.vtable .== 0) == 70
        @test sum(H.vtable) ≈ 2 * 3.3630246382916664

        # aspect ratio
        S = (4,2,2)
        addr = BoseFS(prod(S), 1 => 1)
        H = HOCartesianEnergyConservedPerDim(addr; S)
        @test H.aspect1 == (1.0,3.0,3.0)
        H = HOCartesianEnergyConservedPerDim(addr; S, η = (1,2,3))
        @test H.aspect1 == (1.0,2.0,3.0)
        H = HOCartesianEnergyConservedPerDim(addr; S, η = 2)
        @test H.aspect1 == (1.0,2.0,2.0)

        @test eval(Meta.parse(repr(H))) == H
    end

    @testset "HOCartesianCentralImpurity" begin
        # argument checks
        @test_throws ArgumentError HOCartesianCentralImpurity(BoseFS(4, 1=>1); max_nx = 1)
        @test_throws ArgumentError HOCartesianCentralImpurity(BoseFS(4, 1=>1); max_nx = 2)
        @test_throws ArgumentError HOCartesianCentralImpurity(BoseFS(4, 1=>1); max_nx = 4, ηs = (0.5,))

        N = 1
        M = 8
        ηs = (2,)
        P = prod(x -> M÷x + 1, (1,ηs...))
        addr = BoseFS(P, 1 => N)
        H = HOCartesianCentralImpurity(addr; max_nx = M, ηs)
        @test H.aspect == (1.0, float.(ηs)...)

        G = HOCartesianCentralImpurity(addr; S = H.S, ηs)
        @test G == H

        # interaction matrix elements
        @test length(H.vtable) == M÷2 + 1     # 5
        @test sum(H.vtable) ≈ -3.497817080215528

        bsr = BasisSetRepresentation(H; sizelim=Inf)
        @test dimension(bsr) == 15  # dimension(bsr) < dimension(H)
        @test sum(bsr.sparse_matrix) ≈ 142.6393438659114

        @test eval(Meta.parse(repr(H))) == H
    end

    @testset "Angular momentum" begin
        @test_throws ArgumentError AxialAngularMomentumHO((2,); addr = BoseFS(2))
        @test_throws ArgumentError AxialAngularMomentumHO((1,2,3); addr = BoseFS(6))

        S = (3,3,3)
        addr = BoseFS(prod(S), 3 => 2)

        Lz = AxialAngularMomentumHO(S; addr)
        Ly = AxialAngularMomentumHO(S; z_dim=2, addr)
        Lx = AxialAngularMomentumHO(S; z_dim=1, addr)

        Lz_vals = eigvals(Matrix(BasisSetRepresentation(Lz)))
        Ly_vals = eigvals(Matrix(BasisSetRepresentation(Ly)))
        Lx_vals = eigvals(Matrix(BasisSetRepresentation(Lx)))

        expected = [-4, -2, 0, 0, 2, 4]
        @test Lz_vals ≈ expected
        @test Ly_vals ≈ expected
        @test Lx_vals == [0.0]  # initial state is excited purely in x dimension
    end

    @testset "find blocks" begin
        N = 2
        D = 2
        M = 4
        S = ntuple(_ -> M + 1, D)
        addr = BoseFS(prod(S), 1 => N)
        H = HOCartesianEnergyConservedPerDim(addr; S)
        block_df_vert = get_all_blocks(H; max_energy = N*D/2 + M, method = :vertices)
        block_df_comb = get_all_blocks(H; max_energy = N*D/2 + M, method = :comb)

        # different methods find the same blocks but with different key addresses
        vert_blocks = block_df_vert[!,[:block_E0,:block_size]]
        comb_blocks = block_df_comb[!,[:block_E0,:block_size]]
        @test vert_blocks == comb_blocks

        @test nrow(get_all_blocks(H, max_blocks = 5)) == 5
        @test nrow(get_all_blocks(H, max_blocks = 5, method = :comb)) == 5

        @test nrow(
            @test_logs (:warn,) get_all_blocks(H; target_energy = 100)
        ) == 0
        @test nrow(
            @test_logs (:warn,) get_all_blocks(H; max_energy = 1)
        ) == 0
        @test nrow(
            @test_logs (:warn,) get_all_blocks(H; max_energy = 3, target_energy = 4)
        ) == 0

        df = get_all_blocks(H; save_to_file = "test_block_df.arrow")
        df_file = load_df("test_block_df.arrow")
        @test df[!,[1,2,3,5]] == df_file[!,[1,2,3,5]]

        # HOCartesianContactInteractions requires a valid energy restriction
        @test_throws ArgumentError get_all_blocks(HOCartesianContactInteractions(addr; S))

        # block_by_level = false
        H = HOCartesianContactInteractions(addr; S, block_by_level = false)
        df = get_all_blocks(H)
        @test nrow(df) == 2^D
    end

    @testset "vertices" begin
        n = 3
        for k in 0:n
            @test Hamiltonians._binomial(n, Val(k)) == Base.binomial(n, k)
        end

        @test_throws OverflowError Hamiltonians._first_vertex(n, Val(0))
        @test Hamiltonians._first_vertex(n, Val(1), 0, 0) == n

        @test Hamiltonians.vertices(1, Val(3)) == (3,2,1)
        @test Hamiltonians.vertices(10, Val(3)) == (5,4,3)
        @test Hamiltonians.vertices(n, Val(1)) == (n,)

        @test Hamiltonians.index((3,2,1)) == 1
        @test Hamiltonians.index((5,4,3)) == 10
    end

    @testset "HO utilities" begin
        S = (4,4)
        @test_throws ArgumentError fock_to_cart(BoseFS(1, 1 => 1), S)
        modes = [5, 5, 16]
        addr = BoseFS(prod(S), modes .=> 1)
        @test fock_to_cart(addr, S) == [(0, 1), (0, 1), (3, 3)]
        @test fock_to_cart(addr, S; zero_index = false) == [(1, 2), (1, 2), (4, 4)]

        null_addr = BoseFS(prod(S), 1=>0)
        @test isempty(fock_to_cart(null_addr, S))
    end
end

@testset "dimension and multi-component addresses" begin
    addresses = [
        CompositeFS(FermiFS((1,0,1)), FermiFS((0,1,0))), BoseFS((1,0,1)),
        FermiFS2C((1,0,1), (0,1,0)), BoseFS{missing}(3, 0, 1), HardcoreBoseFS(1,0,1),
        HardcoreBoseFS{missing}(1, 0, 1), FermiFS{missing}(1, 0, 1),
        FermiFS2C{missing}((1, 0, 1), (0, 1, 0))
    ]
    [@test dimension(addr) == dimension(typeof(addr)) for addr in addresses]
    @test dimension(CompositeFS(FermiFS((1,0,1)), FermiFS((0,1,0)))) == 9
    @test dimension(CompositeFS(FermiFS((1,0,1)), FermiFS((0,1,0)), BoseFS((1,0,0)))) == 27
    @test dimension(FermiFS2C{missing}((1, 0, 1), (0, 1, 0))) == 64
    @test dimension(HardcoreBoseFS{missing}(1, 0, 1)) == 8
    @test dimension(FermiFS{missing}(1, 0, 1)) == 8

    h = ExtendedHubbardReal1D(HardcoreBoseFS{missing}(1, 1, 0))
    @test dimension(h) == 3
    @test dimension(starting_address(h)) == 8
end

@testset "ExtendedHubbardReal1D" begin
    @testset "boundary conditions" begin
        for H in (
            ExtendedHubbardReal1D(FermiFS((1,0,1,0)), v=6, t=2.0, boundary_condition=:twisted),
            ExtendedHubbardReal1D(FermiFS((1,0,1,0)), v=6, t=2.0, boundary_condition=:hard_wall),
            ExtendedHubbardReal1D(FermiFS((1,0,1,0,1,0,1,0,1,0,1)), v=6, t=2.0, boundary_condition=:twisted),
            ExtendedHubbardReal1D(FermiFS((1,0,1,0,1,0,1,0,1,0,1)), v=6, t=2.0, boundary_condition=:hard_wall),
            ExtendedHubbardReal1D(FermiFS((1,0,1,0,1,0,1,0,1,0,1)), v=6, t=2.0, boundary_condition=π),
            ExtendedHubbardReal1D(BoseFS(1,0,2,1); u=1+0.5im, boundary_condition=:hard_wall))

            addr = starting_address(H)
            H1 = ExtendedHubbardReal1D(addr, v=6, t=2.0)
            addr2, me = get_offdiagonal(H1, addr, 2)
            if H.boundary_condition == :twisted
                @test get_offdiagonal(H, addr, 2)[2] == - me
                @test diagonal_element(H, addr) == diagonal_element(H1, addr)
            elseif H.boundary_condition == :hard_wall
                @test get_offdiagonal(H, addr, 2)[2] == 0.0
            elseif H.boundary_condition isa Number
                @test get_offdiagonal(H, addr, 2)[2] == me*exp(-im*H.boundary_condition)
                @test diagonal_element(H, addr) == diagonal_element(H1, addr)
            end
            @test get_offdiagonal(H, addr, 2)[1] == addr2
        end
        @test_throws ArgumentError ExtendedHubbardReal1D(BoseFS(1,1,1,1); boundary_condition=:hrad_wall)
    end

    @testset "interaction" begin
        for H in (
            ExtendedHubbardReal1D(FermiFS((1,0,1,0)), v=6, t=2.0, power = nothing),
            ExtendedHubbardReal1D(FermiFS((1,0,1,0)), v=6, t=2.0, power = 3)
        )

            addr = starting_address(H)
            H1 = ExtendedHubbardReal1D(addr; v=6, t=2.0)
            @test get_offdiagonal(H, addr, 2) == get_offdiagonal(H1, addr, 2)
            ebhinteraction, bhinteraction = Hamiltonians.extended_hubbard_interaction(H, addr, H.power)
            @test diagonal_element(H, addr) == convert(eltype(H),  6 * ebhinteraction)
        end
        @test_throws ArgumentError ExtendedHubbardReal1D(BoseFS(1,1,1,1); power = :nearest_neighbor)
    end
end

@testset "Small 1D Hubbard with complex parameters" begin
    for H in (
        ExtendedHubbardReal1D(FermiFS(1, 0, 1, 0), boundary_condition=:twisted), # Hermitian
        ExtendedHubbardReal1D(FermiFS(1, 0, 1, 0), boundary_condition=0.5), # Hermitian
        ExtendedHubbardReal1D(FermiFS(1, 0, 1, 0), t=2.0+3im, power=3), # Hermitian
        ExtendedHubbardReal1D(BoseFS(1, 0, 1, 0), v=6, t=2.0+3im), # Hermitian
        ExtendedHubbardReal1D(FermiFS(1, 0, 1, 0), v=6 + 0.5im, t=2.0), # non-Hermitian
        ExtendedHubbardReal1D(BoseFS{missing}(3, 0, 1), u=6 + 3im, t=2.0), # non-Hermitian
        ExtendedHubbardReal1D(BoseFS{missing}(3, 0, 1), u=6 + 3im, t=0), # diagonal and non-Hermitian
        ExtendedHubbardReal1D(BoseFS{missing}(3, 0, 1), t=0), # diagonal and Hermitian
        HubbardReal1D(BoseFS(1,1,1),t=1.0im),
        HubbardReal1D(BoseFS(1,1,1),u=1.0im),
        ExtendedHubbardReal1D(BoseFS(1,1,1),t=1.0im),
        ExtendedHubbardReal1D(BoseFS(1,1,1),u=1.0im),
    )
        test_hamiltonian_structure(H)
    end
    h = ExtendedHubbardReal1D(BoseFS{missing}(3, 0, 1); t=0) # diagonal and Hermitian
    @test LOStructure(h) isa IsDiagonal
    @test adjoint(h) == h
    h2 = ExtendedHubbardReal1D(BoseFS{missing}(3, 0, 1); u=6 + 3im, t=0)
    # diagonal and non-Hermitian
    @test LOStructure(h2) isa AdjointKnown
    @test h2'.u == conj(h2.u)
    @test diagonal_element(h2, BoseFS{missing}(3,0,1)) == 21 + 9im
    start_at = DVec(BoseFS{missing}(3,0,1) => 1)
    @test_throws ArgumentError ProjectorMonteCarloProblem(h2; start_at)
    start_at = [DVec(BoseFS{missing}(3,0,1) => 1) DVec(BoseFS{missing}(3,0,1) => 1)]
    @test_throws ArgumentError ProjectorMonteCarloProblem(h2; start_at, n_spectral=2)
    h3 = HubbardReal1D(BoseFS(1,1,1),u=1.0im)
    @test h3'.u == -1.0im
    @test diagonal_element(h3, BoseFS(2,1,0)) == 1.0im
    @test diagonal_element(h3', BoseFS(2,1,0)) == -1.0im
    h4 = HubbardReal1DEP(BoseFS(1,1,1),u=1.0im,v_ho=1.0)
    @test h4'.u == -1.0im
    @test diagonal_element(h4, BoseFS(0,1,2)) == 3 + 1.0im
    @test diagonal_element(h4', BoseFS(0,1,2)) == 3 - 1.0im
end

@testset "Comparison of ExtendedHubbardMom1D with ExtendedHubbardReal1D" begin
    addr_f = FermiFS{3,6}(0,1,1,1,0,0)
    addr_b = BoseFS{3,6}(0,0,3,0,0,0)
    for boundary_condition in ([i*π for i in 0.0:0.2:1.0]...,)
        HR_f = ExtendedHubbardReal1D(addr_f; boundary_condition)
        HM_f = ExtendedHubbardMom1D(addr_f; boundary_condition)
        HR_b = ExtendedHubbardReal1D(addr_b; boundary_condition)
        HM_b = ExtendedHubbardMom1D(addr_b; boundary_condition)
        @test round.(eigvals(Matrix(HM_f)), digits=8) ⊆ round.(eigvals(Matrix(HR_f)), digits=8)
        @test round.(eigvals(Matrix(HM_b)), digits=8) ⊆ round.(eigvals(Matrix(HR_b)), digits=8)
    end
end

@testset "ReducedDensityMatrix" begin
    dvec_f = PDVec(FermiFS{2,4}(1,1,0,0) => 0.5, FermiFS{2,4}(0,0,1,1)=>0.5)
    dvec_b = PDVec(BoseFS{4,4}(0,0,2,2) => 0.5, BoseFS{4,4}(2,2,0,0)=>0.5)
    op = ReducedDensityMatrix(1)
    spd_b = zeros(4,4)
    spd_f = zeros(4,4)
    for i in 1:4, j in 1:4
        spd_b[i,j] = dot(dvec_b, SingleParticleExcitation(i, j), dvec_b)
        spd_f[i,j] = dot(dvec_f, SingleParticleExcitation(i, j), dvec_f)
    end
    tpd_f = zeros(6,6)
    t1 = 0; t2 = 0
    for i in 1:4, j in i+1:4;
        t1 += 1; t2 = 0;
        for k in 1:4, l in k+1:4
            t2+=1; tpd_f[t1,t2] = dot(dvec_f, TwoParticleExcitation(i, j, k, l), dvec_f);
        end
    end
    @test dot(dvec_f, op, dvec_f) == spd_f
    @test dot(dvec_b, op, dvec_b) == spd_b
    @test dot(dvec_f, ReducedDensityMatrix(2), dvec_f) == tpd_f
    @test_throws ArgumentError dot(dvec_b, ReducedDensityMatrix(2), dvec_b)
    @test LOStructure(op) isa IsHermitian
    test_observable_interface(ReducedDensityMatrix(1), BoseFS{4,4}(2,2,0,0))
    test_observable_interface(ReducedDensityMatrix(2), FermiFS{2,4}(1,1,0,0))
    for r in (ReducedDensityMatrix(1), ReducedDensityMatrix{ComplexF32}(2))
        # Check that the result of show can be pasted into the REPL
        @test eval(Meta.parse(repr(r))) == r
    end
    # complex hermitian Hamiltonian still produces approx hermitian RDM
    H = HubbardReal1D(BoseFS(0,1,2,0); t = 1+im)
    res = solve(ExactDiagonalizationProblem(H))
    gs = res.vectors[1]
    rdm = ReducedDensityMatrix{ComplexF64}(1)
    m = dot(gs, rdm, gs)
    @test all(x -> abs(x) < √eps(Float64), m - m') # hermitian up to floating point noise

    # a global relative phase in the vectors results in a global phase in the RDM
    m_phase = dot(im * gs, rdm, gs)
    @test all(x -> abs(x) < √eps(Float64), m_phase + im * m)

    # complex non-hermitian Hamiltonian still produces approx hermitian RDM
    Hc = HubbardReal1D(BoseFS(0,1,2,0); u = 1+im)
    resc = solve(ExactDiagonalizationProblem(Hc))
    gsc = resc.vectors[1]
    mc = dot(gsc, rdm, gsc)
    @test all(x -> abs(x) < √eps(Float64), mc - mc') # hermitian up to floating point noise
end

@testset "HamiltonianProduct" begin
    addr = BoseFS(2,0,0)

    H = HubbardReal1D(addr)
    start_at = DVec(addr => 10; style=IsStochasticWithThreshold(0.1))
    P = H*H
    problem = ProjectorMonteCarloProblem(P; start_at, last_step=10000, target_walkers=10000)
    df = DataFrame(solve(problem))
    energy = shift_estimator(df; skip=5000)
    @test energy.mean ≈ eigvals(Matrix(P))[1] atol=5*energy.err

    H1 = HubbardReal1D(addr;u=1.0im)
    H2 = ExtendedHubbardReal1D(addr)
    @test LOStructure(H2*H2) == IsHermitian()
    P = H1*H2
    @test LOStructure(P) == AdjointKnown()
    c = operator_column(P, addr)

    for _ in 1:20
        a, p, v = random_offdiagonal(c)
        @test (a => v) in offdiagonals(c)
    end

    ods_product = sum(DVec(od) for od in offdiagonals(c))
    ods_product[addr] += diagonal_element(c)

    c2 = operator_column(H2, addr)
    c1 = operator_column(H1, addr)
    ods_manual = DVec(addr => diagonal_element(c2)*diagonal_element(c1))
    for (add1, val1) in offdiagonals(c1)
        ods_manual += DVec(add1 => val1*diagonal_element(c2))
    end
    for (add2, val2) in offdiagonals(c2)
        c1 = operator_column(H1, add2)
        ods_manual += DVec(add2 => val2*diagonal_element(c1))
        for (add1, val1) in offdiagonals(c1)
            ods_manual += DVec(add1 => val1*val2)
        end
    end
    @test ods_product == ods_manual

    basis = build_basis(addr)
    @test Matrix(H1, basis) * Matrix(H2, basis) ≈ Matrix(H1 * H2, basis)

    addr = FermiFS(1,0,0)
    H3 = HubbardReal1D(addr)
    @test_throws ArgumentError H1*H3

    addr = FermiFS(1,1,1)
    H4 = HubbardReal1D(addr)
    P = H4*H4
    c = operator_column(P, addr)
    @test iszero(last.(collect(offdiagonals(c))))

    @testset "ScaledOrShifted scaling" begin
        addr = BoseFS(2,0,0)
        basis = build_basis(addr)
        H = HubbardReal1D(addr)

        H1 = @inferred 2*H
        @test Matrix(H1) == 2*Matrix(H)
        @test LOStructure(H1) == LOStructure(H)
        H1_twice = 2 * H1
        @test H1_twice isa ScaledOrShiftedHamiltonian
        @test Matrix(H1_twice, basis) ≈ 4 * Matrix(H, basis)
        H1_same = 1 * H1
        @test H1_same isa ScaledOrShiftedHamiltonian
        @test Matrix(H1_same, basis) ≈ Matrix(H1, basis)

        H2 = 3im*H
        @test Matrix(H2) == 3im*Matrix(H)
        @test eltype(H2) <: Complex
        @test LOStructure(H2) == AdjointKnown()
        @test H2' == -3im*H
    end
end

@testset "HamiltonianSum" begin
    addr = BoseFS(0,1,2,3,4)
    H1 = HubbardReal1D(addr; u=2, t=2)
    H2 = ExtendedHubbardReal1D(addr; v=3)
    S1 = H1+H2
    @test S1 == add(H1, H2; weight=0.5)
    @test LOStructure(S1) == IsHermitian()

    basis = build_basis(addr)
    @test Matrix(H1, basis) + Matrix(H2, basis) ≈ Matrix(S1, basis)

    result = solve(ExactDiagonalizationProblem(S1))
    energy = result.values[1]
    vec = result.vectors[1]
    @test energy ≈ rayleigh_quotient(H1, vec) + rayleigh_quotient(H2, vec)

    od1 = collect(offdiagonals(H1*addr))
    od2 = collect(offdiagonals(H2*addr))
    col = S1*addr
    odsum = collect(offdiagonals(col))
    @test DVec(od1) + DVec(od2) == DVec(odsum)

    for _ in 1:20
        a, p, v = random_offdiagonal(col)
        @test (a => v) in odsum
    end

    S2 = 2im*H1 + 3*H2
    @test S2 == add(H1, H2, 2im, 3)
    @test LOStructure(S2) == AdjointKnown()
    @test rayleigh_quotient(S2, vec) ≈ 2im*rayleigh_quotient(H1, vec) + 3*rayleigh_quotient(H2, vec)

    H3 = HubbardReal1D(addr; t=0)
    S3 = H3 + H1
    @test DVec(offdiagonals(S3*addr)) == DVec(offdiagonals(H1*addr))

    addr = FermiFS(1,0,0)
    H4 = HubbardReal1D(addr)
    @test_throws ArgumentError H1 + H4
end

@testset "ScaledOrShiftedHamiltonian" begin
    addr = BoseFS(1,1)
    H = HubbardRealSpace(addr)
    basis = build_basis(addr)

    @testset "construction and arithmetic" begin
        S = ScaledOrShiftedHamiltonian(H, 2, -3)
        @test S.alpha == 2
        @test S.beta == -3
        @test parent_operator(S) == H

        Sidentity = @inferred ScaledOrShiftedHamiltonian(H, 1, 0)
        @test Sidentity isa ScaledOrShiftedHamiltonian
        @test Sidentity.alpha == 1
        @test Sidentity.beta == 0
        @test Matrix(Sidentity, basis) ≈ Matrix(H, basis)

        @test (@inferred H - 2I) == ScaledOrShiftedHamiltonian(H, One(), -2)
        @test (@inferred H + 3I) == ScaledOrShiftedHamiltonian(H, One(), 3)
        @test 2 * H == ScaledOrShiftedHamiltonian(H, 2, Zero())
        @test Matrix(H - 2I, basis) ≈ Matrix(H, basis) - 2I
        @test Matrix(2 * H + 3I, basis) ≈ 2 * Matrix(H, basis) + 3I

        @test (@inferred add(H, 2I, -4, 3)) == ScaledOrShiftedHamiltonian(H, 3, -8)
        @test (@inferred add(2I, H, 3, -4)) == ScaledOrShiftedHamiltonian(H, 3, -8)
    end

    @testset "structure and adjoint" begin
        Sreal = ScaledOrShiftedHamiltonian(H, 2.5, -1.5)
        Scomplex = ScaledOrShiftedHamiltonian(H, 1im, 2.0)
        @test LOStructure(Sreal) == IsHermitian()
        @test LOStructure(Scomplex) == AdjointKnown()

        Sadj = Scomplex'
        @test Sadj == ScaledOrShiftedHamiltonian(H', -1im, 2.0)
        @test Matrix(Sadj, basis) ≈ Matrix(Scomplex, basis)'

        Hnh = HubbardReal1D(BoseFS(1,2,3,4); u=1.0im)
        @test LOStructure(ScaledOrShiftedHamiltonian(Hnh, 2.0, 1.0)) == LOStructure(Hnh)
    end

    @testset "nested composition and repr" begin
        S0 = ScaledOrShiftedHamiltonian(H, 2, -1) # 2H - I
        S1 = ScaledOrShiftedHamiltonian(S0, -1, 3) # -(2H - I) + 3 = -2H + 4I
        @test S1 == ScaledOrShiftedHamiltonian(H, -2, 4)
        @test Matrix(S1, basis) ≈ -2 * Matrix(H, basis) + 4I
        @test eval(Meta.parse(repr(S1))) == S1

        Sidentity = ScaledOrShiftedHamiltonian(H, 1, 0)
        Snoop = ScaledOrShiftedHamiltonian(Sidentity, 1, 0)
        @test Snoop isa ScaledOrShiftedHamiltonian
        @test Snoop == Sidentity
        @test eval(Meta.parse(repr(Snoop))) == Snoop
        @test ScaledOrShiftedHamiltonian(H, One(), Zero()) == H
    end
end

@testset "Operator Traits" begin
    struct TestHamiltonian <: Rimu.AbstractHamiltonian{Float64} end
    Rimu.allows_address_type(::TestHamiltonian, ::Type{<:Any}) = true
    struct TestColumn{A,O} <: Rimu.AbstractOperatorColumn{A,Float64,O}
        operator::O
        address::A
    end
    function Rimu.operator_column(h::TestHamiltonian, address)
        return TestColumn(h, address)
    end
    Rimu.parent_operator(c::TestColumn) = c.operator
    Rimu.starting_address(c::TestColumn) = c.address
    h = TestHamiltonian()
    addr = FermiFS{2,4}(1,0,1,0)
    col = operator_column(h, addr)
    @test col == h * addr
    @test has_random_offdiagonal(typeof(h))  # default trait
    @test has_random_offdiagonal(h) # works with instance
    @test has_random_offdiagonal(col) # works with column
    @test_throws MethodError random_offdiagonal(col) # not implemented

    @test has_iterable_offdiagonals(typeof(h))  # default trait
    @test has_iterable_offdiagonals(h) # works with instance
    @test has_iterable_offdiagonals(col) # works with column
    @test_throws MethodError offdiagonals(col) # not implemented

    # Operator
    op = SingleParticleExcitation(1, 2)
    @test has_iterable_offdiagonals(op)
    @test !has_random_offdiagonal(op)
    @test offdiagonals(op * BoseFS(1, 1)) isa AbstractVector
    @test length(offdiagonals(op * BoseFS(1, 1))) == 1

    # Observable
    obs = ReducedDensityMatrix(1)
    @test !has_iterable_offdiagonals(obs)
    @test !has_random_offdiagonal(obs)
end

@testset "AbstractOperatorColumn" begin
    # standard Hamiltonian
    h = HubbardReal1D(BoseFS(1,1,1), t=1.0)
    addr = BoseFS(0,2,1)
    col = operator_column(h, addr)
    @test col == h * addr
    @test parent_operator(col) == h
    @test starting_address(col) == addr
    @test diagonal_element(col) == col[addr]
    @test col[fs"|0 1 2⟩"] == -2 # off-diagonal element
    @test col[fs"|0 0 3⟩"] == 0 # zero off-diagonal element
    @test_throws MethodError col[fs"|0 0 4⟩"] # not in the address space
    @test length(collect(offdiagonals(col))) == 4
    dv = DVec(addr => 1.0)
    @test dv ⋅ col == (col ⋅ dv)' == col[addr]
end
