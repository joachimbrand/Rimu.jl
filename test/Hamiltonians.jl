using KrylovKit
using LinearAlgebra
using Random
using Rimu
using Test
using DataFrames

function exact_energy(ham)
    dv = DVec(starting_address(ham) => 1.0)
    all_results = eigsolve(ham, dv, 1, :SR; issymmetric = LOStructure(ham) == IsHermitian())
    return all_results[1][1]
end

"""
    test_hamiltonian_interface(H, addr)

The main purpose of this test function is to check that all required methods are defined.
"""
function test_hamiltonian_interface(H)
    addr = starting_address(H)
    @testset "$(nameof(typeof(H)))" begin
        @testset "*, mul!, and call" begin
            v = DVec(addr => eltype(H)(2.0))
            v′ = H(v)
            v″ = H * v
            v‴ = similar(v)
            H(v‴, v)
            v⁗ = similar(v)
            mul!(v⁗, H, v)
            @test v′ == v″ == v‴ == v⁗
        end
        @testset "diagonal_element" begin
            if eltype(H) <: Real
                @test diagonal_element(H, addr) isa Real
            else
                @test norm(diagonal_element(H, addr)) ≥ 0
            end
        end
        if !(H isa HOCartesianContactInteractions)  # offdiagonals not consistent with interface
            @testset "hopping" begin
                h = offdiagonals(H, addr)
                @test eltype(h) == Tuple{typeof(addr), eltype(H)}
                @test length(h) == num_offdiagonals(H, addr)
                for i in 1:length(h)
                    @test h[i] == get_offdiagonal(H, addr, i)
                    @test h[i] isa eltype(h)
                end
            end
        end
        @testset "LOStructure" begin
            @test LOStructure(H) isa LOStructure
            if LOStructure(H) isa IsHermitian || LOStructure(H) isa IsDiagonal
                @test H' === H
            elseif LOStructure(H) isa AdjointKnown
                @test begin H'; true; end # make sure no error is thrown
            else
                @test_throws ErrorException H'
            end
        end
        @testset "dimension" begin
            @test dimension(H) ≥ dimension(H, starting_address(H))
            @test dimension(Float64, H) isa Float64
            @test dimension(Int, H) == dimension(H)
        end
        @testset "allowed_address_type" begin
            @test addr isa allowed_address_type(H)
        end
    end
end

using Rimu.Hamiltonians: momentum_transfer_excitation

@testset "momentum_transfer_excitation" begin
    @testset "BoseFS" begin
        add1 = BoseFS((0,1,1,0))
        add2 = BoseFS((1,0,0,1))
        for i in 1:4
            ex = momentum_transfer_excitation(add1, i, OccupiedModeMap(add1); fold=true)
            @test ex[1] == add2
            @test ex[2] == 1

            ex = momentum_transfer_excitation(add1, i, OccupiedModeMap(add1); fold=false)
            @test ex[1] == add2
            @test ex[2] == 1
        end

        add3 = BoseFS((1,1,0,0))
        for i in 1:4
            ex = momentum_transfer_excitation(add3, i, OccupiedModeMap(add3); fold=true)
            @test ex[2] == 1

            ex = momentum_transfer_excitation(add3, i, OccupiedModeMap(add3); fold=false)
            @test ex[2] == 0
        end

        add4 = BoseFS((0,3,0))
        add5 = BoseFS((1,1,1))
        for i in 1:2
            ex = momentum_transfer_excitation(add4, i, OccupiedModeMap(add4); fold=false)
            @test ex[1] == add5
            @test ex[2] ≈ √6

            ex = momentum_transfer_excitation(add4, i, OccupiedModeMap(add4); fold=true)
            @test ex[1] == add5
            @test ex[2] ≈ √6
        end
    end
    @testset "FermiFS" begin
        add1 = FermiFS((0,0,1,0))
        add2 = FermiFS((0,1,0,0))
        occ1 = OccupiedModeMap(add1)
        occ2 = OccupiedModeMap(add2)
        for i in 1:3
            ex = momentum_transfer_excitation(add1, add2, i, occ1, occ2; fold=true)
            @test ex[3] == 1

            ex = momentum_transfer_excitation(add1, add2, i, occ1, occ2; fold=false)
            @test ex[3] == 1
        end

        add3 = FermiFS((1,0,0,0))
        add4 = FermiFS((0,1,0,0))
        occ3 = OccupiedModeMap(add3)
        occ4 = OccupiedModeMap(add4)
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

@testset "Interface tests" begin
    for H in (
        HubbardReal1D(BoseFS((1, 2, 3, 4)); u=1.0, t=2.0),
        HubbardReal1DEP(BoseFS((1, 2, 3, 4)); u=1.0, t=2.0, v_ho=3.0),
        HubbardMom1D(BoseFS((6, 0, 0, 4)); t=1.0, u=0.5),
        HubbardMom1D(BoseFS((6, 0, 0, 4)); t=1.0, u=0.5 + im),
        ExtendedHubbardReal1D(BoseFS((1,0,0,0,1)); u=1.0, v=2.0, t=3.0),
        HubbardRealSpace(BoseFS((1, 2, 3)); u=[1], t=[3]),
        HubbardRealSpace(FermiFS((1, 1, 1, 1, 1, 0, 0, 0)); u=[0], t=[3]),
        HubbardRealSpace(
            CompositeFS(
                FermiFS((1, 1, 1, 1, 1, 0, 0, 0)),
                FermiFS((1, 1, 1, 1, 0, 0, 0, 0)),
            ); t=[1, 2], u=[0 3; 3 0]
        ),

        BoseHubbardReal1D2C(BoseFS2C((1,2,3), (1,0,0))),
        BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0))),

        GutzwillerSampling(HubbardReal1D(BoseFS((1,2,3)); u=6); g=0.3),
        GutzwillerSampling(BoseHubbardMom1D2C(BoseFS2C((3,2,1), (1,2,3)); ua=6); g=0.3),
        GutzwillerSampling(HubbardReal1D(BoseFS((1,2,3)); u=6 + 2im); g=0.3),

        MatrixHamiltonian(Float64[1 2;2 0]),
        GutzwillerSampling(MatrixHamiltonian([1.0 2.0;2.0 0.0]); g=0.3),
        Rimu.Hamiltonians.TransformUndoer(
            GutzwillerSampling(MatrixHamiltonian([1.0 2.0; 2.0 0.0]); g=0.3)
        ),

        Transcorrelated1D(CompositeFS(FermiFS((0,0,1,1,0)), FermiFS((0,1,1,0,0))); t=2),
        Transcorrelated1D(CompositeFS(FermiFS((0,0,1,0)), FermiFS((0,1,1,0))); v=3, v_ho=1),

        HubbardMom1DEP(BoseFS((0,0,5,0,0))),
        HubbardMom1DEP(CompositeFS(FermiFS((0,1,1,0,0)), FermiFS((0,0,1,0,0))), v_ho=5),

        ParitySymmetry(HubbardRealSpace(CompositeFS(BoseFS((1,2,0)), FermiFS((0,1,0))))),
        TimeReversalSymmetry(HubbardMom1D(FermiFS2C((1,0,1),(0,1,1)))),
        TimeReversalSymmetry(BoseHubbardMom1D2C(BoseFS2C((0,1,1),(1,0,1)))),
        Stoquastic(HubbardMom1D(BoseFS((0,5,0)))),
        momentum(HubbardMom1D(BoseFS((0,5,0)))),

        HOCartesianContactInteractions(BoseFS((2,0,0,0))),
        HOCartesianEnergyConservedPerDim(BoseFS((2,0,0,0))),
        HOCartesianCentralImpurity(BoseFS((1,0,0,0,0)))
    )
        test_hamiltonian_interface(H)
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
    @test diagonal_element(HM2Cu0, bs2) ≈ 6*t*(2pi/num_modes(bs2))^2

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

@testset "2C model properties" begin
    flip(b) = BoseFS2C(b.bsb, b.bsa)
    addr1 = near_uniform(BoseFS2C{1,100,20})
    addr2 = near_uniform(BoseFS2C{100,1,20})

    for Hamiltonian in (BoseHubbardReal1D2C, BoseHubbardMom1D2C)
        @testset "$Hamiltonian" begin
            H1 = BoseHubbardReal1D2C(addr1; ta=1.0, tb=2.0, ua=0.5, ub=0.7, v=0.2)
            H2 = BoseHubbardReal1D2C(addr2; ta=2.0, tb=1.0, ua=0.7, ub=0.5, v=0.2)
            @test starting_address(H1) == addr1
            @test LOStructure(H1) == IsHermitian()

            hops1 = collect(offdiagonals(H1, addr1))
            hops2 = collect(offdiagonals(H2, addr2))
            sort!(hops1, by=a -> first(a).bsa)
            sort!(hops2, by=a -> first(a).bsb)

            addrs1 = first.(hops1)
            addrs2 = flip.(first.(hops2))
            values1 = last.(hops1)
            values2 = last.(hops1)
            @test addrs1 == addrs2
            @test values1 == values2

            @test eval(Meta.parse(repr(H1))) == H1
            @test eval(Meta.parse(repr(H2))) == H2
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

        comp = CompositeFS(bose, bose)
        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), t=[1, 2], u=[1 2; 3 4],
        )
        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), t=[1, 2], u=[2 2; 2 2; 2 2],
        )

        @test_throws ArgumentError HubbardRealSpace(
            comp; geometry=PeriodicBoundaries(3,2), v=[1 1; 1 1; 1 1],
        )

        @test_throws ArgumentError HubbardRealSpace(BoseFS2C((1,2,3), (3,2,1)))

        @test_logs (:warn,) HubbardRealSpace(FermiFS((1,0)), u=[2])
        @test_logs (:warn,) HubbardRealSpace(
            CompositeFS(BoseFS((1,1)), FermiFS((1,0))); u=[2 2; 2 2]
        )

        H = HubbardRealSpace(comp, t=[1,2], u=[1 2; 2 3])
        @test eval(Meta.parse(repr(H))) == H
    end
    @testset "Offdiagonals" begin
        f = near_uniform(FermiFS{3,12})

        H = HubbardRealSpace(f, geometry=PeriodicBoundaries(3, 4))
        od_values = last.(offdiagonals(H, f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 6

        H = HubbardRealSpace(f, geometry=PeriodicBoundaries(4, 3))
        od_values = last.(offdiagonals(H, f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 8

        H = HubbardRealSpace(f, geometry=HardwallBoundaries(3, 4))
        od_values = last.(offdiagonals(H, f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 3

        H = HubbardRealSpace(f, geometry=HardwallBoundaries(4, 3))
        od_values = last.(offdiagonals(H, f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 12
        @test length(od_nonzeros) == 4

        H = HubbardRealSpace(f, geometry=LadderBoundaries(2, 6))
        od_values = last.(offdiagonals(H, f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 9
        @test length(od_nonzeros) == 5

        hard_ladder = LadderBoundaries(2, 6, subgeometry=HardwallBoundaries)
        H = HubbardRealSpace(f, geometry=hard_ladder)
        od_values = last.(offdiagonals(H, f))
        od_nonzeros = filter(!iszero, od_values)
        @test length(od_values) == 9
        @test length(od_nonzeros) == 3
    end
    @testset "1D Bosons (single)" begin
        H1 = HubbardReal1D(BoseFS((1, 1, 1, 1, 1, 0)); u=2, t=3)
        H2 = HubbardRealSpace(BoseFS((1, 1, 1, 1, 1, 0)); u=[2], t=[3])

        @test exact_energy(H1) == exact_energy(H2)
    end
    @testset "1D Bosons (2-component)" begin
        add1 = BoseFS2C(
            (1, 1, 1, 0, 0, 0),
            (1, 0, 0, 0, 0, 0),
        )
        H1 = BoseHubbardReal1D2C(add1, ua=2, v=3, tb=4)

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

        E1 = exact_energy(H1)
        E2 = exact_energy(H2)
        E3 = exact_energy(H3)
        E4 = exact_energy(H4)
        E5 = exact_energy(H5)

        @test E1 ≈ E2 rtol=0.0001
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
        @testset "hardwall and ladder" begin
            geom1 = LadderBoundaries(2, 3, subgeometry=HardwallBoundaries)
            geom2 = HardwallBoundaries(2, 3)
            geom3 = HardwallBoundaries(3, 2)
            bose = BoseFS((1, 1, 1, 0, 0, 0))
            fermi = FermiFS((1, 0, 0, 0, 1, 0))

            H1 = HubbardRealSpace(bose, geometry=geom1)
            H2 = HubbardRealSpace(bose, geometry=geom2)
            H3 = HubbardRealSpace(bose, geometry=geom3)
            @test exact_energy(H1) == exact_energy(H2)
            @test exact_energy(H1) ≈ exact_energy(H3)

            H1 = HubbardRealSpace(fermi, geometry=geom1)
            H2 = HubbardRealSpace(fermi, geometry=geom2)
            H3 = HubbardRealSpace(fermi, geometry=geom3)
            @test exact_energy(H1) == exact_energy(H2)
            @test exact_energy(H1) ≈ exact_energy(H3)
        end
    end
end

@testset "Importance sampling" begin
    @testset "Gutzwiller" begin
        @testset "Gutzwiller transformation" begin
            for H in (
                HubbardMom1D(BoseFS((2,2,2)), u=6),
                ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
                BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0)), ub=2.0),
            )
                # GutzwillerSampling with parameter zero is exactly equal to the original H
                G = GutzwillerSampling(H, 0.0)
                addr = starting_address(H)
                @test starting_address(G) == addr
                @test all(x == y for (x, y) in zip(offdiagonals(H, addr), offdiagonals(G, addr)))
                @test LOStructure(G) isa AdjointKnown
                @test LOStructure(Rimu.Hamiltonians.TransformUndoer(G,G)) isa AdjointKnown

                @test eval(Meta.parse(repr(G))) == G
                @test eval(Meta.parse(repr(G'))) == G'

                g = rand()
                G = GutzwillerSampling(H, g)
                for i in 1:num_offdiagonals(G, addr)
                    addr2, me = get_offdiagonal(G, addr, i)
                    w = exp(-g * (diagonal_element(H, addr2) - diagonal_element(H, addr)))
                    @test get_offdiagonal(H, addr, i)[2] * w == me
                    @test get_offdiagonal(H, addr, i)[1] == addr2
                    @test diagonal_element(H, addr2) == diagonal_element(G, addr2)
                end
            end
        end

        @testset "Gutzwiller observables" begin
            for H in (
                HubbardReal1D(BoseFS((2,2,2)), u=6),
                HubbardMom1D(BoseFS((2,2,2)), u=6),
                ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
                # BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0)), ub=2.0), # multicomponent not implemented for G2RealCorrelator
            )
                # energy
                g = rand()
                x = rand()
                G = GutzwillerSampling(H, g)
                add = starting_address(H)
                dv = DVec(add => x)
                # transforming the Hamiltonian again should be consistent
                fsq = Rimu.Hamiltonians.TransformUndoer(G)
                fHf = Rimu.Hamiltonians.TransformUndoer(G, H)
                Ebare = dot(dv, H, dv)/dot(dv, dv)
                Egutz = dot(dv, G, dv)/dot(dv, dv)
                Etrans = dot(dv, fHf, dv)/dot(dv, fsq, dv)
                @test Ebare ≈ Egutz ≈ Etrans

                # general operators
                m = num_modes(add)
                g2vals = map(d -> dot(dv, G2RealCorrelator(d), dv)/dot(dv, dv), 0:m-1)
                g2transformed = map(d -> dot(dv, Rimu.Hamiltonians.TransformUndoer(G,G2RealCorrelator(d)), dv)/dot(dv, fsq, dv), 0:m-1)
                @test all(g2vals ≈ g2transformed)

                # type promotion
                G2mom = G2MomCorrelator(1)
                @test eltype(Rimu.Hamiltonians.TransformUndoer(G, G2mom)) == eltype(G2mom)
            end
        end
    end

    @testset "GuidingVector" begin
        H = HubbardMom1D(BoseFS((2,2,2)), u=6)
        v = DVec(
            BoseFS{6,3}((0, 0, 6)) => 0.0770580680636451,
            BoseFS{6,3}((6, 0, 0)) => 0.0770580680636451,
            BoseFS{6,3}((1, 1, 4)) => 0.3825802976327182,
            BoseFS{6,3}((4, 1, 1)) => 0.3825802976327182,
            BoseFS{6,3}((0, 6, 0)) => 0.04322440994245527,
            BoseFS{6,3}((3, 3, 0)) => 0.2565124277520772,
            BoseFS{6,3}((3, 0, 3)) => 0.3460652270329457,
            BoseFS{6,3}((0, 3, 3)) => 0.2565124277520772,
            BoseFS{6,3}((1, 4, 1)) => 0.28562685053740633,
            BoseFS{6,3}((2, 2, 2)) => 0.6004825560434165;
            capacity=100,
        )
        @testset "GuidingVector transformation" begin
            @testset "With empty vector" begin
                G = GuidingVectorSampling(H, empty(v), 0.2)

                addr = starting_address(H)
                @test starting_address(G) == addr
                @test all(x == y for (x, y) in zip(offdiagonals(H, addr), offdiagonals(G, addr)))
                @test LOStructure(G) isa AdjointKnown
                @test LOStructure(Rimu.Hamiltonians.TransformUndoer(G,G)) isa AdjointKnown
            end

            @testset "With non-empty vector" begin
                G = GuidingVectorSampling(H, v, 0.2)
                addr = starting_address(H)
                @test starting_address(G) == addr
                @test LOStructure(G) isa AdjointKnown
                @test LOStructure(Rimu.Hamiltonians.TransformUndoer(G,G)) isa AdjointKnown
                @test G == GuidingVectorSampling(H; vector = v, eps = 0.2) # call signature

                for i in 1:num_offdiagonals(G, addr)
                    addr2, me = get_offdiagonal(G, addr, i)
                    top = ifelse(v[addr2] < 0.2, 0.2, v[addr2])
                    bot = ifelse(v[addr] < 0.2, 0.2, v[addr])
                    w = top / bot
                    @test get_offdiagonal(H, addr, i)[2] * w ≈ me
                    @test get_offdiagonal(H, addr, i)[1] == addr2
                    @test diagonal_element(H, addr2) == diagonal_element(G, addr2)
                end
            end
        end

        @testset "Guiding vector observables" begin
            for H in (
                HubbardReal1D(BoseFS((2,2,2)), u=6),
                HubbardMom1D(BoseFS((2,2,2)), u=6),
                ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
                # BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0)), ub=2.0), # multicomponent not implemented for G2RealCorrelator
            )
                # energy
                x = rand()
                G = GuidingVectorSampling(H, v, 0.2)
                add = starting_address(H)
                dv = DVec(add => x)
                # transforming the Hamiltonian again should be consistent
                fsq = Rimu.Hamiltonians.TransformUndoer(G)
                fHf = Rimu.Hamiltonians.TransformUndoer(G, H)
                Ebare = dot(dv, H, dv)/dot(dv, dv)
                Egutz = dot(dv, G, dv)/dot(dv, dv)
                Etrans = dot(dv, fHf, dv)/dot(dv, fsq, dv)
                @test Ebare ≈ Egutz ≈ Etrans

                # general operators
                m = num_modes(add)
                g2vals = map(d -> dot(dv, G2RealCorrelator(d), dv)/dot(dv, dv), 0:m-1)
                g2transformed = map(d -> dot(dv, Rimu.Hamiltonians.TransformUndoer(G,G2RealCorrelator(d)), dv)/dot(dv, fsq, dv), 0:m-1)
                @test all(g2vals ≈ g2transformed)

                # type promotion
                G2mom = G2MomCorrelator(1)
                @test eltype(Rimu.Hamiltonians.TransformUndoer(G, G2mom)) == eltype(G2mom)
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
            ) isa AdjointUnknown
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
            )) isa AdjointUnknown
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
            @test !isa(try Rimu.Hamiltonians.TransformUndoer(G) catch e e end, Exception)
            @test !isa(try Rimu.Hamiltonians.TransformUndoer(G,H) catch e e end, Exception)
        end
        # unsupported
        for H in (
            HubbardMom1D(BoseFS((2,2,2)), u=6),
            ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
            BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0)), ub=2.0),
        )
            @test_throws ArgumentError Rimu.Hamiltonians.TransformUndoer(H)
            @test_throws ArgumentError Rimu.Hamiltonians.TransformUndoer(H, H)
        end
    end
end

@testset "AbstractMatrix and MatrixHamiltonian" begin
    # lomc!() with AbstractMatrix
    ham = HubbardReal1D(BoseFS((1, 1, 1, 1)))
    dim = dimension(ham)
    @test dim ≤ dimension(Int, starting_address(ham)) == dimension(starting_address(ham))
    bsr = BasisSetRep(ham, starting_address(ham))
    sm, basis = sparse(bsr), bsr.basis
    @test dim == length(basis)
    # run lomc! in deterministic mode with Matrix and Vector
    a = lomc!(sm, ones(dim)).df
    b = lomc!(sm, ones(dim)).df
    @test a.shift ≈ b.shift
    # run lomc! in deterministic mode with Hamiltonian and DVec
    v = DVec(k=>1.0 for k in basis; style=IsDeterministic()) # corresponds to `ones(dim)`
    c = lomc!(ham, v).df
    @test a.shift ≈ c.shift

    # MatrixHamiltonian
    @test_throws ArgumentError MatrixHamiltonian([1 2 3; 4 5 6])
    @test_throws ArgumentError MatrixHamiltonian(sm, starting_address = dim+1)
    # adjoint nonhermitian
    nonhermitian = MatrixHamiltonian([1 2; 4 5])
    @test LOStructure(nonhermitian) == AdjointKnown()
    @test get_offdiagonal(nonhermitian,2,1)[2] == get_offdiagonal(nonhermitian',1,1)[2]

    # wrap sparse matrix as MatrixHamiltonian
    mh =  MatrixHamiltonian(sm)
    # adjoint IsHermitian
    @test LOStructure(mh) == IsHermitian()
    @test mh' == mh

    @test starting_address(mh) == 1
    @test dimension(mh) == dim

    # lomc!
    # float walkernumber triggers IsDeterministic algorithm
    d = lomc!(mh, ones(dim)).df
    @test d.shift ≈ a.shift
    # integer walkernumber triggers IsStochasticInteger algorithm
    Random.seed!(15)
    e = lomc!(mh, ones(Int,dim)).df
    @test ≈(e.shift[end], a.shift[end], atol=0.3)
    # wrap full matrix as MatrixHamiltonian
    fmh =  MatrixHamiltonian(Matrix(sm))
    f = lomc!(fmh, ones(dim)).df
    @test f.shift ≈ a.shift
end

@testset "G2MomCorrelator" begin
    # v0 is the exact ground state from BoseHubbardMom1D2C(aIni;ua=0,ub=0,v=0.1)
    bfs1=BoseFS([0,2,0])
    bfs2=BoseFS([0,1,0])
    aIni = BoseFS2C(bfs1,bfs2)
    v0 = DVec(
        BoseFS2C((0, 2, 0), (0, 1, 0)) => 0.9999389545691221,
        BoseFS2C((1, 1, 0), (0, 0, 1)) => -0.007812695959057453,
        BoseFS2C((0, 1, 1), (1, 0, 0)) => -0.007812695959057453,
        BoseFS2C((2, 0, 0), (1, 0, 0)) => 4.046694762039993e-5,
        BoseFS2C((0, 0, 2), (0, 0, 1)) => 4.046694762039993e-5,
        BoseFS2C((1, 0, 1), (0, 1, 0)) => 8.616127793651117e-5,
    )
    g0 = G2MomCorrelator(0)
    g1 = G2MomCorrelator(1)
    g2 = G2MomCorrelator(2)
    g3 = G2MomCorrelator(3)
    @test imag(dot(v0,g0,v0)) == 0 # should be strictly real
    @test abs(imag(dot(v0,g3,v0))) < 1e-10
    @test dot(v0,g0,v0) ≈ 0.65 rtol=0.01
    @test dot(v0,g1,v0) ≈ 0.67 rtol=0.01
    @test dot(v0,g2,v0) ≈ 0.67 rtol=0.01
    @test dot(v0,g3,v0) ≈ 0.65 rtol=0.01
    @test num_offdiagonals(g0,aIni) == 2

    # on first component
    g0f = G2MomCorrelator(0,:first)
    g1f = G2MomCorrelator(1,:first)
    @test imag(dot(v0,g0f,v0)) == 0 # should be strictly real
    @test dot(v0,g0f,v0) ≈ 1.33 rtol=0.01
    @test dot(v0,g1f,v0) ≈ 1.33 + 7.08e-5im rtol=0.01
    # on second component
    g0s = G2MomCorrelator(0,:second)
    g1s = G2MomCorrelator(1,:second)
    #@test_throws ErrorException("invalid ONR") get_offdiagonal(g0s,aIni,1) # should fail due to invalid ONR
    @test dot(v0,g0s,v0) ≈ 1/3
    @test dot(v0,g1s,v0) ≈ 1/3
    # test against BoseFS
    ham1 = HubbardMom1D(bfs1)
    ham2 = HubbardMom1D(bfs2)
    @test num_offdiagonals(g0f,aIni) == num_offdiagonals(ham1,bfs1)
    @test num_offdiagonals(g0s,aIni) == num_offdiagonals(ham2,bfs2)
    aIni = BoseFS2C(bfs2,bfs1) # flip bfs1 and bfs2
    @test get_offdiagonal(g0s,aIni,1) == (BoseFS2C(BoseFS{1,3}((0, 1, 0)),BoseFS{2,3}((1, 0, 1))), 0.47140452079103173)
    # test on BoseFS
    @test diagonal_element(g0s,bfs1) == 4/3
    @test diagonal_element(g0s,bfs2) == 1/3
end

@testset "G2RealCorrelator" begin
    m = 6
    n1 = 4
    n2 = m
    add1 = BoseFS((n1,0,0,0,0,0))
    add2 = near_uniform(BoseFS{n2,m})

    # localised state
    @test diagonal_element(G2RealCorrelator(0), add1) == n1 * (n1 - 1) / m
    @test diagonal_element(G2RealCorrelator(1), add1) == 0.

    # constant density state
    @test diagonal_element(G2RealCorrelator(0), add2) == (n2/m) * ((n2/m) - 1)
    @test diagonal_element(G2RealCorrelator(1), add2) == (n2/m)^2

    # local-local
    comp = CompositeFS(add1,add1)
    @test diagonal_element(G2RealCorrelator(0), comp) == 2n1 * (2n1 - 1) / m
    @test diagonal_element(G2RealCorrelator(1), comp) == 0.

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
end

@testset "SuperfluidCorrelator" begin
    m = 6
    n1 = 4
    n2 = m
    add1 = BoseFS((n1,0,0,0,0,0))
    add2 = near_uniform(BoseFS{n2,m})

    # localised state
    @test diagonal_element(SuperfluidCorrelator(0), add1) == n1/m
    @test diagonal_element(SuperfluidCorrelator(1), add1) == 0.

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

    # non unital localised state
    @test diagonal_element(S0, non_unital_localised_state) ≈ 20/9
    @test diagonal_element(S1, non_unital_localised_state) ≈ (-4/9)*exp(im * -2pi/3)

    # non unital near uniform state
    @test diagonal_element(S0, non_unital_uniform_state) ≈ 2/9

    # constant density localised state
    @test diagonal_element(S0, localised_state) == 5.
    @test diagonal_element(S1, localised_state) ≈ 1
    @test diagonal_element(S2, localised_state) ≈ -1

    # constant density uniform state
    @test diagonal_element(S0, uniform_state) == 0
    @test diagonal_element(S2, uniform_state) == 0
    
end

@testset "Momentum" begin
    @test diagonal_element(Momentum(), BoseFS((0,0,2,1,3))) ≡ 2.0
    @test diagonal_element(Momentum(fold=false), BoseFS((0,0,2,1,3))) ≡ 7.0
    @test diagonal_element(Momentum(1), BoseFS((1,0,0,0))) ≡ -1.0
    @test_throws MethodError diagonal_element(Momentum(2), BoseFS((0,1,0)))

    for add in (BoseFS2C((0,1,2,3,0), (1,2,3,4,5)), FermiFS2C((1,0,0,1), (0,0,1,0)))
        @test diagonal_element(Momentum(1), add) + diagonal_element(Momentum(2), add) ≡
            diagonal_element(Momentum(0), add)
    end

    @test num_offdiagonals(Momentum(), BoseFS((0,1,0))) == 0
    @test LOStructure(Momentum(2; fold=true)) == IsDiagonal()
    @test Momentum(1)' === Momentum(1)
end

@testset "DensityMatrixDiagonal" begin
    @test diagonal_element(DensityMatrixDiagonal(5), FermiFS((0,1,0,1,0,1,0))) == 0
    @test diagonal_element(DensityMatrixDiagonal(2; component=1), BoseFS((1,5,1,0))) == 5

    for add in (
        CompositeFS(BoseFS((1,2,3,4,5)), BoseFS((5,4,3,2,1))),
        BoseFS2C((1,2,3,4,5), (5,4,3,2,1))
    )
        for i in 1:5
            @test diagonal_element(DensityMatrixDiagonal(i, component=1), add) == i
            @test diagonal_element(DensityMatrixDiagonal(i, component=2), add) == 6 - i
            @test diagonal_element(DensityMatrixDiagonal(i), add) == 6
        end
    end

    @test num_offdiagonals(DensityMatrixDiagonal(1), BoseFS((0,1,0))) == 0
    @test LOStructure(DensityMatrixDiagonal(2)) == IsDiagonal()
    @test DensityMatrixDiagonal(15)' === DensityMatrixDiagonal(15)
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
        even_b = BasisSetRep(ParitySymmetry(ham))
        odd_b = BasisSetRep(ParitySymmetry(ham; odd=true))

        for add in even_b.basis
            @test add == min(add, reverse(add))
        end
        for add in odd_b.basis
            @test add == min(add, reverse(add))
            @test add ≠ reverse(add)
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
        even_b = BasisSetRep(ParitySymmetry(ham))

        ham_m = Matrix(ham)
        even_m = Matrix(even_b)

        @test ham_m == even_m
        @test issymmetric(even_m)
    end
end

@testset "TimeReversalSymmetry" begin
    @test_throws ArgumentError TimeReversalSymmetry(HubbardMom1D(BoseFS((1, 1))))
    @test_throws ArgumentError TimeReversalSymmetry(BoseHubbardMom1D2C(BoseFS2C((1, 1),(2,1))))
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
    @testset "2-particle BoseHubbardMom1D2C" begin
        ham = BoseHubbardMom1D2C(BoseFS2C((0,1,1),(1,0,1)))
        even = TimeReversalSymmetry(ham)
        odd = TimeReversalSymmetry(ham; even=false)

        h_eigs = eigvals(Matrix(ham))
        p_eigs = sort!(vcat(eigvals(Matrix(even)), eigvals(Matrix(odd))))

        @test starting_address(even) == time_reverse(starting_address(ham))
        @test h_eigs ≈ p_eigs

        @test issymmetric(Matrix(odd))
        @test issymmetric(Matrix(even))
        @test LOStructure(odd) isa IsHermitian
    end

end

@testset "BasisSetRep" begin
    @testset "basics" begin
        m = 100
        n = 100
        addr = BoseFS(Tuple(i == 1 ? n : 0 for i in 1:m))
        ham = HubbardReal1D(addr)
        @test_throws ArgumentError BasisSetRep(ham) # dimension too large
        m = 2
        n = 10
        addr = near_uniform(BoseFS{n,m})
        ham = HubbardReal1D(addr)
        bsr = BasisSetRep(ham; nnzs = dimension(ham))
        @test length(bsr.basis) == dimension(bsr) ≤  dimension(ham)
        @test_throws ArgumentError BasisSetRep(ham, BoseFS((1,2,3))) # wrong address type
        @test Matrix(bsr) == Matrix(bsr.sm) == Matrix(ham)
        @test sparse(bsr) == bsr.sm == sparse(ham)
        addr2 = bsr.basis[2]
        @test starting_address(BasisSetRep(ham, addr2)) ==  addr2
        @test isreal(ham) == (eltype(ham) <: Real)
        @test isdiag(ham) == (LOStructure(ham) ≡ IsDiagonal())
        @test ishermitian(ham) == (LOStructure(ham) ≡ IsHermitian())
        @test issymmetric(ham) == (ishermitian(ham) && isreal(ham))
    end

    @testset "filtering" begin
        ham = HubbardReal1D(near_uniform(BoseFS{10,2}))
        bsr_orig = BasisSetRep(ham; sort=true)
        mat_orig = Matrix(bsr_orig)
        mat_cut_index = diag(mat_orig) .< 30
        mat_cut_manual = mat_orig[mat_cut_index, mat_cut_index]
        bsr = BasisSetRep(ham; cutoff=30, sort=true)
        mat_cut = Matrix(bsr)
        @test mat_cut == mat_cut_manual
        # pass a basis and generate truncated BasisSetRep
        bsrt = BasisSetRep(ham, bsr.basis; filter= Returns(false), sort=true)
        @test bsrt.basis == bsr.basis
        @test bsr.sm == bsrt.sm
        # pass addresses and generate reachable basis
        @test BasisSetRep(ham, bsr.basis, sort=true).basis == bsr_orig.basis

        filterfun(fs) = maximum(onr(fs)) < 8
        mat_cut_index = filterfun.(BasisSetRep(ham; sort=true).basis)
        mat_cut_manual = mat_orig[mat_cut_index, mat_cut_index]
        mat_cut = Matrix(ham; filter=filterfun, sort=true)
        @test mat_cut == mat_cut_manual
    end

    @testset "getindex" begin
        ham = HubbardReal1D(near_uniform(BoseFS{10,2}))
        bsr = BasisSetRep(ham; sort=true)
        b = bsr.basis
        @test [ham[i, j] for i in b, j in b] == Matrix(bsr)
    end

    @testset "momentum blocking" begin
        add1 = BoseFS((2,0,0,0))
        add2 = BoseFS((0,1,0,1))
        ham = HubbardMom1D(add1)

        @test Matrix(ham, add1; sort=true) == Matrix(ham, add2; sort=true)
        @test Matrix(ham, add1) ≠ Matrix(ham, add2)

        add1 = BoseFS((2,0,0,0,0))
        add2 = BoseFS((0,1,0,0,1))
        ham = HubbardMom1D(add1)

        @test Matrix(ham, add1; sort=true) == Matrix(ham, add2; sort=true)
        @test Matrix(ham, add1) ≠ Matrix(ham, add2)
    end

    using Rimu.Hamiltonians: fix_approx_hermitian!, isapprox_enforce_hermitian!
    using Rimu.Hamiltonians: build_sparse_matrix_from_LO
    using Random
    @testset "fix_approx_hermitian!" begin
        # generic `Matrix`
        Random.seed!(17)
        mat = rand(5,5)
        @test !ishermitian(mat)
        @test_throws ArgumentError fix_approx_hermitian!(mat; test_approx_symmetry=true)
        @test !ishermitian(mat) # still not hermitian
        fix_approx_hermitian!(mat; test_approx_symmetry=false)
        @test ishermitian(mat) # now it is hermitian

        # sparse matrix
        Random.seed!(17)
        mat = sparse(rand(5,5))
        @test !ishermitian(mat)
        @test_throws ArgumentError fix_approx_hermitian!(mat; test_approx_symmetry=true)
        @test !ishermitian(mat) # still not hermitian

        # subtle symmetry violation due to `ParitySymmetry` wrapper
        ham = HubbardMom1D(BoseFS((1, 0, 1, 2, 0)))
        even = ParitySymmetry(ham; odd=false)
        odd = ParitySymmetry(ham; even=false)

        even_sm, _ = build_sparse_matrix_from_LO(even)
        even_m = Matrix(even) # symmetrised version via BasisSetRep

        @test !issymmetric(even_sm) # not symmetric due to floating point errors
        @test issymmetric(even_m) # because it was passed through `fix_approx_hermitian!`
        @test even_sm ≈ even_m # still approximately the same!
    end

    @testset "basis-only" begin
        m = 5
        n = 5
        add = near_uniform(BoseFS{n,m})
        ham = HubbardReal1D(add)
        @test_throws ArgumentError build_basis(ham, BoseFS((1,2,3))) # wrong address type
        # same basis as BSR
        bsr = BasisSetRep(ham)
        basis = build_basis(ham)
        @test basis == bsr.basis
        @test basis == build_basis(ham, basis) # passing multiple addresses
        # sorting
        basis = build_basis(ham, add; sort = true)
        @test basis == sort!(bsr.basis)
        # filtering
        @test_throws ArgumentError build_basis(ham, add; sizelim = 100)
        @test length(build_basis(ham, add; cutoff = -1)) == 1 # no new addresses added
        cutoff = n * (n-1) / 4  # half maximum energy
        bsr = BasisSetRep(ham, add; cutoff)
        basis = build_basis(ham, add; cutoff)
        @test basis == bsr.basis
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

        bsr = BasisSetRep(H; sizelim=Inf)
        @test dimension(bsr) == 15  # dimension(bsr) < dimension(H)
        @test sum(bsr.sm) ≈ 142.6393438659114

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
        
        Lz_vals = eigvals(Matrix(BasisSetRep(Lz)))
        Ly_vals = eigvals(Matrix(BasisSetRep(Ly)))
        Lx_vals = eigvals(Matrix(BasisSetRep(Lx)))

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

        null_addr = BoseFS(prod(S),)
        @test isempty(fock_to_cart(null_addr, S))
    end
end
