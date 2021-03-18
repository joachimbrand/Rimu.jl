using LinearAlgebra
using Rimu
using Test

using Rimu.Hamiltonians: LOStructure, Hermitian, AdjointKnown, AdjointUnknown

"""
    test_hamiltonian_interface(H, addr)

The main purpose of this test function is to check that all required methods are defined.
"""
function test_hamiltonian_interface(H)
    addr = starting_address(H)
    @testset "$(nameof(typeof(H)))" begin
        @testset "*, mul!, and call" begin
            v = DVec2(addr => eltype(H)(2.0), capacity=100)
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
                @test diagonal_element(H, addr) ≥ 0
            else
                @test norm(diagonal_element(H, addr)) ≥ 0
            end
        end
        @testset "hopping" begin
            h = offdiagonals(H, addr)
            @test eltype(h) == Tuple{typeof(addr), eltype(H)}
            @test length(h) == num_offdiagonals(H, addr)
            for i in 1:length(h)
                @test h[i] == get_offdiagonal(H, addr, i)
                @test h[i] isa eltype(h)
            end
        end
        @testset "LOStructure" begin
            @test LOStructure(H) isa LOStructure
            if LOStructure(H) isa Hermitian
                @test H' == H
            elseif LOStructure(H) isa AdjointKnown
                @test begin H'; true; end # make sure no error is thrown
            else
                @test_throws ErrorException H'
            end
        end
        @testset "dimension" begin
            @test dimension(H) isa Int
            @test dimension(Float64, H) isa Float64
            @test dimension(Int, H) === dimension(H)
        end
        @testset "starting address" begin
            @test num_modes(H) == num_modes(addr)
        end
    end
end

@testset "Interface tests" begin
    for H in (
        HubbardReal1D(BoseFS((1, 2, 3, 4)); u=1.0, t=2.0),
        HubbardMom1D(BoseFS((6, 0, 0, 4)); t=1.0, u=0.5),
        HubbardMom1D(BoseFS((6, 0, 0, 4)); t=1.0, u=0.5 + im),
        ExtendedHubbardReal1D(BoseFS((1,0,0,0,1)); u=1.0, v=2.0, t=3.0),

        BoseHubbardReal1D2C(BoseFS2C((1,2,3), (1,0,0))),
        BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0))),

        BoseHubbardReal1D(n=5, m=5, AT=BoseFS{5,5}),
        BoseHubbardMom1D(n=10, m=5, add=BoseFS((2, 2, 3, 3, 0))),

        ExtendedBHReal1D(),

        GutzwillerSampling(HubbardReal1D(BoseFS((1,2,3)); u=6); g=0.3),
        GutzwillerSampling(BoseHubbardMom1D2C(BoseFS2C((3,2,1), (1,2,3)); ua=6); g=0.3),
        GutzwillerSampling(HubbardReal1D(BoseFS((1,2,3)); u=6 + 2im); g=0.3),
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
end

@testset "1C model properties" begin
    addr = nearUniform(BoseFS{100,100})

    for Hamiltonian in (HubbardReal1D, HubbardMom1D)
        @testset "$Hamiltonian" begin
            H = Hamiltonian(addr; t=1.0, u=2.0)
            @test H.t == 1.0
            @test H.u == 2.0
            @test LOStructure(H) == Hermitian()
            @test starting_address(H) == addr
            @test eval(Meta.parse(repr(H))) == H
        end
    end
end

@testset "2C model properties" begin
    flip(b) = BoseFS2C(b.bsb, b.bsa)
    addr1 = nearUniform(BoseFS2C{1,100,20})
    addr2 = nearUniform(BoseFS2C{100,1,20})

    for Hamiltonian in (BoseHubbardReal1D2C, BoseHubbardMom1D2C)
        @testset "$Hamiltonian" begin
            H1 = BoseHubbardReal1D2C(addr1; ta=1.0, tb=2.0, ua=0.5, ub=0.7, v=0.2)
            H2 = BoseHubbardReal1D2C(addr2; ta=2.0, tb=1.0, ua=0.7, ub=0.5, v=0.2)
            @test starting_address(H1) == addr1
            @test LOStructure(H1) == Hermitian()

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

@testset "Gutzwiller sampling" begin
    for H in (
        HubbardMom1D(BoseFS((2,2,2)), u=6),
        ExtendedHubbardReal1D(BoseFS((1,1,1,1,1,1,1,1,1,1,1,1)), u=6, t=2.0),
        BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0)), ub=2.0),
    )
        # GutzwillerSampling with parameter zero is exactly equal to the original H
        G = GutzwillerSampling(H, 0.0)
        addr1 = starting_address(H)
        @test starting_address(G) == addr1
        @test all(x == y for (x, y) in zip(hops(H, addr1), hops(G, addr1)))
        @test LOStructure(G) isa AdjointKnown

        @test eval(Meta.parse(repr(G))) == G
        @test eval(Meta.parse(repr(G'))) == G'

        g = rand()
        G = GutzwillerSampling(H, g)
        for i in 1:num_offdiagonals(G, addr1)
            addr2, me = get_offdiagonal(G, addr1, i)
            w = exp(-g * (diagonal_element(H, addr2) - diagonal_element(H, addr1)))
            @test get_offdiagonal(H, addr1, i)[2] * w == me
            @test get_offdiagonal(H, addr1, i)[1] == addr2
            @test diagonal_element(H, addr2) == diagonal_element(G, addr2)
        end
    end

    @testset "adjoint" begin
        ###
        ### Define Hamiltonian from a matrix.
        ###
        struct IntAddress <: AbstractFockAddress
            v::Int
        end

        struct MatrixHam{T} <: AbstractHamiltonian{T}
            arr::Matrix{T}
        end

        Rimu.diagonal_element(m::MatrixHam, i) = m.arr[i.v, i.v]
        Rimu.num_offdiagonals(m::MatrixHam, i) = size(m.arr, 1) - 1
        function Rimu.get_offdiagonal(m::MatrixHam, i, j)
            if j ≥ i.v # skip diagonal
                j += 1
            end
            return IntAddress(j), m.arr[i.v, j]
        end

        Rimu.starting_address(::MatrixHam) = IntAddress(1)

        LinearAlgebra.adjoint(m::MatrixHam) = MatrixHam(collect(m.arr'))
        Hamiltonians.LOStructure(::Type{<:MatrixHam}) = AdjointKnown()

        dm(h) = Hamiltonians.build_sparse_matrix_from_LO(h)[1] |> Matrix

        M = MatrixHam(rand(Complex{Float64}, (20, 20)))
        @test dm(M) == M.arr
        @test dm(M') == M.arr'
        @test dm(GutzwillerSampling(M, 0.2)') == dm(GutzwillerSampling(M, 0.2))'
        @test LOStructure(GutzwillerSampling(M, 0.2)) isa AdjointKnown
        @test LOStructure(GutzwillerSampling(HubbardReal1D(BoseFS((1,2)),t=0+2im), 0.2)) isa
            AdjointUnknown
    end
end

@testset "old tests" begin
    ham = BoseHubbardReal1D(
        n = 9,
        m = 9,
        u = 6.0,
        t = 1.0,
        AT = BoseFS{9,9})
    @test dimension(ham) == 24310

    aIni = Rimu.Hamiltonians.nearUniform(ham)
    @test aIni == BoseFS{9,9}((1,1,1,1,1,1,1,1,1))

    hp = offdiagonals(ham,aIni)
    @test length(hp) == 18
    @test hp[18][1] == BoseFS{9,9}(BitString{17}(0x000000000000d555))
    @test hp[18][2] ≈ -√2
    @test diagonal_element(ham,aIni) == 0
    os = BoseFS([12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4])
    @test Rimu.Hamiltonians.bose_hubbard_interaction(os) == 148
    @test Rimu.Hamiltonians.extended_bose_hubbard_interaction(os) == (53, 148)
    @test Rimu.Hamiltonians.numberoccupiedsites(os) == 9
    hnnn = Rimu.Hamiltonians.hopnextneighbour(BoseFS{25,16}(BitString{40}(0xf342564fff)),3)
    bs = BoseFS(BitString{40}(0xf342564fff))
    hnnbs = Rimu.Hamiltonians.hopnextneighbour(bs,3)
    @test hnnn == hnnbs

    svec = DVec2(Dict(aIni => 2.0), dimension(ham))
    v2 = ham(svec)
    v3 = ham*v2
    @test norm(v3,1) ≈ 1482.386824949077
    @test v2 == mul!(similar(svec), ham, svec)
    @test norm(v2) ≈ 12
    @test v2 == ham*svec
    @test dot(v2,ham,svec) == v2⋅(ham*svec) ≈ 144
    @test -⋅(UniformProjector(),ham,svec)≈⋅(NormProjector(),ham,svec)≈norm(v2,1)
    @test dot(Norm2Projector(),v2) ≈ norm(v2,2)
    @test Hamiltonians.LOStructure(ham) == Hamiltonians.Hermitian()
    aIni2 = nearUniform(BoseFS{9,9})
    hamc = BoseHubbardReal1D(aIni2, u=6.0+0im, t=1.0+0im) # formally a complex operator
    @test Hamiltonians.LOStructure(hamc) == Hamiltonians.AdjointUnknown()
    @test dot(v3,ham,svec) ≈ dot(v3,hamc,svec) ≈ dot(svec,ham,v3) ≈ dot(svec,hamc,v3) ≈ 864
    hamcc = BoseHubbardReal1D(aIni2, u=6.0+0.1im, t=1.0+2im) # a complex operator
    vc2 = hamcc*svec
    @test isreal(dot(vc2,hamcc,svec))
    @test dot(vc2,hamc,svec) ≉ dot(svec,hamc,vc2)

    @test adjoint(ham) == ham' == ham
    @test Rimu.Hamiltonians.LOStructure(hamcc) == Rimu.Hamiltonians.AdjointUnknown()
    @test_throws ErrorException hamcc'
end
