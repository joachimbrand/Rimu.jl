using LinearAlgebra
using Rimu
using Rimu.RMPI
using Rimu.RMPI: sort_and_count!
using Test

@testset "DistributeStrategies" begin
    # `DistributeStrategy`s
    ham = HubbardReal1D(BoseFS((1,2,3)))
    for setup in [RMPI.mpi_no_exchange, RMPI.mpi_all_to_all, RMPI.mpi_point_to_point]
        dv = DVec(starting_address(ham)=>10; style=IsDynamicSemistochastic())
        v = MPIData(dv; setup)
        df, state = lomc!(ham,v)
        @test size(df) == (100, 11)
    end
    # need to do mpi_one_sided separately
    dv = DVec(starting_address(ham)=>10; style=IsDynamicSemistochastic())
    v = RMPI.mpi_one_sided(dv; capacity = 1000)
    df, state = lomc!(ham,v)
    @test size(df) == (100, 11)
end

@testset "sort_and_count!" begin
    for l in (1, 2, 30, 1000)
        for k in (2, 10, 100)
            @testset "k=$k, l=$l" begin
                ordfun(x) = hash(x, hash(1)) % k
                vals = rand(1:10, l)
                counts = zeros(Int, k)
                displs = zeros(Int, k)

                RMPI.sort_and_count!(counts, displs, vals, ordfun.(vals), (0, k-1))
                @test issorted(vals, by=ordfun)
                @test sum(counts) == l

                for i in 0:(k - 1)
                    c = counts[i + 1]
                    d = displs[i + 1]
                    r = (1:c) .+ d
                    ords = ordfun.(vals)
                    @test all(ords[r] .== i)
                end
            end
        end
    end
end

@testset "MPIData" begin
    dv1 = MPIData(DVec(4 => 1, 3 => 2, 2 => 3, 1 => 4))
    dv2 = MPIData(empty(localpart(dv1)))

    @testset "Iteration and reductions" begin
        @test sort(collect(localpart(values(dv1)))) == 1:4

        @test sum(first, pairs(dv1)) == 10
        @test sum(last, pairs(dv1)) == 10
        @test prod(keys(dv1)) == 24
        @test sum(values(dv2)) == 0
    end
    @testset "Errors" begin
        @test_throws ErrorException [p for p in pairs(dv1)]
        @test_throws ErrorException [k for k in keys(dv1)]
        @test_throws ErrorException [v for v in values(dv1)]
    end
    @testset "Norms" begin
        @test walkernumber(dv1) ≡ 10.0
        @test norm(dv1, 1) ≡ 10.0
        @test norm(dv1, 2) ≈ 5.477225575
        @test norm(dv1, Inf) ≡ 4.0

        @test walkernumber(dv2) ≡ 0.0
        @test norm(dv2, 1) ≡ 0.0
        @test norm(dv2, 2) ≡ 0.0
        @test norm(dv2, Inf) ≡ 0.0

        @test_throws ErrorException norm(dv1, 3)
        @test_throws ErrorException norm(dv2, 15)
    end
    @testset "dot" begin
        @test dot(dv1, dv2) == 0
        @test dot(dv1, dv1) == dot(localpart(dv1), dv1)
        rand_ham = MatrixHamiltonian(rand(ComplexF64, 4,4))
        ldv1 = localpart(dv1)
        @test norm(dot(dv1, rand_ham, dv1)) ≈ norm(dot(ldv1, rand_ham, ldv1))
    end
end
