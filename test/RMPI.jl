using LinearAlgebra
using Rimu
using Rimu.RMPI
using Rimu.RMPI: sort_and_count!
using SplittablesBase: halve, amount
using Test

@testset "RNG independence" begin
    m = n = 6
    add = nearUniform(BoseFS{n,m})
    svec = DVec(add => 2)
    dv = MPIData(svec)
    @test ConsistentRNG.check_crng_independence(dv) ==
        mpi_size()*Threads.nthreads()*fieldcount(ConsistentRNG.CRNG)
end

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
                ordfun(x) = hash(x) % k
                vals = rand(Int, l)
                counts = zeros(Int, k)
                displs = zeros(Int, k)

                sort_and_count!(counts, displs, vals, ordfun.(vals), (0, k-1))
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

    @testset "Iteration" begin
        @test sort(collect(values(dv1))) == 1:4
        @test sort(collect(keys(dv1))) == 1:4
        @test sum(first, pairs(dv1)) == 10
        @test sum(last, pairs(dv1)) == 10

        a, b = halve(pairs(dv1))
        @test amount(a) + amount(b) == amount(pairs(dv1))
        @test prod(first, a) * prod(first, b) == 24

        @test sum(values(dv2)) == 0
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
        rand_ham = MatrixHamiltonian(rand(4,4))
        ldv1 = localpart(dv1)
        @test norm(dot(dv1, rand_ham, dv1)) ≈ norm(dot(ldv1, rand_ham, ldv1))
    end
end
