using LinearAlgebra
using MPI
using Random
using Rimu
using StaticArrays
using Statistics
using Test

using Rimu.RMPI
using Rimu.StatsTools
using Rimu.ConsistentRNG
using Rimu.RMPI: targetrank, mpi_synchronize!

const N_REPEATS = 5

"""
    rand_onr(N, M)

Generate random ONR with N particles in M sites.
"""
function rand_onr(N, M)
    result = zeros(MVector{M,Int})
    for _ in 1:N
        result[rand(1:M)] += 1
    end
    return result.data
end

"""
    correct_ranks(md)

Check if all entries in `md` are located on the correct rank.
"""
function correct_ranks(md)
    return mapreduce(|, pairs(md); init=true) do ((k, v))
        targetrank(k, mpi_size()) == mpi_rank()
    end
end

# Ignore all printing on ranks other than root.
if mpi_rank() != mpi_root
    redirect_stderr(devnull)
    redirect_stdout(devnull)
end

"""
    setup_dv(type, args...; kwargs...)

Create a local and distributed versions of dvec of type `type`. Ensure they both have the same
contents.
"""
function setup_dv(type, args...; md_kwargs=(;), kwargs...)
    v = type(args...; kwargs...)
    if mpi_rank() == mpi_root
        dv = MPIData(copy(v); md_kwargs...)
    else
        dv = MPIData(empty(v); md_kwargs...)
    end
    return v, dv
end

@testset "MPI tests" begin
    @testset "MPIData" begin
        for type in (InitiatorDVec, DVec)
            @testset "Single component $type" begin
                for i in 1:N_REPEATS
                    add = BoseFS((0,0,10,0,0))
                    H = HubbardMom1D(add)
                    Random.seed!(7350 * i)
                    v, dv = setup_dv(
                        type, [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:100]
                    )
                    mpi_synchronize!(dv)

                    Random.seed!(1337 * i)
                    w, dw = setup_dv(
                        type, [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:20]
                    )
                    mpi_synchronize!(dw)

                    @test correct_ranks(dv)
                    @test length(v) == length(dv)
                    @test correct_ranks(dw)
                    @test length(w) == length(dw)

                    @testset "Basics" begin
                        @test norm(v) ≈ norm(dv)
                        @test norm(v, 1) ≈ norm(dv, 1)
                        @test norm(v, 2) ≈ norm(dv, 2)
                        @test norm(v, Inf) ≈ norm(dv, Inf)
                        @test sum(values(v)) ≈ sum(values(dv))
                        f((k, v)) = (k == add) + v > 0
                        @test mapreduce(f, |, pairs(v); init=true) ==
                            mapreduce(f, |, pairs(dv); init=true)
                    end

                    @testset "Operations" begin
                        @test dot(v, w) ≈ dot(dv, dw)

                        @test dot(v, H, w) ≈ dot(v, H, dw)
                        @test dot(w, H, v) ≈ dot(w, H, dv)

                        @test dot(freeze(dw), dv) ≈ dot(w, v)
                        @test dot(freeze(dv), dw) ≈ dot(v, w)

                        du = MPIData(H * dv)
                        u = H * v
                        @test correct_ranks(du)

                        @test length(u) == length(du)
                        @test norm(u, 1) ≈ norm(du, 1)
                        @test norm(u, 2) ≈ norm(du, 2)
                        @test norm(u, Inf) ≈ norm(du, Inf)
                    end
                end
            end
            @testset "Two-component $type" begin
                for i in 1:N_REPEATS
                    add = BoseFS2C((0,0,10,0,0), (0,0,2,0,0))
                    H = BoseHubbardMom1D2C(add)
                    Random.seed!(7350 * i)
                    v, dv = setup_dv(
                        type, [BoseFS2C(rand_onr(10, 5), rand_onr(2, 5)) => rand() for _ in 1:100]
                    )
                    mpi_synchronize!(dv)

                    Random.seed!(1337 * i)
                    w, dw = setup_dv(
                        type, [BoseFS2C(rand_onr(10, 5), rand_onr(2, 5)) => rand() for _ in 1:20]
                    )
                    mpi_synchronize!(dw)

                    @testset "Operations" begin
                        @test dot(v, w) ≈ dot(dv, dw)

                        @test dot(v, H, w) ≈ dot(v, H, dw)
                        @test dot(w, H, v) ≈ dot(w, H, dv)
                        G1 = G2Correlator(1)
                        G3 = G2Correlator(3)
                        @test dot(v, G1, w) ≈ dot(v, G1, dw)
                        @test dot(w, G3, v) ≈ dot(w, G3, dv)

                        @test dot(freeze(dw), dv) ≈ dot(w, v)
                        @test dot(freeze(dv), dw) ≈ dot(v, w)

                        du = MPIData(H * dv)
                        u = H * v
                        @test correct_ranks(du)

                        @test length(u) == length(du)
                        @test norm(u, 1) ≈ norm(du, 1)
                        @test norm(u, 2) ≈ norm(du, 2)
                        @test norm(u, Inf) ≈ norm(du, Inf)

                        du = MPIData(G3 * dv)
                        u = G3 * v
                        @test correct_ranks(du)

                        @test length(u) == length(du)
                        @test norm(u, 1) ≈ norm(du, 1)
                        @test norm(u, 2) ≈ norm(du, 2)
                        @test norm(u, Inf) ≈ norm(du, Inf)
                    end
                end
            end
        end
        @testset "Communication strategies" begin
            # The idea here is to generate a big DVec and make sure all communication
            # strategies distribute it in the same manner.
            for i in 1:N_REPEATS
                sorted = map((
                    (; setup=RMPI.mpi_point_to_point),
                    (; setup=RMPI.mpi_one_sided, capacity=10_000),
                    (; setup=RMPI.mpi_all_to_all),
                )) do kw
                    Random.seed!(2021 * hash(mpi_rank()) * i)
                    source = DVec([BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:10_000])
                    target = MPIData(similar(source); kw...)
                    RMPI.mpi_combine_walkers!(target, source)
                    return target
                end

                for v in sorted
                    @test correct_ranks(v)
                end
                @test localpart(sorted[1]) == localpart(sorted[2])
                @test localpart(sorted[2]) == localpart(sorted[3])
            end
        end
    end

    @testset "Ground state energy estimates" begin
        # H1 = HubbardReal1D(BoseFS((1,1,1,1,1,1,1)); u=6.0)
        # E0 = eigsolve(H1, DVec(starting_address(H1) => 1.0), 1, :SR; issymmetric=true)[1][1]
        E0 = -4.6285244934941305
        mpi_seed_CRNGs!(1000_000_000)

        for (setup, kwargs) in (
            (RMPI.mpi_point_to_point, (;)),
            (RMPI.mpi_all_to_all, (;)),
            (RMPI.mpi_one_sided, (; capacity=1000)),
        )
            @testset "Regular with $setup and post-steps" begin
                H = HubbardReal1D(BoseFS((1,1,1,1,1,1,1)); u=6.0)
                dv = MPIData(
                    DVec(starting_address(H) => 3; style=IsDynamicSemistochastic());
                    setup,
                    kwargs...
                )

                post_step = (
                    ProjectedEnergy(H, dv),
                    SignCoherence(copy(localpart(dv))),
                    WalkerLoneliness(),
                    Projector(proj_1=Norm2Projector()),
                )
                df = lomc!(H, dv; post_step, laststep=5000).df

                # Shift estimate.
                Es, σs = mean_and_se(df.shift[2000:end])
                s_low, s_high = Es - 2σs, Es + 2σs
                # Projected estimate.
                r = ratio_of_means(df.hproj[2000:end], df.vproj[2000:end])
                p_low, p_high = quantile(r, [0.0015, 0.9985])

                @test s_low < E0 < s_high
                @test p_low < E0 < p_high
                @test all(-1 .≤ df.coherence .≤ 1)
                @test all(0 .≤ df.loneliness .≤ 1)
            end
            @testset "Initiator with $setup" begin
                H = HubbardMom1D(BoseFS((0,0,0,7,0,0,0)); u=6.0)
                dv = MPIData(
                    InitiatorDVec(starting_address(H) => 3);
                    setup,
                    kwargs...
                )
                df = lomc!(H, dv; laststep=5000).df

                # Shift estimate.
                Es, _ = mean_and_se(df.shift[2000:end])
                @test E0 ≤ Es
            end
        end
    end

    @testset "Same seed gives same results" begin
        # Note: all_to_all will give different results here, as it sorts slightly differently.
        # The entries in the vectors are the same (tested above), but they appear in a
        # different order.
        add = BoseFS2C((1, 1, 1, 1, 1), (1, 0, 0, 0, 0))
        H = BoseHubbardReal1D2C(add; v=2)
        dv = DVec(add => 1)

        dv_ptp = MPIData(copy(dv); setup=RMPI.mpi_point_to_point)
        mpi_seed_CRNGs!(17)
        df_ptp = lomc!(H, dv_ptp).df

        dv_os = MPIData(copy(dv); setup=RMPI.mpi_one_sided, capacity=1000)
        mpi_seed_CRNGs!(17)
        df_os = lomc!(H, dv_os).df

        @test df_ptp == df_os
    end

    # Make sure all ranks came this far.
    @testset "Finish" begin
        @test MPI.Allreduce(true, &, mpi_comm())
    end
end

if mpi_rank() ≠ mpi_root
    redirect_stderr(stderr)
    redirect_stdout(stdout)
end
