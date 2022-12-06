using LinearAlgebra
using Random
using Rimu
using Test
using KrylovKit
using StaticArrays
using MPI

using Rimu.RMPI
using Rimu.StatsTools
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

@mpi_root @info "Running MPI tests..."
RMPI.mpi_allprintln("hello")

# Ignore all printing on ranks other than root. Passing an argument to this script disables
# this.
if isnothing(get(ARGS, 1, nothing)) && mpi_rank() != mpi_root
    redirect_stderr(devnull)
    redirect_stdout(devnull)
end
if !isnothing(get(ARGS, 1, nothing))
    @mpi_root @info "Debug printing enabled"
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
            @testset "copy_to_local" begin
                dv = MPIData(type(mpi_rank() => 1, -1 => 10))
                loc = RMPI.copy_to_local(dv)
                @test length(loc) == mpi_size() + 1
                @test loc[-1] == 10 * mpi_size()
            end
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

                        @test dot(freeze(dw), dv) ≈ dot(w, v)
                        @test dot(freeze(dv), dw) ≈ dot(v, w)

                        for op in (H, DensityMatrixDiagonal(1))
                            @test dot(v, op, w) ≈ dot(v, op, dw)
                            @test dot(w, op, v) ≈ dot(w, op, dv)
                            @test dot(w, op, v) ≈ dot(dw, op, dv)
                            @test dot(w, op, v) ≈ dot(dw, op, v)

                            du = MPIData(op * dv)
                            u = op * v
                            @test correct_ranks(du)

                            @test length(u) == length(du)
                            @test norm(u, 1) ≈ norm(du, 1)
                            @test norm(u, 2) ≈ norm(du, 2)
                            @test norm(u, Inf) ≈ norm(du, Inf)
                        end
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
                        G1 = G2MomCorrelator(1)
                        G3 = G2MomCorrelator(3)
                        @test dot(v, G1, w) ≈ dot(v, G1, dw)
                        @test dot(w, G3, v) ≈ dot(w, G3, dv)
                        @test dot(w, G3, v) ≈ dot(dw, G3, v)
                        @test dot(w, G3, v) ≈ dot(dw, G3, dv)

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
                    mpi_seed!(i)
                    source = DVec(
                        [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:10_000]
                    )
                    target = MPIData(similar(source); kw...)
                    RMPI.mpi_combine_walkers!(target, source)
                    return target
                end

                for v in sorted
                    @test correct_ranks(v)
                end

                ptp_pairs = sort(collect(pairs(localpart(sorted[1]))))
                os_pairs = sort(collect(pairs(localpart(sorted[3]))))
                ata_pairs = sort(collect(pairs(localpart(sorted[2]))))

                @test first.(ptp_pairs) == first.(os_pairs)
                @test first.(ata_pairs) == first.(os_pairs)

                @test last.(ptp_pairs) ≈ last.(os_pairs)
                @test last.(ata_pairs) ≈ last.(os_pairs)
            end
        end
    end

    @testset "PDVec" begin
        add = FermiFS2C((1,1,1,0,0,0),(1,1,1,0,0,0))
        ham = HubbardRealSpace(add)

        dv = DVec(add => 1.0)
        pv = PDVec(add => 1.0)

        res_dv = eigsolve(ham, dv, 1, :SR; issymmetric=true)
        res_pv = eigsolve(ham, pv, 1, :SR; issymmetric=true)

        prop = Rimu.DictVectors.OperatorMulPropagator(ham, pv)
        res_prop = eigsolve(prop, pv, 1, :SR; issymmetric=true)

        @test res_dv[1][1] ≈ res_pv[1][1]
        @test res_dv[1][1] ≈ res_prop[1][1]

        dv = res_dv[2][1]
        pv = copy(res_pv[2][1])
        @test norm(pv) ≈ 1
        @test length(pv) == length(dv)
        @test sum(values(pv)) ≈ sum(values(dv))
        normalize!(pv, 1)
        @test norm(pv, 1) ≈ 1
        rmul!(pv, 2)
        @test norm(pv, 1) ≈ 2

        pv1 = copy(res_pv[2][1])
        pv2 = mul!(similar(pv), ham, pv)

        @test dot(pv2, pv1) ≈ dot(pv2, dv)
        @test dot(pv1, pv2) ≈ dot(dv, pv2)
        @test dot(freeze(pv2), pv1) ≈ dot(pv2, pv1)
        @test dot(pv1, freeze(pv2)) ≈ dot(pv1, pv2)

        if mpi_size() > 1
            @test pv.communicator isa DictVectors.PointToPoint
            @test mpi_size() == mpi_size(pv.communicator)
            @test_throws DictVectors.CommunicatorError iterate(pairs(pv))

            local_pairs = collect(pairs(localpart(pv)))
            local_vals = sum(abs2, values(localpart(pv)))

            total_len = MPI.Allreduce(length(local_pairs), +, MPI.COMM_WORLD)
            total_vals = MPI.Allreduce(local_vals, +, MPI.COMM_WORLD)

            @test total_len == length(dv)
            @test total_vals == sum(abs2, values(pv))
        end

        @testset "dot" begin

            add = BoseFS((0,0,10,0,0))
            H = HubbardMom1D(add)
            pairs_v = [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:100]
            pairs_w = [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:20]
            v = DVec(pairs_v)
            w = DVec(pairs_w)
            pv = PDVec(pairs_v)
            pw = PDVec(pairs_w)

            @test dot(v, w) ≈ dot(pv, pw)

            @test dot(freeze(pw), pv) ≈ dot(w, v) ≈ dot(pw, freeze(pv))
            @test dot(freeze(pv), pw) ≈ dot(v, w) ≈ dot(pv, freeze(pw))

            for op in (H, DensityMatrixDiagonal(1))
                @test dot(v, op, w) ≈ dot(v, op, pw)
                @test dot(w, op, v) ≈ dot(w, op, pv)
                @test dot(w, op, v) ≈ dot(pw, op, pv)
                @test dot(w, op, v) ≈ dot(pw, op, v)

                pu = op * pv
                u = op * v
                @test length(u) == length(pu)
                @test norm(u, 1) ≈ norm(pu, 1)
                @test norm(u, 2) ≈ norm(pu, 2)
                @test norm(u, Inf) ≈ norm(pu, Inf)
            end
        end
    end

    @testset "Ground state energy estimates" begin
        # H1 = HubbardReal1D(BoseFS((1,1,1,1,1,1,1)); u=6.0)
        # E0 = eigvals(Matrix(H1))[1]
        E0 = -4.628524493494574
        mpi_seed!(1000_000_000)

        for (setup, kwargs) in (
            (RMPI.mpi_point_to_point, (;)),
            (RMPI.mpi_all_to_all, (;)),
            (RMPI.mpi_one_sided, (; capacity=1000)),
            (:PDVec, (;)),
        )
            @testset "Regular with $setup and post-steps" begin
                H = HubbardReal1D(BoseFS((1,1,1,1,1,1,1)); u=6.0)
                if setup == :PDVec
                    dv = PDVec(starting_address(H) => 3; style=IsDynamicSemistochastic())
                else
                    dv = MPIData(
                        DVec(starting_address(H) => 3; style=IsDynamicSemistochastic());
                        setup,
                        kwargs...
                    )
                end

                post_step = (
                    ProjectedEnergy(H, dv),
                    SignCoherence(copy(localpart(dv))),
                    WalkerLoneliness(),
                    Projector(proj_1=Norm2Projector()),
                )
                df = lomc!(H, dv; post_step, laststep=5000).df

                # Shift estimate.
                Es, σs = mean_and_se(df.shift[2000:end])
                s_low, s_high = Es - 3σs, Es + 3σs
                # Projected estimate.
                r = ratio_of_means(df.hproj[2000:end], df.vproj[2000:end])
                p_low, p_high = pquantile(r, [0.0015, 0.9985])

                @test s_low < E0 < s_high
                @test p_low < E0 < p_high
                @test all(-1 .≤ df.coherence .≤ 1)
                @test all(0 .≤ df.loneliness .≤ 1)
            end
            @testset "Initiator with $setup" begin
                H = HubbardMom1D(BoseFS((0,0,0,7,0,0,0)); u=6.0)
                add = starting_address(H)

                if setup == :PDVec
                    dv = PDVec(add => 3; initiator_threshold=1)
                else
                    dv = MPIData(InitiatorDVec(add => 3); setup, kwargs...)
                end
                s_strat = DoubleLogUpdate(targetwalkers=100)
                df = lomc!(H, dv; laststep=5000, s_strat).df

                # Shift estimate.
                Es, _ = mean_and_se(df.shift[2000:end])
                @test E0 ≤ Es
            end
            for initiator in (true, false)
                if setup === RMPI.mpi_one_sided
                    # Skip one sided here, because for some reason blocking fails
                    @warn "Skipping one-sided"
                    continue
                end
                @testset "AllOverlaps with $setup and initiator=$initiator" begin
                    H = HubbardMom1D(BoseFS((0,0,5,0,0)))
                    add = starting_address(H)
                    N = num_particles(add)
                    M = num_modes(add)

                    if setup == :PDVec
                        dv = PDVec(add => 3; style=IsDynamicSemistochastic(), initiator)
                    elseif initiator
                        dv = MPIData(InitiatorDVec(
                            add => 3; style=IsDynamicSemistochastic()
                        ); setup, kwargs...)
                    else
                        dv = MPIData(DVec(
                            add => 3; style=IsDynamicSemistochastic()
                        ); setup, kwargs...)
                    end

                    # Diagonal
                    replica = AllOverlaps(2; operator=ntuple(DensityMatrixDiagonal, M))
                    df,_ = lomc!(H, dv; replica, laststep=10_000)

                    density_sum = sum(1:M) do i
                        top = df[!, Symbol("c1_Op", i, "_c2")]
                        bot = df.c1_dot_c2
                        pmean(ratio_of_means(top, bot; skip=5000))
                    end
                    @test density_sum ≈ N rtol=1e-3

                    # Not Diagonal
                    ops = ntuple(x -> G2Correlator(x - cld(M, 2)), M)
                    replica = AllOverlaps(2; operator=ops)
                    df,_ = lomc!(H, dv; replica, laststep=10_000)

                    g2s = map(1:M) do i
                        top = df[!, Symbol("c1_Op", i, "_c2")]
                        bot = df.c1_dot_c2
                        pmean(ratio_of_means(top, bot; skip=5000))
                    end
                    for i in 1:cld(M, 2)
                        @test real(g2s[i]) ≈ real(g2s[end - i + 1]) rtol=1e-3
                        @test imag(g2s[i]) ≈ -imag(g2s[end - i + 1]) rtol=1e-3
                    end
                    @test real(sum(g2s)) ≈ N^2 rtol=1e-2
                    @test imag(sum(g2s)) ≈ 0 atol=1e-3
                end
            end
        end
    end

    @testset "Same seed gives same results" begin
        for kwargs in (
            (; setup=RMPI.mpi_point_to_point),
            (; setup=RMPI.mpi_one_sided, capacity=10_000),
            (; setup=RMPI.mpi_all_to_all),
        )
            # The entries in the vectors are the same (tested above), but they appear in a
            # different order.
            add = BoseFS2C((1, 1, 1, 1, 1), (1, 0, 0, 0, 0))
            H = BoseHubbardReal1D2C(add; v=2)
            dv = DVec(add => 1)

            dv_1 = MPIData(copy(dv); kwargs...)
            mpi_seed!(17)
            df_1 = lomc!(H, dv_1).df

            dv_2 = MPIData(copy(dv); kwargs...)
            mpi_seed!(17)
            df_2 = lomc!(H, dv_2).df

            @test df_1 == df_2
        end
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
