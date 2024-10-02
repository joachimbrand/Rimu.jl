using LinearAlgebra
using Random
using Rimu
using Test
using KrylovKit
using StaticArrays
using MPI

using Rimu.StatsTools
using Rimu.DictVectors: PointToPoint, AllToAll, copy_to_local!, NonInitiatorValue

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

@mpi_root @info "Running MPI tests..."
mpi_allprintln("hello")

# Ignore all printing on ranks other than root. Passing an argument to this script disables
# this.
if isnothing(get(ARGS, 1, nothing)) && mpi_rank() != mpi_root
    redirect_stderr(devnull)
    redirect_stdout(devnull)
end
if !isnothing(get(ARGS, 1, nothing))
    @mpi_root @info "Debug printing enabled"
end

@testset "MPI tests" begin
    @testset "PDVec" begin
        addr = FermiFS2C((1,1,1,0,0,0),(1,1,1,0,0,0))
        ham = HubbardRealSpace(addr)

        dv = DVec(addr => 1.0)
        if mpi_size() > 1
            K = typeof(addr)
            V = Rimu.DictVectors.NonInitiatorValue{Float64}
            for communicator in (AllToAll{K,V}(report=true), PointToPoint{K,V}(report=true))
                @testset "$(nameof(typeof(communicator)))" begin
                    pv = PDVec(addr => 1.0; communicator)
                    @test pv.communicator isa typeof(communicator)

                    res_dv = eigsolve(ham, dv, 1, :SR; issymmetric=true)
                    res_pv = eigsolve(ham, pv, 1, :SR; issymmetric=true)
                    # `issymmetric` kwarg only needed for pre v1.9 julia versions

                    @test res_pv[2][1].communicator isa typeof(communicator)

                    @test res_dv[1][1] ≈ res_pv[1][1] || res_dv[1][1] ≈ -res_pv[1][1]

                    wm = working_memory(pv)
                    local_copy = DVec(copy_to_local!(wm, res_pv[2][1]))
                    @test res_dv[2][1] ≈ local_copy || res_dv[2][1] ≈ -local_copy

                    dv = copy(res_dv[2][1])
                    pv = copy(res_pv[2][1])
                    @test norm(pv) ≈ 1
                    @test length(pv) == length(dv)
                    @test sum(values(pv)) ≈ sum(values(dv)) ||
                        sum(values(pv)) ≈ -sum(values(dv))
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

                    @test mpi_size() == mpi_size(pv.communicator)
                    @test_throws DictVectors.CommunicatorError iterate(pairs(pv))

                    local_pairs = collect(pairs(localpart(pv)))
                    local_vals = sum(abs2, values(localpart(pv)))

                    total_len = MPI.Allreduce(length(local_pairs), +, MPI.COMM_WORLD)
                    total_vals = MPI.Allreduce(local_vals, +, MPI.COMM_WORLD)

                    @test total_len == length(dv)
                    @test total_vals == sum(abs2, values(pv))
                end
            end

            @testset "dot" begin
                addr = BoseFS((0,0,10,0,0))
                H = HubbardMom1D(addr)
                D = DensityMatrixDiagonal(1)
                G2 = G2RealSpace(PeriodicBoundaries(5))

                K = typeof(addr)
                V = Rimu.DictVectors.NonInitiatorValue{Float64}
                for communicator in (AllToAll{K,V}(), PointToPoint{K,V}())
                    @testset "$(nameof(typeof(communicator)))" begin
                        # Need to seed here to get the same random vectors on all ranks.
                        Random.seed!(1)
                        pairs_v = [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:100]
                        pairs_w = [BoseFS(rand_onr(10, 5)) => 2 - 4rand() for _ in 1:20]

                        v = DVec(pairs_v)
                        w = DVec(pairs_w)
                        pv = PDVec(pairs_v; communicator)
                        pw = PDVec(pairs_w; communicator)

                        @test norm(v) ≈ norm(pv)
                        @test length(w) == length(pw)

                        @test dot(v, w) ≈ dot(pv, pw)

                        @test dot(freeze(pw), pv) ≈ dot(w, v) ≈ dot(pw, freeze(pv))
                        @test dot(freeze(pv), pw) ≈ dot(v, w) ≈ dot(pv, freeze(pw))
                        wm = PDWorkingMemory(pv)

                        for op in (H, D)
                            @test dot(v, op, w) ≈ dot(pv, op, pw)
                            @test dot(w, op, v) ≈ dot(pw, op, pv)

                            @test dot(v, op, w) ≈ dot(pv, op, pw, wm)
                            @test dot(w, op, v) ≈ dot(pw, op, pv, wm)

                            pu = op * pv
                            u = op * v
                            @test length(u) == length(pu)
                            @test norm(u, 1) ≈ norm(pu, 1)
                            @test norm(u, 2) ≈ norm(pu, 2)
                            @test norm(u, Inf) ≈ norm(pu, Inf)
                        end
                        # dot only for G2
                        @test dot(v, G2, w) ≈ dot(pv, G2, pw)
                        @test dot(w, G2, v) ≈ dot(pw, G2, pv)

                        @test dot(v, G2, w) ≈ dot(pv, G2, pw, wm)
                        @test dot(w, G2, v) ≈ dot(pw, G2, pv, wm)

                        @test dot(pv, (H, D), pw, wm) == (dot(pv, H, pw), dot(pv, D, pw))
                        @test dot(pv, (H, D), pw) == (dot(pv, H, pw), dot(pv, D, pw))
                    end
                end
            end
        end
    end

    @testset "Ground state energy estimates" begin
        # H1 = HubbardReal1D(BoseFS((1,1,1,1,1,1,1)); u=6.0)
        # E0 = eigvals(Matrix(H1))[1]
        E0 = -4.628524493494574
        mpi_seed!(1000_000_000)

        @testset "Regular FCIQMC with post-steps" begin
            H = HubbardReal1D(BoseFS((1,1,1,1,1,1,1)); u=6.0)
            dv = PDVec(starting_address(H) => 3; style=IsDynamicSemistochastic())

            post_step_strategy = (
                ProjectedEnergy(H, dv),
                SignCoherence(copy(localpart(dv))),
                WalkerLoneliness(),
                Projector(proj_1=Norm2Projector()),
            )
            prob = ProjectorMonteCarloProblem(
                H; start_at=dv, post_step_strategy, last_step=5000
            )
            df = DataFrame(solve(prob))

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
        @testset "Initiator FCIQMC" begin
            H = HubbardMom1D(BoseFS((0,0,0,7,0,0,0)); u=6.0)
            addr = starting_address(H)

            dv = PDVec(addr => 3; initiator_threshold=1)
            shift_strategy = DoubleLogUpdate(targetwalkers=100)
            prob = ProjectorMonteCarloProblem(
                H; start_at=dv, last_step=5000, shift_strategy
            )
            df = DataFrame(solve(prob))

            # Shift estimate.
            Es, _ = mean_and_se(df.shift[2000:end])
            @test E0 ≤ Es
        end
        for initiator in (true, false)
            @testset "AllOverlaps with initiator=$initiator" begin
                H = HubbardMom1D(BoseFS((0,0,5,0,0)))
                addr = starting_address(H)
                N = num_particles(addr)
                M = num_modes(addr)

                dv = PDVec(addr => 3; style=IsDynamicSemistochastic(), initiator)

                # Diagonal
                replica_strategy = AllOverlaps(2; operator=ntuple(DensityMatrixDiagonal, M))
                prob = ProjectorMonteCarloProblem(
                    H; start_at=dv, replica_strategy, last_step=10_000
                )
                df = DataFrame(solve(prob))

                density_sum = sum(1:M) do i
                    top = df[!, Symbol("c1_Op", i, "_c2")]
                    bot = df.c1_dot_c2
                    pmean(ratio_of_means(top, bot; skip=5000))
                end
                @test density_sum ≈ N rtol=1e-3

                # Not Diagonal
                ops = ntuple(x -> G2MomCorrelator(x - cld(M, 2)), M)
                replica_strategy = AllOverlaps(2; operator=ops)
                prob = ProjectorMonteCarloProblem(
                    H; start_at=dv, replica_strategy, last_step=10_000
                )
                df = DataFrame(solve(prob))

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

    # Make sure all ranks came this far.
    @testset "Finish" begin
        # MPI.jl currently doesn't properly map logical operators (MPI v0.20.8)
        @test MPI.Allreduce(true, MPI.LAND, mpi_comm())
        # @test MPI.Allreduce(true, &, mpi_comm())
    end
end

if mpi_rank() ≠ mpi_root
    redirect_stderr(stderr)
    redirect_stdout(stdout)
end
