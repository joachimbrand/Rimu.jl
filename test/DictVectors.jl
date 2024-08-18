using LinearAlgebra
using Random
using Rimu
using Rimu.DictVectors
using Rimu.StochasticStyles: IsStochastic2Pop, StochasticStyle
using StaticArrays
using Suppressor
using Test

function test_dvec_interface(type; kwargs...)
    @testset "$type w/ $kwargs" begin
        @testset "basics" begin
            @testset "constructors" begin
                u = type(zip((:a, :b, :c), (1, 2, 3)); kwargs...)
                v = type(Dict(:a => 1, :b => 2, :c => 3); kwargs...)
                w = type(:a => 1, :b => 2, :c => 3; kwargs...)

                @test u[:a] == v[:a] == w[:a] == 1
                @test u == v == w
                @test copy(u) == u
                @test copy(u) ≢ u

                @test sizehint!(u, 100) == u
                @test sizehint!(u, 100) ≡ u
                @test localpart(u) ≡ localpart(localpart(u))

                @test type(u; kwargs...) == u
                @test type(v; kwargs...) ≢ v
            end
            @testset "isapprox" begin
                u = type(1 => 1.0, 2 => 2.0; kwargs...)
                v = type(1 => 1.1, 2 => 2.0; kwargs...)
                w = type(1 => 1.0, 2 => 2.0, 3 => 1e-9; kwargs...)

                @test u ≠ v
                @test v ≠ w
                @test w ≠ u

                @test !isapprox(u, v; rtol=0.09)
                @test isapprox(u, v; rtol=0.1)
                @test !isapprox(v, u; rtol=0.09)
                @test isapprox(v, u; rtol=0.1)

                @test !isapprox(u, w)
                @test isapprox(u, w; atol=1e-9)
                @test !isapprox(w, u)
                @test isapprox(w, u; atol=1e-9)
            end
            @testset "properties" begin
                u = type(:a => 1, :b => 2; kwargs...)
                @test valtype(u) ≡ scalartype(u) ≡ Int
                @test keytype(u) ≡ Symbol
                @test eltype(u) ≡ Pair{Symbol,Int}
                @test isreal(u)
                @test ndims(u) == 1
                @test u isa AbstractDVec{Symbol,Int}

                v = type(0.5 => 0.1im; kwargs...)
                @test valtype(v) ≡ scalartype(v) ≡ ComplexF64
                @test keytype(v) ≡ Float64
                @test eltype(v) ≡ Pair{Float64,ComplexF64}
                @test !isreal(v)
                @test ndims(v) == 1
                @test v isa AbstractDVec{Float64,ComplexF64}
            end
            @testset "show" begin
                h, _ = displaysize()
                @test length(split(sprint(show, type(zip(rand(100), rand(100)))), '\n')) < h
            end
            @testset "$type is a default dict" begin
                u = type(zip(1:5, 1:5); kwargs...)

                for i in 5:-1:1
                    @test u[i] ≡ i
                end
                @test u[6] ≡ 0
                @test u[0] ≡ 0

                @test length(u) == 5
                u[5] = 0
                @test length(u) == 4
                delete!(u, 4)
                @test length(u) == 3
                u[3] -= 3
                @test length(u) == 2
                deposit!(u, 2, -2, 2 => 0)
                @test length(u) == 1

                @test u[1] ≡ 1
                for i in 2:6
                    @test u[i] ≡ 0
                end
                for i in 2:6
                    u[i] = i
                end
                for i in 1:6
                    @test u[i] ≡ i
                end
                @test length(u) == 6
            end
        end
        @testset "$type is a vector" begin
            @testset "zerovector(!)" begin
                u = type(9 => 1 + im, 2 => 2 - im, 13 => im, 4 => -im; kwargs...)
                e = type{Int,Complex{Int}}(; kwargs...)

                @test isempty(e)
                @test e == zerovector(u) == empty(u) == similar(u)
                @test e == zerovector!(copy(u)) == zerovector!(copy(u)) == empty!(copy(u))
                @test e ≡ zerovector!(e) ≡ zerovector!!(e) ≡ empty!(e)
                @test typeof(empty(u)) === typeof(u)
                @test typeof(zerovector(u)) === typeof(u)

                @test scalartype(zerovector(u, Int)) ≡ Int
                @test scalartype(empty(u, Int)) ≡ Int
                @test eltype(similar(u, Float64, Int)) ≡ Pair{Float64,Int}
                @test eltype(similar(u, String)) ≡ Pair{Int,String}
                @test eltype(similar(u, Float64, String)) ≡ Pair{Float64,String}

                v = type(1 => 1; kwargs...)
                @test zerovector!!(v, Int) ≡ v
                @test zerovector!!(v, Float64) ≢ v
            end
            @testset "scale(!)" begin
                u = type(1 => 1.0 + im, 2 => -2.0im; kwargs...)
                v = type(1 => 3.5 + 3.5im, 2 => -7.0im; kwargs...)
                w = type(1 => 3.5 + 3.5im, 2 => -7.0im, 3 => 0; kwargs...)
                y = type(1 => 0; kwargs...) # different value type

                @test scale(u, 3.5) == 3.5u == v
                @test scale!!(copy(u), 3.5) == scale!(copy(u), 3.5) == v
                @test scale!!(y, copy(u), 3.5) == v ≠ y
                @test scale!!(copy(w), copy(u), 3.5) == scale!(copy(w), copy(u), 3.5) == v
                @test scale!(zerovector(u), u, 3.5) == v
                @test lmul!(3.5, copy(u)) == v
                @test rmul!(copy(u), 3.5) == v
                @test mul!(copy(u), u, 3.5) == v
                @test u == type(1 => 1.0 + im, 2 => -2.0im)
                @test isempty(0 * u)
                @test isempty(scale(u, 0))
                @test isempty(scale!(copy(u), 0))
                @test isempty(scale!(copy(u), u, 0))
                @test isempty(lmul!(0, copy(u)))
                @test isempty(rmul!(copy(u), 0))

                w = type(1 => 1; kwargs...)
                @test scale(w, 1 + im) == (1 + im) * w == type(1 => 1 + im)
                @test scale!!(w, 1 + im) ≢ w
                @test scale!!(w, 1) ≡ w
            end
            @testset "add(!)" begin
                u = type(45 => 10.0, 12 => 3.5; kwargs...)
                v = type(45 => -10.0, 13 => -1.0, 12 => 1.0; kwargs...)
                w = type(13 => -1.0, 12 => 4.5; kwargs...)

                @test add(u, v) == u + v == w
                @test add!(copy(u), v) == w
                @test axpy!(1, u, copy(v)) == w

                x = type(13 => 7, 45 => 90; kwargs...)
                @test add(v, u, 2, -7) == 2u - 7v == x
                @test axpby!(2, u, -7, copy(v)) == x

                @test u + type(12 => -3.5 + im; kwargs...) == type(45 => 10, 12 => im)

                @test add!!(v, u, 2, -7) ≡ v
                @test add!!(v, u, 2 + im, -7) ≢ v
                @test add!!(v, u, 2, -7 - im) ≢ v
                @test add!!(v, type(1 => im; kwargs...), 2, -7 - im) ≢ v
            end
            @testset "inner" begin
                u = type(zip(1:4, [1, 1.5, im, -im]); kwargs...)
                v = type(zip(1:3, [im, 1.2, -im]); kwargs...)
                result = im + 1.5*1.2 + -1
                @test inner(u, v) == dot(u, v) == result
                @test inner(v, u) == dot(v, u) == conj(result)

                w = type{Int,Int}(; kwargs...)
                @test iszero(inner(w, w))
                @test iszero(inner(w, v))
                @test iszero(inner(u, w))
            end
            @testset "norm" begin
                vector = rand(10)
                u = type(zip(rand(Int, 10), vector); kwargs...)
                @test norm(normalize!(copy(u))) == norm(normalize(u)) ≈ 1
                @test norm(u) ≈ norm(vector)

                v = type{Int,ComplexF64}(; kwargs...)
                @test iszero(norm(v))
            end
        end
        @testset "iteration and reductions" begin
            @testset "iterate" begin
                ks = 1:10
                vs = rand(10)
                ps = map(Pair, ks, vs)

                u = type(Dict(ps); kwargs...)

                @test [kv for kv in pairs(u)] == collect(pairs(u))
                @test issetequal(ps, collect(pairs(u)))

                @test [k for k in keys(u)] == collect(keys(u))
                @test issetequal(collect(keys(u)), ks)

                @test [v for v in values(u)] == collect(values(u))
                @test issetequal(collect(values(u)), vs)
            end
            @testset "mapreduce" begin
                ks = 1:10
                vs = rand(1:100, 10)
                ps = map(Pair, ks, vs)

                u = type(Dict(ps); kwargs...)

                @test reduce(+, values(u); init=0) == sum(vs)
                @test reduce(*, keys(u)) == prod(ks)
                @test mapreduce(x -> x[1], +, pairs(u)) == sum(ks)

                @test mapreduce(x -> x + 1.1, +, values(u)) ≈ sum(x -> x + 1.1, vs)
                @test mapreduce(abs2, *, keys(u)) == prod(abs2, ks)
                @test mapreduce(last, max, pairs(u)) == maximum(vs)

                @test sum(sqrt ∘ abs2, u) ≈ sum(sqrt ∘ abs2, vs)
                @test minimum(abs2, values(u)) == minimum(abs2, vs)
                @test maximum(x -> x + 1.1, keys(u)) ≈ maximum(x -> x + 1.1, ks)
                @test prod(p -> p[1] - p[2], pairs(u)) == prod(p -> p[1] - p[2], ps)

                v = type{Int,Int}(; kwargs...)
                @test mapreduce(x -> x + 1, +, keys(v); init=0) == 0
            end
            @testset "all, any" begin
                ks = ['a', 'b', 'c', 'd', 'e', 'f']
                vs = rand(6)
                ps = map(Pair, ks, vs)
                dvec = type(ps; kwargs...)

                @test all(in(ps), pairs(dvec))
                @test all(in(ks), keys(dvec))
                @test all(in(vs), values(dvec))
                @test !all(in(vs[1:end-1]), values(dvec))

                @test any(==(ps[end]), pairs(dvec))
                @test any(==(ks[1]), keys(dvec))
                @test any(==(vs[end ÷ 2]), values(dvec))
                @test !any(>(1), values(dvec))
            end
        end
        @testset "projectors" begin
            u = type(zip(1:20, [rand(-5:5) + rand(-5:5) * im for _ in 1:20]); kwargs...)
            @test UniformProjector() ⋅ u == sum(u)
            @test UniformProjector()[2] == 1
            @test NormProjector() ⋅ u == norm(u, 1)
            @test Norm2Projector() ⋅ u == norm(u, 2)
            @test Norm1ProjectorPPop() ⋅ u ==
                norm(real.(values(u)), 1) + im*norm(imag.(values(u)), 1)
            @test DictVectors.PopsProjector() ⋅ u ==
                real(collect(values(u))) ⋅ imag(collect(values(u)))

            fu = freeze(u)
            @test fu isa AbstractProjector
            @test inner(fu, u) ≈ inner(u, fu) ≈ sum(abs2, u)
        end
        @testset "StochasticStyle" begin
            @test StochasticStyle(type(:a => 1; kwargs...)) isa IsStochasticInteger{Int}
            @test StochasticStyle(type(:a => 1.5; kwargs...)) isa IsDeterministic
            @test StochasticStyle(type(:a => 1 + 2im; kwargs...)) isa IsStochastic2Pop
            @test StochasticStyle(type(:a => SA[1 1; 1 1]; kwargs...)) isa StyleUnknown
        end
        @testset "Stochastic styles convert eltype" begin
            u = type(:a => 0f0; kwargs...)
            @test valtype(u) === Float32
            @test StochasticStyle(u) == IsDeterministic{Float32}()

            v = type(:a => 1; style=IsDynamicSemistochastic(), kwargs...)
            @test v[:a] isa Float64

            w = type(:a => 1.0; style=IsStochasticInteger(), kwargs...)
            @test w isa type{Symbol,Int}

            x = type(:a => 1.0; style=IsStochastic2Pop(), kwargs...)
            @test !isreal(x)
        end
    end
end

@testset "DVec" begin
    @testset "interface tests" begin
        test_dvec_interface(DVec; capacity=200)
    end

    @testset "DVec with StaticArray" begin
        sa = SA[1.0 2.0; 3.0 4.0]
        sai = SA[1 2; 3 4]
        @test DVec(:a => sa) == DVec(:a => sai)
        @test_throws ArgumentError DVec(:a => sai; style=IsDynamicSemistochastic())
        dict = Dict(:a => sai)
        @test_throws ArgumentError DVec(dict; style=IsDynamicSemistochastic())
        @test DVec(Dict(:a => sa)) == DVec(:a => sa)
        dv = DVec(:a => sai)
        @test_throws ArgumentError empty(dv; style=IsDynamicSemistochastic())
    end
end

@testset "InitiatorDVec" begin
    @testset "interface tests" begin
        test_dvec_interface(InitiatorDVec; capacity=100)
        test_dvec_interface(InitiatorDVec; initiator=DictVectors.CoherentInitiator(1))
    end
end

using Rimu.DictVectors: num_segments, is_distributed, SegmentedBuffer, replace_collections!

@testset "PDVec" begin
    @testset "constructor errors" begin
        @test_throws ArgumentError PDVec(1 => 1; initiator="none")
        @test_throws ArgumentError PDVec(1 => 1; communicator="none")
    end

    @testset "internals" begin
        @testset "SegmentedBuffer" begin
            buf = SegmentedBuffer{Float64}()
            vecss = (
                ([1.0,2.0,3.0], [4.0,5.0], [6.0,7.0,8.0,9.0,0.0], Float64[], [10.0]),
                (Float64[], rand(5), rand(3)),
                (Float64[],),
            )
            for vecs in vecss
                replace_collections!(buf, vecs)

                @test length(buf) == length(vecs)
                @test buf.offsets[end] == sum(length, vecs)
                @test buf.buffer == reduce(vcat, vecs)
                for (i, v) in enumerate(vecs)
                    @test buf[i] == v
                end
            end
        end
    end

    @testset "operations" begin
        @testset "properties" begin
            pd1 = PDVec(zip(1:10, 10:-1.0:1))
            pd2 = PDVec(zip(1:10, 10:-1.0:1))
            pd2[1] += 1e-13
            pd3 = PDVec(zip(1:10, 10:-1.0:1); style=IsDynamicSemistochastic())
            pd4 = PDVec(zip(1:9, 10:-1.0:2))

            @test num_segments(pd1) == Threads.nthreads()
            @test num_segments(pd2) == Threads.nthreads()

            @test StochasticStyle(pd1) ≡ IsDeterministic()
            @test StochasticStyle(pd3) ≡ IsDynamicSemistochastic()

            @test length(pd1) == length(pd2) == length(pd3) == 10
            @test pd1 == pd3
            @test pd1 != pd2
            @test pd2 ≈ pd3
            @test pd2 ≉ pd3 atol=1e-16
            @test pd3 != pd4

            @test length(pd4 - pd3) == 1

            @test dot(pd1, pd1) == sum(abs2, values(pd1))
            @test dot(pd1, pd3) == sum(abs2, values(pd1))

            @test !is_distributed(pd1)
            @test !is_distributed(values(pd1))

            @test real(pd1) == pd1
            @test isempty(imag(pd1))
        end

        @testset "DVec with PDVec" begin
            pv = PDVec(zip(1:10, 10:-1:1))
            dv = PDVec{Int,Int}()
            copyto!(dv, pv)
            @test dv == pv
            @test dot(dv, pv) / (norm(dv) * norm(pv)) ≈ 1

            add!(pv, dv)
            @test pv == 2 * dv
            @test 0.5 * pv == dv
        end

        @testset "map(!)" begin
            pd1 = PDVec(zip(2:2:12, [1, -1, 2, -2, 3, -3]))
            pd1_m = map(x -> x + 1, values(pd1))
            @test length(pd1_m) == 5
            @test pd1_m[2] == 2
            @test pd1 ≠ pd1_m

            map!(x -> x + 1, values(pd1))
            @test length(pd1) == 5
            @test pd1[2] == 2

            @test pd1_m == pd1

            pd2 = similar(pd1)
            map!(x -> x - 2, pd2, values(pd1))
            @test length(pd2) == 4
            @test pd2[6] == 1

            pd3 = map!(x -> x + 4, pd2, values(pd2))
            @test pd3 === pd2
            @test length(pd2) == 3
        end

        @testset "filter(!)" begin
            pd1 = PDVec(zip(1:6, [1, -1, 2, -2, 3, -3]))
            pd2 = similar(pd1)
            pd2 = filter!(>(0), pd2, values(pd1))
            @test all(>(0), values(pd2))
            @test length(pd2) == 3

            pd3 = filter(x -> x % 2 == 0, keys(pd1))
            @test all(<(0), values(pd3))
            @test length(pd3) == 3

            filter!(p -> p[1] - p[2] ≠ 0, pairs(pd1))
            @test length(pd1) == 5

            filter!(iseven, keys(pd1))
            @test length(pd1) == 3

            filter!(iseven, pd1, values(pd1))
            @test length(pd1) == 1
        end

        @testset "operator dot" begin
            add = FermiFS2C((1,1,0,0), (0,0,1,1))
            H = HubbardMom1D(add)
            T = Transcorrelated1D(add)
            D = DensityMatrixDiagonal(1)

            dv1 = H * DVec(add => 1.0)
            dv2 = T * DVec(add => 1.0)
            pv1 = H * PDVec(add => 1.0)
            pv2 = T * PDVec(add => 1.0)
            wm = PDWorkingMemory(pv1)

            for op in (H, T, D)
                @test dot(pv1, op, pv2) ≈ dot(dv1, op, dv2)
                @test dot(dv1, op, pv2) ≈ dot(pv1, op, dv2)

                @test dot(pv1, op, pv2, wm) ≈ dot(pv1, op, dv2)
            end
        end
    end

    @testset "interface tests" begin
        test_dvec_interface(PDVec)
        test_dvec_interface(PDVec; initiator=true)
    end
end

# tests for FrozenDVec
@testset "FrozenDVec" begin
    dv = DVec(BoseFS(1, 2) => 2.3, BoseFS(2, 1) => 2.2)
    fdv = freeze(dv)
    @test collect(fdv) == collect(pairs(fdv)) == collect(pairs(dv))
    @test walkernumber(fdv) == walkernumber(dv)
    @test collect(keys(fdv)) == collect(keys(dv))
    @test collect(values(fdv)) == collect(values(dv))
    @test length(fdv) == length(dv)
    @test fdv == freeze(fdv)
    @test inner(fdv, dv) ≈ norm(dv, 2)^2
end
