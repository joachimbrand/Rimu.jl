using LinearAlgebra
using Random
using Rimu
using Rimu.DictVectors
using Rimu.StochasticStyles: IsStochastic2Pop
using StaticArrays
using Suppressor
using Test

function test_dvec_interface(type, keys, vals; kwargs...)
    K = eltype(keys)
    V = eltype(vals)
    pairs = [k => v for (k, v) in zip(keys, vals)]

    @testset "$type{$K,$V}" begin
        @testset "constructors" begin
            dvec1 = type(pairs...; kwargs...)
            dvec2 = type(Dict(pairs...))
            for (k, v) in pairs
                @test dvec1[k] == dvec2[k] == v
            end
            @test dvec1 == dvec2

            dvec3 = type(Dict{K,V}(); kwargs...)
            dvec4 = type{K,V}()

            @test isempty(dvec4)
            for (k, _) in pairs
                @test dvec3[k] == dvec4[k] == zero(V)
            end

            dvec5 = type(dvec2; kwargs...)
            @test dvec5 == dvec2

            dvec6 = type(IdDict(pairs))
            for (k, v) in pairs
                @test dvec6[k] == v
            end

            @test sizehint!(dvec1, 1000) === dvec1

            @test localpart(dvec1) === localpart(localpart(dvec1))
        end
        @testset "empty, similar" begin
            dvec1 = type(pairs; kwargs...)
            dvec2 = empty(empty(empty(empty(dvec1))))
            dvec3 = similar(similar(similar(similar(dvec1))))
            @test typeof(dvec1) == typeof(dvec2) == typeof(dvec3)
            @test isempty(dvec2)
            @test isempty(dvec3)
            @test keytype(dvec1) == keytype(dvec2) == keytype(dvec3)

            dvec4 = empty(dvec1, String)
            @test keytype(dvec4) == keytype(dvec1)
            @test valtype(dvec4) == String

            dvec5 = empty(dvec1, String, String)
            @test keytype(dvec5) == String
            @test valtype(dvec5) == String
        end
        @testset "setindex, delete" begin
            dvec = type(pairs...; kwargs...)
            @test length(dvec) == length(pairs)
            zero!(dvec)
            @test length(dvec) == 0
            for (k, v) in shuffle(pairs)
                dvec[k] = v
            end
            for (k, v) in shuffle(pairs)
                @test dvec[k] == v
                delete!(dvec, k)
                @test iszero(dvec[k])
            end
            @test isempty(dvec)
            for (k, v) in shuffle(pairs)
                dvec[k] += v
            end
            for (k, v) in shuffle(pairs)
                dvec[k] -= v
            end
            @test isempty(dvec)
        end
        @testset "types and traits" begin
            dvec = type(pairs...; kwargs...)
            @test dvec isa AbstractDVec{K,V}

            @test eltype(dvec) ≡ Pair{K,V}
            @test eltype(typeof(dvec)) ≡ Pair{K,V}

            @test keytype(dvec) ≡ K
            @test keytype(typeof(dvec)) ≡ K

            @test valtype(dvec) ≡ V
            @test valtype(typeof(dvec)) ≡ V

            @test isreal(dvec) == (V <: Real)
            @test ndims(dvec) == 1
        end
        @testset "norm, normalize" begin
            dvec = type(Dict(pairs))
            for p in (1, 2, Inf)
                @test norm(dvec, p) == norm(vals, p)
            end
            @test norm(dvec) == norm(dvec, 2)
            @test_throws ErrorException norm(dvec, 3)

            @test norm(empty(dvec)) == 0
            @test norm(empty(dvec), 1) == 0
            @test norm(empty(dvec), 2) == 0
            @test norm(empty(dvec), Inf) == 0

            if valtype(dvec) == float(valtype(dvec))
                @test norm(normalize(dvec)) ≈ 1

                normalize!(dvec, 1)
                @test norm(dvec, 1) ≈ 1

                @test norm(dvec, Inf) == maximum(abs, values(dvec))
                normalize!(dvec, Inf)
                @test norm(dvec, Inf) ≈ 1
            end
        end
        @testset "copy" begin
            dvec1 = type(Dict(pairs))
            dvec2 = type{K,V}()

            copy!(dvec2, dvec1)
            for (k, v) in pairs
                @test dvec2[k] == v
            end

            dvec3 = copy(dvec1)
            empty!(dvec1)
            for (k, v) in pairs
                @test dvec3[k] == v
            end
        end
        @testset "all, any" begin
            dvec = type(DVec(pairs))

            @test all(Base.pairs(dvec)) do p
                p in pairs
            end
            @test all(Base.keys(dvec)) do k
                k in keys
            end
            @test all(Base.values(dvec)) do v
                v in vals
            end

            @test any(Base.pairs(dvec)) do p
                p == pairs[end]
            end
            @test any(Base.keys(dvec)) do k
                k == keys[1]
            end
            @test any(Base.values(dvec)) do v
                v == vals[end ÷ 2]
            end
        end
        @testset "mul!, *, rmul!, lmul!" begin
            dvec = type(Dict(pairs))
            res1 = type{K,V}()
            mul!(res1, dvec, one(V))
            @test res1 == dvec
            mul!(res1, dvec, zero(V))
            @test isempty(res1)
            mul!(res1, dvec, V(2))
            res2 = V(2) * dvec
            res3 = dvec * V(2)
            @test res1 == res2 == res3
            for (k, v) in pairs
                @test res1[k] == 2v
            end

            rmul!(dvec, V(3))
            for (k, v) in pairs
                @test dvec[k] == 3v
            end

            lmul!(V(2), dvec)
            for (k, v) in pairs
                @test dvec[k] == 6v
            end
            @test isempty(lmul!(0, copy(dvec)))
            @test isempty(rmul!(copy(dvec), 0))
        end
        @testset "add!, +, -" begin
            dvec1 = type(Dict(pairs))
            dvec2 = type(Dict(pairs[1:2:end]))
            add!(dvec1, dvec2)
            for (i, (k, v)) in enumerate(pairs)
                if isodd(i)
                    @test dvec1[k] == 2v
                else
                    @test dvec1[k] == v
                end
            end
            @test dvec1 == type(Dict(pairs)) + dvec2

            copy!(dvec2, dvec1)
            add!(dvec1, type{K,V}())
            @test dvec1 == dvec2

            @test isempty(dvec1 - dvec1)
            @test type(pairs) - type(pairs[1:2:end]) == type(pairs[2:2:end])
        end
        @testset "axpy!" begin
            dvec1 = type(Dict(pairs))
            dvec2 = type(Dict(pairs[1:2:end]))
            axpy!(V(2), dvec1, dvec2)
            for (i, (k, v)) in enumerate(pairs)
                if isodd(i)
                    @test dvec2[k] == 3v
                else
                    @test dvec2[k] == 2v
                end
            end
        end
        @testset "axpby!" begin
            dvec1 = type(Dict(pairs))
            dvec2 = type(Dict(pairs[1:2:end]))
            axpby!(V(2), dvec1, V(3), dvec2)
            for (i, (k, v)) in enumerate(pairs)
                if isodd(i)
                    @test dvec2[k] == 5v
                else
                    @test dvec2[k] == 2v
                end
            end
        end
        @testset "dot" begin
            dvec1 = type(Dict(pairs))
            dvec2 = type(Dict(pairs[1:2:end]))
            dvec3 = type(Dict(pairs[2:2:end]))

            @test dvec1 ⋅ dvec1 ≈ norm(dvec1)^2
            @test dvec1 ⋅ dvec2 ≈ norm(dvec2)^2
        end
        @testset "iteration" begin
            dvec = type(Dict(pairs))

            dvec_pairs = [kv for kv in Base.pairs(dvec)]
            @test issetequal(pairs, dvec_pairs)

            dvec_keys = [k for k in Base.keys(dvec)]
            @test issetequal(dvec_keys, keys)

            dvec_vals = [k for k in Base.values(dvec)]
            @test issetequal(dvec_vals, vals)
        end
        @testset "projection" begin
            dvec = type(Dict(pairs))
            @test UniformProjector() ⋅ dvec == sum(dvec) == sum(vals)
            @test UniformProjector()[2] == 1
            if valtype(dvec) isa AbstractFloat
                @test NormProjector() ⋅ dvec == norm(dvec, 1)
            end
            @test Norm2Projector() ⋅ dvec == norm(dvec, 2)
            @test Norm1ProjectorPPop() ⋅ dvec ==
                norm(real.(values(dvec)), 1) + im*norm(imag.(values(dvec)), 1)
            @test DictVectors.PopsProjector()⋅dvec ==
                real(collect(values(dvec))) ⋅ imag(collect(values(dvec)))
        end
        @testset "show" begin
            h, _ = displaysize()
            @test length(split(sprint(show, type(Dict(pairs))), '\n')) < h
        end
    end

    @testset "StochasticStyle" begin
        @test StochasticStyle(type(:a => 1)) isa IsStochasticInteger{Int}
        @test StochasticStyle(type(:a => 1.5; kwargs...)) isa IsDeterministic
        @test StochasticStyle(type(:a => 1 + 2im; kwargs...)) isa IsStochastic2Pop
        @test StochasticStyle(type(:a => SA[1 1; 1 1]; kwargs...)) isa StyleUnknown
    end
end

@testset "DVec" begin
    @testset "interface tests" begin
        keys1 = shuffle(1:20)
        vals1 = shuffle(1:20)
        test_dvec_interface(DVec, keys1, vals1; capacity=100)

        keys2 = ['x', 'y', 'z', 'w', 'v']
        vals2 = [1.0 + 2.0im, 3.0 - 4.0im, 0.0 - 5.0im, -2.0 + 0.0im, 12.0 + im]
        test_dvec_interface(DVec, keys2, vals2; capacity=200)
    end

    @testset "Stochastic styles convert eltype" begin
        dvec1 = DVec(:a => 0f0)
        @test valtype(dvec1) === Float32
        @test StochasticStyle(dvec1) == IsDeterministic{Float32}()

        dvec2 = DVec(:a => 1, style=IsDynamicSemistochastic())
        @test dvec2[:a] isa Float64

        dvec3 = DVec(:a => 1.0, style=IsStochasticInteger())
        @test dvec3 isa DVec{Symbol,Int}

        dvec4 = DVec(:a => 1.0, style=IsStochastic2Pop())
        @test !isreal(dvec4)
    end
end

@testset "InitiatorDVec" begin
    @testset "interface tests" begin
        keys1 = shuffle(1:20)
        vals1 = shuffle(1:20)
        test_dvec_interface(InitiatorDVec, keys1, vals1; capacity=100)

        keys2 = ['x', 'y', 'z', 'w', 'v']
        vals2 = [1.0 + 2.0im, 3.0 - 4.0im, 0.0 - 5.0im, -2.0 + 0.0im, 12.0 + im]
        test_dvec_interface(InitiatorDVec, keys2, vals2; capacity=200)
    end

    @testset "Stochastic styles convert eltype" begin
        dvec1 = InitiatorDVec(:a => 0f0)
        @test valtype(dvec1) === Float32
        @test StochasticStyle(dvec1) == IsDeterministic{Float32}()

        dvec2 = InitiatorDVec(:a => 1, style=IsDynamicSemistochastic())
        @test dvec2[:a] isa Float64

        dvec3 = InitiatorDVec(:a => 1.0, style=IsStochasticInteger())
        @test dvec3 isa InitiatorDVec{Symbol,Int}

        dvec4 = InitiatorDVec(:a => 1.0, style=IsStochastic2Pop())
        @test !isreal(dvec4)
    end
end

using Rimu.DictVectors: num_segments, is_distributed

@testset "PDVec" begin
    # This is done first to catch the maxlog=1 warnings
    @testset "operations" begin
        @testset "properties" begin
            pd1 = PDVec(zip(1:10, 10:-1.0:1))
            pd2 = PDVec(zip(1:10, 10:-1.0:1))
            pd2[1] += 1e-13
            pd3 = PDVec(zip(1:10, 10:-1.0:1); style=IsDynamicSemistochastic())

            @test num_segments(pd1) == Threads.nthreads()
            @test num_segments(pd2) == Threads.nthreads()

            @test StochasticStyle(pd1) ≡ IsDeterministic()
            @test StochasticStyle(pd3) ≡ IsDynamicSemistochastic()

            @test length(pd1) == length(pd2) == length(pd3) == 10
            @test pd1 == pd3
            @test pd1 != pd2
            @test pd2 ≈ pd3
            @test pd2 ≉ pd3 atol=1e-16

            @test !is_distributed(pd1)

            @test real(pd1) == pd1
            @test isempty(imag(pd1))
        end

        @testset "uneven vs even segemnts" begin
            pd1 = PDVec(zip(1:10, 10:-1:1))
            pd2 = PDVec{Int,Int}()
            copyto!(pd2, pd1)
            @test pd1 == pd2
            @test dot(pd1, pd2) > 0

            dv = DVec(pd1)
            pd3 = PDVec(dv)
            @test dv == pd3 == pd1

            add!(pd1, pd2)
            @test pd1 == 2 * pd2
        end

        @testset "map!" begin
            pd1 = PDVec(zip(2:2:12, [1,-1,2,-2,3,-3]))
            map!(x -> x + 1, values(pd1))
            @test length(pd1) == 5
            @test pd1[2] == 2

            pd2 = similar(pd1)
            map!(x -> x - 2, pd2, values(pd1))
            @test length(pd2) == 4
            @test pd2[6] == 1

            pd3 = map!(x -> x + 4, pd2, values(pd2))
            @test pd3 === pd2
            @test length(pd2) == 3
        end
    end

    @testset "interface tests" begin
        keys1 = shuffle(1:20)
        vals1 = shuffle(1:20)
        test_dvec_interface(PDVec, keys1, vals1)

        keys2 = ['x', 'y', 'z', 'w', 'v']
        vals2 = [1.0 + 2.0im, 3.0 - 4.0im, 0.0 - 5.0im, -2.0 + 0.0im, 12.0 + im]
        test_dvec_interface(PDVec, keys2, vals2)
    end

    @testset "Stochastic styles convert eltype" begin
        dvec1 = PDVec(:a => 0f0)
        @test valtype(dvec1) === Float32
        @test StochasticStyle(dvec1) == IsDeterministic{Float32}()

        dvec2 = PDVec(:a => 1, style=IsDynamicSemistochastic())
        @test dvec2[:a] isa Float64

        dvec3 = PDVec(:a => 1.0, style=IsStochasticInteger())
        @test dvec3 isa PDVec{Symbol,Int}

        dvec4 = PDVec(:a => 1.0, style=IsStochastic2Pop())
        @test !isreal(dvec4)
        @test isreal(real(dvec4))
        @test isreal(imag(dvec4))
    end
end
