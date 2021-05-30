using LinearAlgebra
using Random
using Rimu
using Rimu.DictVectors
using Rimu.DictVectors: IsStochastic2Pop
using StaticArrays
using Suppressor
using Test

function test_dvec_interface(type, keys, vals, cap)
    K = eltype(keys)
    V = eltype(vals)
    pairs = [k => v for (k, v) in zip(keys, vals)]

    @testset "$type{$K,$V}" begin
        @testset "constructors" begin
            dvec1 = type(pairs...; capacity=cap)
            dvec2 = type(Dict(pairs...))
            for (k, v) in pairs
                @test dvec1[k] == dvec2[k] == v
            end
            @test dvec1 == dvec2

            dvec3 = type(Dict{K,V}(); capacity=cap)
            dvec4 = type{K,V}()

            @test isempty(dvec4)
            for (k, _) in pairs
                @test dvec3[k] == dvec4[k] == zero(V)
            end

            dvec5 = type(dvec2)
            @test dvec5 == dvec2

            dvec6 = type(IdDict(pairs))
            for (k, v) in pairs
                @test dvec6[k] == v
            end
        end
        @testset "empty, similar" begin
            dvec1 = type(pairs...; capacity=cap)
            dvec2 = empty(empty(empty(empty(dvec1))))
            dvec3 = similar(similar(similar(similar(dvec1))))
            @test typeof(dvec1) == typeof(dvec2) == typeof(dvec3)
            @test isempty(dvec2)
            @test isempty(dvec3)
            @test keytype(dvec1) == keytype(dvec2) == keytype(dvec3)
        end
        @testset "setindex, delete" begin
            dvec = type(pairs...; capacity=cap)
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
        end
        @testset "types and traits" begin
            dvec = type(pairs...; capacity=cap)
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
        @testset "norm" begin
            dvec = type(Dict(pairs))
            for p in (1, 2, Inf)
                @test norm(dvec, p) == norm(vals, p)
            end
            @test norm(dvec) == norm(dvec, 2)
            @test_throws ErrorException norm(dvec, 3)

            @test norm(empty(dvec)) == 0.0
            @test norm(empty(dvec), 1) == 0.0
            @test norm(empty(dvec), 2) == 0.0
            @test norm(empty(dvec), Inf) == 0.0
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
        @testset "mul!, *, rmul!" begin
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
        end
        @testset "ThreadsX" begin
            dvec = type(Dict(pairs))
            @test ThreadsX.sum(values(dvec)) ≈ sum(values(dvec))
            @test ThreadsX.sum(last, pairs(dvec)) ≈ sum(last, pairs(dvec))
        end
        @testset "add!" begin
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

            copy!(dvec2, dvec1)
            add!(dvec1, type{K,V}())
            @test dvec1 == dvec2
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
            @test dot(dvec1, (dvec2, dvec3)) ≈ norm(dvec1)^2
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
            @test UniformProjector() ⋅ dvec == sum(dvec)
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
        @test StochasticStyle(type(:a => 1.5; capacity=5)) isa IsDeterministic
        @test StochasticStyle(type(:a => 1 + 2im; capacity=5)) isa IsStochastic2Pop
        @test StochasticStyle(type(:a => SA[1 1; 1 1]; capacity=5)) isa StyleUnknown
    end
end

@testset "DVec" begin
    @testset "interface tests" begin
        keys1 = shuffle(1:20)
        vals1 = shuffle(1:20)
        test_dvec_interface(DVec, keys1, vals1, 100)

        keys2 = ['x', 'y', 'z', 'w', 'v']
        vals2 = [1.0 + 2.0im, 3.0 - 4.0im, 0.0 - 5.0im, -2.0 + 0.0im, 12.0 + im]
        test_dvec_interface(DVec, keys2, vals2, 200)
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
        test_dvec_interface(InitiatorDVec, keys1, vals1, 100)

        keys2 = ['x', 'y', 'z', 'w', 'v']
        vals2 = [1.0 + 2.0im, 3.0 - 4.0im, 0.0 - 5.0im, -2.0 + 0.0im, 12.0 + im]
        test_dvec_interface(InitiatorDVec, keys2, vals2, 200)
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

@testset "deprecations" begin
    warning = @capture_err copytight(DVec(:a => 1))
    @test warning ≠ ""

    warning = @capture_err DVec2(:a => 1)
    @test warning ≠ ""
    @test DVec2(:a => 1) isa DVec{Symbol,Int}

    @test_throws ErrorException capacity(DVec(:a => 1))

    warning = @capture_err IsStochastic()
    @test warning ≠ ""
    @test IsStochastic() === IsStochasticInteger()
end
