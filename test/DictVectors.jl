using LinearAlgebra
using Random
using Rimu
using Rimu.DictVectors
using Rimu.StochasticStyles: IsStochastic2Pop
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
            end
            @testset "isapprox" begin
                u = type(1 => 1.0, 2 => 2.0; kwargs...)
                v = type(1 => 1.1, 2 => 2.0; kwargs...)
                w = type(1 => 1.0, 2 => 2.0, 3 => 1e-9; kwargs...)

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
                v = type(0.5 => 0.1im; kwargs...)

                @test valtype(u) ≡ scalartype(u) ≡ Int
                @test keytype(u) ≡ Symbol
                @test eltype(u) ≡ Pair{Symbol,Int}
                @test isreal(u)
                @test ndims(u) == 1
                @test u isa AbstractDVec{Symbol,Int}

                @test valtype(v) ≡ scalartype(v) ≡ ComplexF64
                @test keytype(v) ≡ Float64
                @test eltype(v) ≡ Pair{Float64,ComplexF64}
                @test !isreal(v)
                @test ndims(v) == 1
                @test v isa AbstractDVec{Float64,ComplexF64}
            end
            @testset "show" begin
                h, _ = displaysize()
                @test length(split(sprint(show, type(zip(rand(100), rand(100))), '\n'))) < h
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
            end
            @testset "scale(!)" begin
                u = type(1 => 1.0 + im, 2 => -2.0im; kwargs...)
                v = type(1 => 3.5 + 3.5im, 2 => -7.0im; kwargs...)

                @test scale(u, 3.5) == 3.5u == v
                @test scale!!(copy(u), 3.5) == scale!(copy(u), 3.5) == v
                @test scale!(zerovector(u), u, 3.5) == v
                @test lmul!(2, copy(u)) == rmul!(copy(u), 2) == mul!(copy(u), u, 2) == v
                @test u == type(1 => 1.0 + im, 2 => -2.0im)
                @test isempty(0 * u)
                @test isempty(scale(u, 0))
                @test isempty(scale!(copy(u), 0))
                @test isempty(scale!(copy(u), u, 0))
                @test isempty(lmul!(0, copy(u)))
                @test isempty(rmul!(copy(u), 0))

                w = type(1 => 1; kwargs...)
                @test scale(w, 1 + im) == (1 + im) * w == type(1 => 1 + im)
            end
            @testset "add(!)" begin
                u = type(45 => 10.0, 12 => 3.5; kwargs...)
                v = type(45 => -10.0, 13 => -1.0, 12 => 1.0; kwargs...)
                w = type(13 => -1.0, 12 => 4.5; kwargs...)
                x = type(13 => -7, 45 => -90; kwargs...)

                @test add(u, v) == u + v == w
                @test add!(copy(u), v) == w
                @test axpy!(1, u, copy(v)) == w
                @test add(v, u, 2, -7) == 2u - 7v == x
                @test axpby!(2, u, -7, copy(v)) == x

                @test u + type(12 => -3.5 + im; kwargs...) == type(45 => 10, 12 => -im)
            end
            @testset "inner" begin
                u = type(zip(1:4, [1, 1.5, im, -im]); kwargs...)
                v = type(zip(1:3, [im, 1.2, -im]); kwargs...)
                result = im + 1.5*1.2 + -1
                @test inner(u, v) == dot(u, v) == result
                @test inner(v, u) == dot(v, u) == conj(result)
            end
            @testset "norm" begin
                vector = rand(10)
                u = type(zip(rand(Int, 10), vector); kwargs...)
                @test norm(normalize!(copy(u))) == norm(normalize(u)) ≈ 1
                @test norm(u) ≈ norm(vector)
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

                @test reduce(+, values(u)) == sum(vs)
                @test reduce(+, keys(u)) == sum(ks)

                @test mapreduce(x -> x + 1.1, +, values(u)) ≈ sum(x -> x + 1.1, vs)
                @test mapreduce(abs2, *, keys(u)) == prod(abs2, ks)
                @test mapreduce(last, max, pairs(u)) == maximum(vs)

                @test sum(sqrt ∘ abs2, u) ≈ sum(sqrt ∘ abs2, vs)
                @test minimum(abs2, values(u)) == minimum(abs2, vs)
                @test maximum(x -> x + 1.1, keys(u)) ≈ maximum(x -> x + 1.1, ks)
                @test prod(p -> p[1] - p[2], pairs(u)) == prod(p -> p[1] - p[2], ps)
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
    end
end

@testset "DVec" begin
    @testset "interface tests" begin
        test_dvec_interface(DVec; capacity=200)
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
        test_dvec_interface(InitiatorDVec; capacity=100)
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
