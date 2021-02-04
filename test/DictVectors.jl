using LinearAlgebra
using Random
using Rimu
using Rimu.DictVectors
using Test

"""
    test_dvec_interface(type::Type, keys::AbstractVector, values::AbstractVector, capacity)

Test the `AbstractDVec` interface.

# Example

```jldoctest
julia> test_dvec_interface(DVec, [1, 2, 3], [4.0, 5.0, 6.0], 10);
Test Summary:       | Pass  Broken  Total
DVec{Int64,Float64} |   71       1     72
```
"""
function test_dvec_interface(type, keys, values, cap)
    K = eltype(keys)
    V = eltype(values)
    @testset "$type{$K,$V}" begin
        pairs = [k => v for (k, v) in zip(keys, values)]

        @testset "constructors" begin
            dvec1 = type(pairs...; capacity=cap)
            dvec2 = type(Dict(pairs...), cap)
            for (k, v) in pairs
                @test dvec1[k] == dvec2[k] == v
            end

            dvec3 = type(Dict{K,V}(); capacity=cap)
            dvec4 = type{K,V}(cap)
            for (k, _) in pairs
                @test dvec3[k] == dvec4[k] == zero(V)
            end

            dvec5 = type(values)
            dvec6 = type(dvec5)
            for k in 1:length(values)
                @test dvec5[k] == dvec6[k] == values[k]
            end
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
        @testset "capacity" begin
            dvec1 = type(pairs...; capacity=cap)
            dvec2 = type(pairs...; capacity=2 * cap)
            # Capacity can get set to a weird value.
            @test capacity(dvec1) ≥ cap
            @test capacity(dvec2) ≥ 2 * cap
            @test capacity((dvec1, dvec2)) == capacity(dvec1) + capacity(dvec2)

            dvec3 = type(values)
            @test capacity(dvec3) ≥ length(values)
        end
        @testset "types and traits" begin
            dvec = type(pairs...; capacity=cap)
            @test dvec isa AbstractDVec{K,V}
            @test eltype(dvec) ≡ V
            @test keytype(dvec) ≡ K
            @test valtype(dvec) ≡ V
            @test pairtype(dvec) ≡ Pair{K,V}
            @test isreal(dvec) == (V <: Real)
            @test ndims(dvec) == 1
        end
        @testset "norm" begin
            dvec = type(Dict(pairs), cap)
            for p in (1, 2, Inf)
                @test norm(dvec, p) == norm(values, p)
            end
            @test norm(dvec) == norm(dvec, 2)
            @test_throws ErrorException norm(dvec, 3)
        end
        @testset "copy" begin
            dvec1 = type(Dict(pairs), cap)
            dvec2 = type{K,V}(cap)

            copy!(dvec2, dvec1)
            for (k, v) in pairs
                @test dvec2[k] == v
            end

            dvec3 = copy(dvec1)
            empty!(dvec1)
            for (k, v) in pairs
                @test dvec3[k] == v
            end

            dvec4 = copytight(dvec3)
            for (k, v) in pairs
                @test dvec4[k] == v
            end
        end
        @testset "fill" begin
            dvec = type(Dict(pairs), cap)
            fill!(dvec, zero(V))
            @test isempty(dvec)
            @test_throws ErrorException fill!(dvec, one(V))
        end
        @testset "mul!, *" begin
            dvec = type(Dict(pairs), cap)
            res1 = type{K,V}(cap)
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

            @test_broken missing * dvec ≡ dvec * missing ≡ missing
        end
        @testset "add!" begin
            dvec1 = type(Dict(pairs), cap)
            dvec2 = type(Dict(pairs[1:2:end]), cap)
            add!(dvec1, dvec2)
            for (i, (k, v)) in enumerate(pairs)
                if isodd(i)
                    @test dvec1[k] == 2v
                else
                    @test dvec1[k] == v
                end
            end

            copy!(dvec2, dvec1)
            add!(dvec1, type{K,V}(cap))
            @test dvec1 == dvec2
        end
        @testset "axpy!" begin
            dvec1 = type(Dict(pairs), cap)
            dvec2 = type(Dict(pairs[1:2:end]), cap)
            axpy!(V(2), dvec1, dvec2)
            for (i, (k, v)) in enumerate(pairs)
                if isodd(i)
                    @test dvec2[k] == 3v
                else
                    @test dvec2[k] == 2v
                end
            end

            ys = Tuple(empty(dvec1) for i in 1:Threads.nthreads())
            axpy!(2.0, dvec1, ys; batchsize=100)
        end
        @testset "axpby!" begin
            dvec1 = type(Dict(pairs), cap)
            dvec2 = type(Dict(pairs[1:2:end]), cap)
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
            dvec1 = type(Dict(pairs), cap)
            dvec2 = type(Dict(pairs[1:2:end]), cap)
            dvec3 = type(Dict(pairs[2:2:end]), cap)

            @test dvec1 ⋅ dvec1 ≈ norm(dvec1)^2
            @test dvec1 ⋅ dvec2 ≈ norm(dvec2)^2
            @test dot(dvec1, (dvec2, dvec3)) ≈ norm(dvec1)^2
        end
        @testset "iteration" begin
            dvec = type(Dict(pairs), cap)

            dvec_pairs = [kv for kv in Base.pairs(dvec)]
            @test issetequal(pairs, dvec_pairs)

            dvec_keys = [k for k in Base.keys(dvec)]
            @test issetequal(dvec_keys, keys)

            dvec_values = [kv for kv in dvec]
            @test issetequal(values, dvec_values)
            dvec_values = [k for k in Base.values(dvec)]
            @test issetequal(dvec_values, values)
        end
        @testset "projection" begin
            dvec = type(Dict(pairs), cap)
            @test UniformProjector() ⋅ dvec == sum(dvec)
            @test NormProjector() ⋅ dvec == norm(dvec, 1)
            @test Norm2Projector() ⋅ dvec == norm(dvec, 2)
        end
    end
end

@testset "DVec" begin
    keys1 = shuffle(1:10)
    vals1 = shuffle(1:10) .* rand((-1.0, 1.0), 10)
    test_dvec_interface(DVec, keys1, vals1, 10)

    keys2 = ['x', 'y', 'z', 'w', 'v']
    vals2 = [1.0 + 2.0im, 3.0 - 4.0im, 0.0 - 5.0im, -2.0 + 0.0im, 12.0 + im]
    test_dvec_interface(DVec, keys2, vals2, 100)
end

#=
DFVec has issues:
- its eltype is not really its eltype or valtype
@testset "DFVec" begin
    test_dvec_interface(DFVec, keys1, vals1, 10)
    test_dvec_interface(DFVec, keys2, vals2, 100)
end
=#

#=
FastDVec has an issue:
- `similar` is not implemented, so `copy(.)` does not work.
@testset "FastDVec" begin
    test_dvec_interface(FastDVec, keys1, vals1, 10)
    test_dvec_interface(FastDVec, keys2, vals2, 100)
end
=#

@testset "FastDVec" begin
    @test FastDVec(i => i^2 for i in 1:10; capacity = 30)|> length == 10
    myfda = FastDVec("a" => 42; capacity = 40)
    myda2 = FastDVec{String,Int}(40)
    myda2["a"] = 42
    @test haskey(myda2,"a")
    @test !haskey(myda2,"b")
    @test myfda == myda2
    myda2["c"] = 422
    myda2["d"] = 45
    myda2["f"] = 412

    @test length(myda2) == 4

    @test norm(myda2)≈592.9730179358922
    @test norm(myda2,1)==921
    @test norm(myda2*2.0,1) == 2*norm(myda2,1)
    @test norm(myda2,Inf)==422
    @inferred norm(myda2,1)
    delete!(myda2,"d")
    delete!(myda2,"c")
    @test myda2.emptyslots == [3,2]
    myda3 = similar(myda2)
    copyto!(myda3,myda2)
    @test length(myda3)==2
    myda3["q"]= 3
    delete!(myda3,"q")
    @test myda2==myda3
    fdv = FastDVec([rand() for i=1:1000], 2000)
    ki = keys(fdv)
    @test sort(collect(ki))==collect(1:1000)
    cdv = FastDVec(fdv)
    @test cdv == fdv
    fdv[1600] = 10.0
    cdv[1600] = 10.0
    @test cdv == fdv
end

@testset "DFVec" begin
    @test DFVec(:a => (2,3); capacity = 10) == DFVec(Dict(:a => (2,3)))
    dv = DVec(:a => 2; capacity = 100)
    cdv = copytight(dv)
    @test dv == cdv
    @test capacity(dv) > capacity(cdv)

    df = DFVec(Dict(3=>(3.5,true)))

    df2 = DFVec(Dict(3=>(3.5,true)), 200)
    @test capacity(df2) == 341
    df3 = DFVec{Int,Float64,Int}(30)
    @test capacity(df3) == 42
    df4 = DFVec([1,2,3,4])

    dv = DVec([1,2,3,4])

    df5 = DFVec(dv)
    @test eltype(dv) == Int
    length(dv)
    @test df4 == df5
    @test df4 ≢ df5
    @test df == df2
    @test df ≠ df4
    @test df5 == dv # checking keys, values, but not flags

    dd = Dict("a"=>1,"b"=>2,"c"=>3)
    ddv = DVec(dd)
    values(ddv)
    fddv = FastDVec(dd)
    @test collect(values(fddv)) == collect(values(ddv))
    ddf = DFVec(ddv)
    @test collect(values(ddf)) == collect(values(ddv))
    dt = Dict("a"=>(1,false),"b"=>(2,true),"c"=>(3,true))
    dtv = DVec(dt)
    collect(values(dtv))
    collect(keys(dtv))
    eltype(dtv)
    valtype(dtv)
    dtf = DFVec(dt)
    @test collect(flags(dtf)) == [true, true, false]
    @test DFVec(dtv) == dtf
    ndfv = DFVec(dtf,500,UInt8)
    dt = Dict(i => (sqrt(i),UInt16(i)) for i in 1:1000)
    dtv = DFVec(dt)
    dv = DVec(dtv)
    @test dtv == dv
    @test dv ≠ DVec(dt)
    dvt = DVec(dt)
    fdvt = DFVec(dvt)
    @test fdvt == dtv
    dtv[218] = (14.7648230602334, 0x00ff) # change flag
    @test fdvt ≠ dtv
end
