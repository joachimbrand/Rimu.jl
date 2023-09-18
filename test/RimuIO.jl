using Test
using Rimu
using Arrow
using Rimu: RimuIO
using DataFrames

tmpdir = mktempdir()

@testset "save_df, load_df" begin
    file = joinpath(tmpdir, "tmp.arrow")
    rm(file; force=true)

    df = DataFrame(a=[1, 2, 3], b=Complex{Float64}[1, 2, 3+im], d=rand(Complex{Int}, 3))
    RimuIO.save_df(file, df)
    df2 = RimuIO.load_df(file)
    @test df == df2

    rm(file)

    # test compression
    r = 10_005
    df2 = DataFrame(a = collect(1:r), b = rand(1:30,r))
    RimuIO.save_df(file, df2)
    compressed = filesize(file)
    rm(file)
    RimuIO.save_df(file, df2, compress=nothing)
    uncompressed = filesize(file)
    rm(file)
    @test compressed < uncompressed
end
@testset "save_dvec, load_dvec" begin
    # BSON is currently broken on 1.8
    if VERSION ≤ v"1.7"
        file1 = joinpath(tmpdir, "tmp1.bson")
        file2 = joinpath(tmpdir, "tmp2.bson")
        rm(file1; force=true)
        rm(file2; force=true)

        add = BoseFS2C((1,1,0,1), (1,1,0,0))
        dv = InitiatorDVec(add => 1.0, style=IsDynamicSemistochastic(abs_threshold=3.5))
        H = BoseHubbardMom1D2C(add)

        _, state = lomc!(H, dv; replica=NoStats(2))
        RimuIO.save_dvec(file1, state.replicas[1].v)
        RimuIO.save_dvec(file2, state.replicas[2].v)

        dv1 = RimuIO.load_dvec(file1)
        dv2 = RimuIO.load_dvec(file2)

        @test dv1 == state.replicas[1].v
        @test typeof(dv2) == typeof(state.replicas[1].v)
        @test StochasticStyle(dv1) == StochasticStyle(state.replicas[1].v)
        @test storage(dv2) == storage(state.replicas[2].v)

        rm(file1; force=true)
        rm(file2; force=true)
    end
end
@testset "Addresses" begin
    for addr in (
        near_uniform(BoseFS{10, 10}),
        BoseFS(101, 5 => 10),
        FermiFS((1,1,1,0,0,0)),
        FermiFS2C(near_uniform(FermiFS{50,100}), FermiFS(100, 1 => 1)),
        CompositeFS(
            BoseFS((1,1,1,1,1)),
            FermiFS((1,0,0,0,0)),
            BoseFS((1,1,0,0,0)),
            FermiFS((1,1,1,0,0)),
        ),
    )
        @testset "$(typeof(addr))" begin
            @testset "ArrowTypes interface" begin
                arrow_name = ArrowTypes.arrowname(typeof(addr))
                ArrowType = ArrowTypes.ArrowType(typeof(addr))
                serialized = ArrowTypes.toarrow(addr)
                @test typeof(serialized) ≡ ArrowType
                meta = ArrowTypes.arrowmetadata(typeof(addr))

                # This takes care of some weirdness with how Arrow handles things.
                if addr isa CompositeFS
                    T = NamedTuple{ntuple(Symbol, num_components(addr)), typeof(serialized)}
                    JuliaType = ArrowTypes.JuliaType(Val(arrow_name), T, meta)
                    result = ArrowTypes.fromarrow(JuliaType, serialized...)
                else
                    T = typeof(serialized)
                    JuliaType = ArrowTypes.JuliaType(Val(arrow_name), T, meta)
                    result = ArrowTypes.fromarrow(JuliaType, serialized)
                end

                @test result ≡ addr
            end
            @testset "saving and loading" begin
                file = joinpath(tmpdir, "tmp-addr.arrow")
                RimuIO.save_df(file, DataFrame(addr = [addr]))
                result = only(RimuIO.load_df(file).addr)
                @test result ≡ addr
                rm(file; force=true)
            end
        end
    end
end
