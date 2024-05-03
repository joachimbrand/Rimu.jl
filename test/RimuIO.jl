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

@testset "Addresses" begin
    for addr in (
        BitString{10}(0b1100110011),
        SortedParticleList((1, 0, 1, 0, 0, 2, 3)),
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
