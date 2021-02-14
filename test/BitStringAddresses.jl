using Rimu
using Rimu.BitStringAddresses
using Rimu.BitStringAddresses: check_consistency
using Test

@testset "BSAdd" begin
    for (T, U) in ((BSAdd64, UInt64), (BSAdd128, UInt128))
        i = rand(U)
        j = rand(U)
        a = T(i)
        b = T(j)
        @test zero(a) == T(zero(U))
        @test (a < b) == (i < j)
        @test (a > b) == (i > j)

        # Unary stuff
        for f in (
            trailing_zeros,
            trailing_ones,
            leading_zeros,
            leading_ones,
            count_ones,
            count_zeros,
            iseven,
            isodd,
            bitstring,
        )
            @test f(a) == f(i)
            @test f(b) == f(j)
        end

        @test (~a).add == ~i
        @test (~b).add == ~j
        # Binary stuff
        for f in (&, |, ⊻)
            @test f(a, b).add == f(i, j)
            @test f(b, a).add == f(j, i)
        end
        # Shifts
        for f in (<<, >>, >>>)
            @test f(a, 10).add == f(i, 10)
            @test f(b, -5).add == f(j, -5)
        end
    end
end

@testset "BitStringAddresses.jl" begin
    # BitAdd
    bs = BitAdd{40}(0xf342564fff)
    bs1 = BitAdd{40}(0xf342564ffd)
    bs2 = BitAdd{144}(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf")
    bs3 = BitAdd{44}(0xf342564fff)
    @test bs > bs1
    @test !(bs == bs1)
    @test !(bs < bs1)
    @test bs2 > bs1
    @test bs3 > bs
    @test bs & bs1 == bs1
    @test bs | bs1 == bs
    @test bs ⊻ bs1 == BitAdd{40}(2)
    @test count_ones(bs2) == 105
    @test count_zeros(bs2) == 39
    w = BitAdd{65}((UInt(31),UInt(15)))
    @test_throws ErrorException check_consistency(w)
    @test_throws ErrorException BitAdd((UInt(31),UInt(15)),65)
    wl = BitAdd((UInt(31),UInt(15)),85)
    @test bs2 == BitAdd(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf",144)
    fa = BitAdd{133}()
    @test trailing_zeros(bs<<3) == 3
    @test trailing_ones(fa) == 133
    @test trailing_ones(fa>>100) == 33
    @test trailing_zeros(fa<<100) == 100
    @test leading_zeros(fa>>130) == 130
    @test leading_ones(fa<<130) == 3
    @test bitstring(bs2) == "000011110011010000100101011001001111111111011111000000001101111111011111110111111101111111011111110111111101111111011111110111111101111111011111"
end

#=
@testset "BoseFS" begin
    @test repr(BoseFS(bs2)) == "BoseFS{BitAdd}((5,7,7,7,7,7,7,7,7,7,7,2,0,0,0,0,0,0,0,5,10,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4,0,0,0,0))"
    @test onr(BoseFS(bs)) == [12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4]
    os = BoseFS{BitAdd}([12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4])
    @test os == BoseFS(bs)
    @test hash(os) == hash(BoseFS(bs))
    @test os.bs == bs
    bfs= BoseFS((1,0,2,1,2,1,1,3))
    onrep = onr(bfs)
    @test typeof(bfs)(onrep) == bfs
end
=#
