using Rimu
using Rimu.BitStringAddresses
using Rimu.BitStringAddresses: check_consistency, occupied_orbitals
using StaticArrays
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
    bs1 = BitAdd{40}(0xf342564fff)
    bs2 = BitAdd{40}(0xf342564ffd)
    bs3 = BitAdd{144}(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf")
    bs4 = BitAdd{44}(0xf342564fff)
    @test bs1 > bs2
    @test !(bs1 == bs2)
    @test !(bs1 < bs2)
    @test bs3 > bs2
    @test bs4 > bs1
    @test bs1 & bs2 == bs2
    @test bs1 | bs2 == bs1
    @test bs1 ⊻ bs2 == BitAdd{40}(2)
    @test ~bs1 == BitAdd{40}(~0xf342564fff & ~UInt64(0) >>> 24)
    @test count_ones(bs3) == 105
    @test count_zeros(bs3) == 39
    w = BitAdd{65}((UInt(31),UInt(15)))
    @test_throws ErrorException check_consistency(w)
    @test_throws ErrorException BitAdd((UInt(31),UInt(15)),65)
    wl = BitAdd((UInt(31),UInt(15)),85)
    @test bs3 == BitAdd(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf",144)
    fa = BitAdd{133}()
    @test trailing_zeros(bs1<<3) == 3
    @test trailing_ones(fa) == 133
    @test trailing_ones(fa>>100) == 33
    @test trailing_zeros(fa<<100) == 100
    @test leading_zeros(fa>>130) == 130
    @test leading_ones(fa<<130) == 3
    @test bitstring(bs3) == "000011110011010000100101011001001111111111011111000000001101111111011111110111111101111111011111110111111101111111011111110111111101111111011111"
end

using Rimu.Hamiltonians: numberoccupiedsites, bosehubbardinteraction

@testset "BoseFS" begin
    middle_full = BoseFS{67,100}(BitAdd{166}(
        SVector(1, ~UInt64(0), UInt64(1) << 63 | UInt64(2))
    ))
    middle_empty = BoseFS{10,150}(BitAdd{158}(SVector{3,UInt64}(255, 0, 3)))
    two_full = BoseFS{136,136}(BitAdd{158}(SVector{3,UInt64}(255, ~UInt64(0), ~UInt64(0))))
    @testset "onr" begin
        middle_full_onr = onr(middle_full)
        @test length(middle_full_onr) == 100
        @test middle_full_onr[63] == 66
        @test middle_full_onr[2] == 1
        @test middle_full_onr[1] == 0
        @test all(iszero, middle_full_onr[[3:62; 64:end]])

        middle_empty_onr = onr(middle_empty)
        @test length(middle_empty_onr) == 150
        @test middle_empty_onr[1] == 2
        @test middle_empty_onr[127] == 8
        @test all(iszero, middle_empty_onr[[2:126; 128:end]])

        two_full_onr = onr(two_full)
        @test length(two_full_onr) == 136
        @test two_full_onr[1] == 136
        @test all(iszero, two_full_onr[2:end])
    end
    @testset "numberoccupiedsites" begin
        @test numberoccupiedsites(middle_full) == 2
        @test numberoccupiedsites(middle_empty) == 2
        @test numberoccupiedsites(two_full) == 1
    end
    @testset "bosehubbardinteraction" begin
        @test bosehubbardinteraction(middle_full) == 66 * 65
        @test bosehubbardinteraction(middle_empty) == 8 * 7 + 2
        @test bosehubbardinteraction(two_full) == 136 * 135
    end

    @testset "occupied_orbitals" begin
        (bosons, orbital, bit), st = iterate(occupied_orbitals(middle_full))
        @test bosons == 1
        @test orbital == 2
        @test bit == 1
        (bosons, orbital, bit), st = iterate(occupied_orbitals(middle_full), st)
        @test bosons == 66
        @test orbital == 63
        @test bit == 63
        @test isnothing(iterate(occupied_orbitals(middle_full), st))
    end

    @testset "Randomized tests" begin
        function rand_onr(N, M)
            result = zeros(MVector{M,Int})
            for _ in 1:N
                result[rand(1:M)] += 1
            end
            return result
        end
        # This function checks if iteration works ok.
        function onr2(bose::BoseFS{N,M}) where {N,M}
            result = zeros(MVector{M,Int32})
            for (n, i, _) in occupied_orbitals(bose)
                @assert n ≠ 0
                result[i] = n
            end
            return SVector(result)
        end

        for _ in 1:10
            for (N, M) in ((16, 16), (64, 32), (200, 200), (100, 100), (200, 20), (20, 200))
                input = rand_onr(N, M)
                bose = BoseFS(input)
                @test num_particles(bose) == N
                @test num_modes(bose) == M
                @test onr(bose) == input
                @test numberoccupiedsites(bose) == count(!iszero, input)
                if bosehubbardinteraction(bose) != sum(input .* (input .- 1))
                    @show input
                end
                @test bosehubbardinteraction(bose) == sum(input .* (input .- 1))

                @test onr2(bose) == input
            end
        end
    end
end
