using Rimu
using Rimu.BitStringAddresses
using Rimu.BitStringAddresses: remove_ghost_bits, has_ghost_bits, bitstring_storage
using Rimu.BitStringAddresses: check_consistency, occupied_orbitals
using StaticArrays
using Test

@testset "BitString" begin
    @testset "bitstirng_storage" begin
        @test length(bitstring_storage(Val(1))) == 1
        @test eltype(bitstring_storage(Val(1))) == UInt64
        @test length(bitstring_storage(Val(33))) == 1
        @test eltype(bitstring_storage(Val(33))) == UInt64
        @test length(bitstring_storage(Val(64))) == 1
        @test eltype(bitstring_storage(Val(64))) == UInt64

        @test length(bitstring_storage(Val(65))) == 1
        @test eltype(bitstring_storage(Val(65))) == UInt128
        @test length(bitstring_storage(Val(128))) == 1
        @test eltype(bitstring_storage(Val(128))) == UInt128

        @test length(bitstring_storage(Val(129))) == 3
        @test eltype(bitstring_storage(Val(129))) == UInt64
        @test length(bitstring_storage(Val(128))) == 1
        @test eltype(bitstring_storage(Val(128))) == UInt128
    end

    @testset "Constructors" begin
        @testset "Constructiong from small ints" begin
            @test BitString{5}(1) ==
                BitString{5}(0x1) ==
                BitString{5}(UInt16(1)) ==
                BitString{5}(big"1") ==
                BitString{5}(Int128(1))

            @test BitString{129}(1) ==
                BitString{129}(0x1) ==
                BitString{129}(UInt16(1)) ==
                BitString{129}(big"1") ==
                BitString{129}(Int128(1))
        end
        @testset "Constructing from 128-bit int. Number of chunks depends on B." begin
            bs128a = @inferred BitString{129}(0x00000decafc0ffee0000000deadbeefe)
            @test bs128a.chunks[1] ≡ zero(UInt64)
            @test bs128a.chunks[2] ≡ 0xdecafc0ffee
            @test bs128a.chunks[3] ≡ 0xdeadbeefe
            @test bs128a == BitString{129}(big"0xdecafc0ffee0000000deadbeefe")

            bs128b = @inferred BitString{128}(0x00000decafc0ffee0000000deadbeefe)
            @test bs128b.chunks[1] ≡ 0x00000decafc0ffee0000000deadbeefe
            @test bs128b == BitString{128}(big"0xdecafc0ffee0000000deadbeefe")
        end
        @testset "Zero" begin
            a = BitString{100}(0)
            @test zero(a) == a

            b = BitString{250}(0x0)
            @test zero(b) == b

            c = BitString{16}(big"0")
            @test zero(typeof(c)) == c
        end
        @testset "no ghost bits" begin
            @test !has_ghost_bits(BitString{5}(0b11111))
            @test !has_ghost_bits(BitString{5}(0b111111))
            @test !has_ghost_bits(BitString{100}(rand(UInt128)))
            @test !has_ghost_bits(BitString{129}(big"0xffffffffffffffffffffffffffffffffffffffff"))
        end
    end

    @testset "Counting operations" begin
        s = BitString{167}(big"0x7fffffffffffffffbff03fffbffffde7d")
    end

    @testset "Bitwise operations" begin
        function rand_bitstring(B)
            s = rand(bitstring_storage(Val(B)))
            return remove_ghost_bits(BitString{B,length(s),eltype(s)}(s))
        end

        for B in (32, 65, 120, 200, 250)
            a = rand_bitstring(B)
            b = rand_bitstring(B)
            one = ~zero(a)

            @test iszero(a & ~a)
            @test a | ~a == ~zero(a)
            @test a ⊻ a == zero(a)
            @test (a << 8) >> 8 ≠ a
            @test (a >> 8) << 8 ≠ a
            @test (a >> 23) >> 41 == a >>> 64
            @test (a << 23) << 41 == a << 64
            @test a >> B == zero(a)
            @test a << B == zero(a)
            @test a << -13 == a >> 13
            @test a >> -17 == a << 17
            @test a >> 0 == a
            @test a << 0 == a

            @test (a >> 13) << 13 == a & (one << 13)
            @test (a << 13) >> 13 == a & (one >> 13)

            @test !has_ghost_bits(~zero(a))
            @test !has_ghost_bits(a << 2)
            @test !has_ghost_bits(a | b)
            @test !has_ghost_bits(a ⊻ b)
            @test !has_ghost_bits(a & b)
            @test !has_ghost_bits(a >> 2)
        end
    end
end

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

using Rimu.Hamiltonians: numberoccupiedsites, bosehubbardinteraction

@testset "BoseFS" begin
    middle_full = BoseFS{67,100}(
        BitString{166,3,UInt64}(SVector(1, ~UInt64(0), UInt64(1) << 63 | UInt64(2)))
    )
    middle_empty = BoseFS{10,150}(
        BitString{159,3,UInt64}(SVector{3,UInt64}(255, 0, 3))
    )
    two_full = BoseFS{136,136}(
        BitString{271,5,UInt64}(
            SVector(UInt64(0), UInt64(0), UInt64(255), ~UInt64(0), ~UInt64(0))
        )
    )
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
            return SVector(result)
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

        for (N, M) in ((16, 16), (64, 32), (200, 200), (100, 100), (200, 20), (20, 200))
            @testset "$N, $M" begin
                for _ in 1:10
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
end
