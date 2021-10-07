using Rimu
using Rimu.BitStringAddresses
using Rimu.BitStringAddresses: remove_ghost_bits, has_ghost_bits
using Rimu.BitStringAddresses: occupied_modes, update_component
using Random
using StaticArrays
using Test

include("excitation_tests.jl")

@testset "BitString" begin
    @testset "num_chunks for Val(B)" begin
        @test_throws ArgumentError num_chunks(Val(0))
        @test_throws ArgumentError num_chunks(Val(-15))
        @test num_chunks(Val(1)) == (1, UInt8)
        @test num_chunks(Val(8)) == (1, UInt8)
        @test num_chunks(Val(9)) == (1, UInt16)
        @test num_chunks(Val(16)) == (1, UInt16)
        @test num_chunks(Val(17)) == (1, UInt32)
        @test num_chunks(Val(32)) == (1, UInt32)
        @test num_chunks(Val(33)) == (1, UInt64)
        @test num_chunks(Val(64)) == (1, UInt64)
        @test num_chunks(Val(65)) == (2, UInt64)
        @test num_chunks(Val(128)) == (2, UInt64)
        @test num_chunks(Val(129)) == (3, UInt64)
        @test num_chunks(Val(200)) == (4, UInt64)
    end

    @testset "Constructors" begin
        @testset "Constructing from small ints" begin
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
            @test bs128b.chunks[1] ≡ 0x00000decafc0ffee
            @test bs128b.chunks[2] ≡ 0x0000000deadbeefe
            @test bs128b == BitString{128}(big"0xdecafc0ffee0000000deadbeefe")
        end
        @testset "Zero" begin
            a = BitString{100}(0)
            @test all(iszero, chunks(a))
            @test zero(a) == a

            b = BitString{250}(0x0)
            @test all(iszero, chunks(b))
            @test zero(b) == b

            c = BitString{16}(big"0")
            @test all(iszero, chunks(c))
            @test zero(typeof(c)) == c
        end
        @testset "no ghost bits" begin
            @test !has_ghost_bits(BitString{5}(0b11111))
            @test !has_ghost_bits(BitString{5}(0b111111))
            @test !has_ghost_bits(BitString{100}(rand(UInt128)))
            @test !has_ghost_bits(BitString{129}(big"0xffffffffffffffffffffffffffffffffffff"))
        end
    end
    @testset "Counting operations" begin
        # Create rand integer with n ones
        function rand_int_n_ones(T, N)
            result = zero(T)
            while count_ones(result) < N
                result |= 1 << rand(1:120)
            end
            return result
        end
        for N in (1, 2, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120)
            i = rand_int_n_ones(UInt128, N)
            j = i & (UInt128(1) << 120 - 1)
            s = BitString{120}(i)
            @test count_ones(s) == count_ones(j)
            @test count_zeros(s) == count_zeros(j) - 8
            @test leading_ones(s) == leading_ones(j << 8)
            @test trailing_ones(s) == trailing_ones(j)
            @test leading_zeros(s) == min(leading_zeros(j << 8), 120)
            @test trailing_zeros(s) == min(trailing_zeros(j), 120)
        end
        for N in (1, 2, 4, 10, 20, 30, 40, 50, 60)
            i = rand_int_n_ones(UInt64, N)
            j = i & (1 << 40 - 1)
            s = BitString{40}(i)
            @test count_ones(s) == count_ones(j)
            @test count_zeros(s) == count_zeros(j) - 24
            @test leading_ones(s) == leading_ones(j << 24)
            @test trailing_ones(s) == trailing_ones(j)
            @test leading_zeros(s) == min(leading_zeros(j << 24), 40)
            @test trailing_zeros(s) == min(trailing_zeros(j), 40)
        end
    end
    @testset "Bitwise operations" begin
        function rand_bitstring(B)
            N, T = num_chunks(Val(B))
            s = rand(SVector{N,T})
            return remove_ghost_bits(BitString{B,N,T}(s))
        end

        for B in (11, 32, 65, 120, 200, 256)
            a = rand_bitstring(B)
            b = rand_bitstring(B)
            one = ~zero(a)

            @test iszero(a & ~a)
            @test a | ~a == ~zero(a)
            @test a ⊻ a == zero(a)
            @test (a << 7) >> 7 == a & ~zero(BitString{B}) >> 7
            @test (a >> 9) << 9 == a & ~zero(BitString{B}) << 9
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

            # This test checks that the result of show can be pasted into the REPL
            @test eval(Meta.parse(repr(a))) == a
        end
    end
end

using Rimu.Hamiltonians: num_occupied_modes, bose_hubbard_interaction, hopnextneighbour

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
    @testset "num_occupied_modes" begin
        @test num_occupied_modes(middle_full) == 2
        @test num_occupied_modes(middle_empty) == 2
        @test num_occupied_modes(two_full) == 1
        @test num_occupied_modes(BoseFS((0, 0, 0, 0))) == 0
    end
    @testset "bose_hubbard_interaction" begin
        @test bose_hubbard_interaction(middle_full) == 66 * 65
        @test bose_hubbard_interaction(middle_empty) == 8 * 7 + 2
        @test bose_hubbard_interaction(two_full) == 136 * 135
    end
    @testset "occupied_modes" begin
        (bosons, orbital, bit), st = iterate(occupied_modes(middle_full))
        @test bosons == 1
        @test orbital == 2
        @test bit == 1
        (bosons, orbital, bit), st = iterate(occupied_modes(middle_full), st)
        @test bosons == 66
        @test orbital == 63
        @test bit == 63
        @test isnothing(iterate(occupied_modes(middle_full), st))

        @test collect(occupied_modes(middle_empty)) == [[2, 1, 0], [8, 127, 128]]
    end
    @testset "Randomized tests" begin
        # Note: the random number for these tests will be the same everytime. This is still
        # an ok way to look for errors.
        function rand_onr_bose(N, M)
            result = zeros(MVector{M,Int})
            for _ in 1:N
                result[rand(1:M)] += 1
            end
            return SVector(result)
        end
        # Should be exactly the same as onr, but slower.
        function onr2(bose::BoseFS{N,M}) where {N,M}
            result = zeros(MVector{M,Int32})
            for (n, i, _) in occupied_modes(bose)
                @assert n ≠ 0
                result[i] = n
            end
            return SVector(result)
        end
        # Should be exactly the same as hopnextneighbour, but slower.
        function hopnextneighbour2(bose::BoseFS{N,M}, chosen) where {N,M}
            o = MVector{M,Int32}(onr(bose))
            site = (chosen + 1) ÷ 2
            curr = 0
            i = 1
            while i ≤ M
                curr += o[i] > 0
                curr == site && break
                i += 1
            end
            if isodd(chosen)
                j = mod1(i + 1, M)
            else
                j = mod1(i - 1, M)
            end
            o[i] -= 1
            o[j] += 1
            return BoseFS{N}(SVector(o)), (o[i] + 1) * (o[j])
        end
        for (N, M) in ((16, 16), (10, 32), (64, 32), (200, 200), (200, 20), (20, 200))
            @testset "$N, $M" begin
                for _ in 1:10
                    input = rand_onr_bose(N, M)
                    bose = BoseFS(input)
                    @test num_particles(bose) == N
                    @test num_modes(bose) == M
                    @test onr(bose) == input
                    @test num_occupied_modes(bose) == count(!iszero, input)
                    @test bose_hubbard_interaction(bose) == sum(input .* (input .- 1))

                    @test onr2(bose) == input

                    @test all(
                        hopnextneighbour2(bose, i) == hopnextneighbour(bose, i)
                        for i in 1:num_occupied_modes(bose) * 2
                    )

                    occupied = [
                        find_occupied_mode(bose, i).mode
                        for i in 1:num_occupied_modes(bose)
                    ]
                    @test occupied == findall(!iszero, onr(bose))

                    check_single_excitations(bose, 64)
                    check_double_excitations(bose, 8)
                    check_triple_excitations(bose, 4)

                    # This test checks that the result of show can be pasted into the REPL
                    @test eval(Meta.parse(repr(bose))) == bose
                end
            end
        end
    end
end

@testset "FermiFS" begin
    small = SVector(1, 0, 0, 0, 0, 1, 1, 1, 0, 0)
    big = [rand(0:1) for _ in 1:70]
    giant = [rand() < 0.1 for _ in 1:200]
    giant_sparse = [1; 1; zeros(Int, 100); 1]
    giant_dense = [1; 0; ones(Int, 50); 0; 1; 1]
    @testset "onr, constructors" begin
        for o in (small, big, giant, giant_sparse, giant_dense)
            f = FermiFS(o)
            @test onr(f) == o
            @test typeof(f)(onr(f)) == f
        end

        @test_throws ErrorException FermiFS((1, 2, 3))
        @test_throws ErrorException FermiFS{2,3}((1, 1, 1))
    end
    @testset "occupied_modes" begin
        for o in (small, big, giant)
            f = FermiFS(o)
            sites = collect(occupied_modes(f))
            @test getproperty.(sites, :mode) == findall(≠(0), onr(f))
        end
    end
    @testset "Randomized Tests" begin
        function rand_onr_fermi(N, M)
            return SVector{M}(shuffle([ones(Int,N); zeros(Int,M - N)]))
        end
        for (N, M) in ((15, 16), (10, 29), (32, 60), (180, 200), (10, 200), (1, 20))
            @testset "$N, $M" begin
                for _ in 1:10
                    input = rand_onr_fermi(N, M)
                    fermi = FermiFS(input)

                    @test num_particles(fermi) == N
                    @test num_modes(fermi) == M
                    @test onr(fermi) == input

                    check_single_excitations(fermi, 64)
                    check_double_excitations(fermi, 8)
                    check_triple_excitations(fermi, 4)

                    f = FermiFS(input)
                    sites = Int[]
                    foreach(occupied_modes(f)) do i
                        push!(sites, i.mode)
                    end
                    @test sites == findall(≠(0), onr(f))
                end
            end
        end
    end
end

@testset "Multicomponent" begin
    @testset "CompositeFS" begin
        fs1 = CompositeFS(
            FermiFS((1,1,0,0,0,0)), FermiFS((1,1,1,0,0,0)), BoseFS((5,0,0,0,0,0))
        )
        @test num_modes(fs1) == 6
        @test num_components(fs1) == 3
        @test num_particles(fs1) == 10
        @test fs1.components[1] == FermiFS((1,1,0,0,0,0))
        @test fs1.components[2] == FermiFS((1,1,1,0,0,0))
        @test fs1.components[3] == BoseFS((5,0,0,0,0,0))
        @test eval(Meta.parse(repr(fs1))) == fs1

        fs2 = CompositeFS(
            FermiFS((0,1,1,0,0,0)), FermiFS((1,0,1,0,1,0)), BoseFS((1,1,1,1,1,0))
        )
        @test fs1 < fs2

        @test_throws ErrorException CompositeFS(BoseFS((1,1)), BoseFS((1,1,1)))

        @inferred update_component(fs1, FermiFS((0,0,0,1,1,0)), Val(1))
        @inferred update_component(fs1, FermiFS((0,0,1,1,1,0)), Val(2))
        @inferred update_component(fs1, BoseFS((1,1,1,1,1,0)), Val(3))
        @test_throws MethodError update_component(fs1, BoseFS((1,1,1,1,1,0)), Val(1))
        @test_throws MethodError update_component(fs1, FermiFS((1,1,1,1,1,0)), Val(1))

        fs3 = Rimu.BitStringAddresses.update_component(fs1, FermiFS((0,0,1,1,1,0)), Val(2))
        @test fs3 == CompositeFS(
            FermiFS((1,1,0,0,0,0)), FermiFS((0,0,1,1,1,0)), BoseFS((5,0,0,0,0,0)),
        )
        @test onr(fs3) == ([1,1,0,0,0,0], [0,0,1,1,1,0], [5,0,0,0,0,0])
        @test onr(fs3, LadderBoundaries(2, 3)) == (
            [1 0 0; 1 0 0], [0 1 1; 0 1 0], [5 0 0; 0 0 0]
        )
    end

    @testset "BoseFS2C" begin
        fs1 = BoseFS2C((1,1,1,0,0,0,0), (1,1,1,0,5,0,0))
        @test num_modes(fs1) == 7
        @test num_components(fs1) == 2
        @test num_particles(fs1) == 11
        @test eval(Meta.parse(repr(fs1))) == fs1
        @test onr(fs1) == ([1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 5, 0, 0])

        fs2 = BoseFS2C(BoseFS((0,0,0,0,0,0,3)), BoseFS((0,2,1,0,5,0,0)))
        @test fs1 < fs2

        @test_throws MethodError BoseFS2C(BoseFS((1,1)), BoseFS((1,1,1)))
    end
end
