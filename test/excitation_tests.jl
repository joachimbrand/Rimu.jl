###
### Helper functions for manually testing excitations. Also used during tests for
### BitStringAddresses.
###
using Random
using Rimu
using StaticArrays
using Test

"""
    rand_address(::Type{<:SingleComponentAddress{N,M}}) where {N,M}

Generate a random address with `N` particles in `M` modes.
"""
function rand_address(::Type{BoseFS{N,M}}) where {N,M}
    result = zeros(MVector{M,Int})
    for _ in 1:N
        result[rand(1:M)] += 1
    end
    return BoseFS(result)
end
function rand_address(::Type{FermiFS{N,M}}) where {N,M}
    return FermiFS(shuffle([ones(Int,N); zeros(Int,M - N)]))
end

"""
    excitation_svec(::SingleComponentAddress, creations, destructions)

Dumb and slow, but easier to verify implementation of `excitation`. Takes mode indices as
integers as `creations` and `destructions`.
"""
function excitation_svec(b::BoseFS{N,M}, creations, destructions) where {N,M}
    onrep = MVector(onr(b))
    value = 1

    for d in reverse(destructions)
        value *= onrep[d]
        @inbounds onrep[d] -= 1
    end
    for c in reverse(creations)
        @inbounds onrep[c] += 1
        value *= onrep[c]
    end
    if value == 0
        return b, 0.0
    else
        return BoseFS{N,M}(Tuple(onrep)), √value
    end
end
function excitation_svec(f::FermiFS{N,M}, creations, destructions) where {N,M}
    onrep = MVector(onr(f))
    num = 0

    for d in reverse(destructions)
        num += sum(onrep[1:d - 1])
        if iszero(onrep[d])
            return f, 0.0
        end
        @inbounds onrep[d] -= 1
    end
    for c in reverse(creations)
        num += sum(onrep[1:c - 1])
        if !iszero(onrep[c])
            return f, 0.0
        end
        @inbounds onrep[c] += 1
    end
    return FermiFS{N,M}(Tuple(onrep)), ifelse(iseven(num), 1.0, -1.0)
end

"""
    excitation_direct(::SingleComponentAddress, creations, destructions)

Wrapper of `excitation` that takes mode indices as integers as `creations` and
`destructions`.
"""
function excitation_direct(b::BoseFS, creations, destructions)
    d_indices = find_mode.(Ref(b), destructions)
    c_indices = find_mode.(Ref(b), creations)
    return excitation(b, c_indices, d_indices)
end
function excitation_direct(f::FermiFS, creations, destructions)
    return excitation(f, creations, destructions)
end

"""
    excitations_correct(add, cs, ds)

Return true if `excitations_direct` and `excitations_svec` give the same result. Print an
error message explaining where it failed otherwise.
"""
function excitations_correct(add, cs, ds)
    res_direct = excitation_direct(add, cs, ds)
    res_svec = excitation_svec(add, cs, ds)
    if res_direct ≠ res_svec
        @error "Failed" add cs ds res_direct res_svec
        return false
    else
        return true
    end
end

"""
    move_particle_correct(add, src, dst)

Compare the result of `move_particle` and `excitation_direct` and return true if they give
the same result. Note: this should be called after ensuring `excitation_direct` is correct.
"""
function move_particle_correct(add, src, dst)
    res_direct = excitation_direct(add, (dst,), (src,))
    res_move = move_particle(add, find_mode(add, src), find_mode(add, dst))
    if res_direct[2] == 0 && res_move[2] == 0
        return true
    elseif res_direct ≠ res_move
        @error "Failed" add src dst res_direct res_move
        return false
    else
        return true
    end
end

"""
    rand_subset(M, num)

Get a random subset of the range `1:M` of length `num`.
"""
function rand_subset(M, num)
    if num ≥ M
        return 1:M
    else
        return shuffle(1:M)[1:num]
    end
end

"""
    check_single_excitations(add, num=Inf)

Check whether `excitation_svec` and `excitation_direct` give the same result for single
interactions. If `num` is not given, all possible combinaions of destruction and creation
operators are used, otherwise `num` of each is used.
"""
function check_single_excitations(add, num=Inf)
    @testset "Single excitations for $add" begin
        M = num_modes(add)
        for i in rand_subset(M, num), j in rand_subset(M, num)
            @test excitations_correct(add, (i,), (j,))
            @test move_particle_correct(add, j, i)
        end
    end
end

"""
    check_double_excitations(add, num=Inf)

Check whether `excitation_svec` and `excitation_direct` give the same result for double
interactions. If `num` is not given, all possible combinaions of destruction and creation
operators are used, otherwise `num` of each is used.
"""
function check_double_excitations(add, num=Inf)
    @testset "Double excitations for $add" begin
        M = num_modes(add)
        for i in rand_subset(M, num), j in rand_subset(M, num)
            for k in rand_subset(M, num), l in rand_subset(M, num)
                @test excitations_correct(add, (i, j), (k, l))
            end
        end
    end
end

"""
    check_triple_excitations(add, num=Inf)

Check whether `excitation_svec` and `excitation_direct` give the same result for triple
interactions. If `num` is not given, all possible combinaions of destruction and creation
operators are used, otherwise `num` of each is used.
"""
function check_triple_excitations(add, num=Inf)
    @testset "Triple excitations for $add" begin
        M = num_modes(add)
        for i in rand_subset(M, num), j in rand_subset(M, num)
            for k in rand_subset(M, num), l in rand_subset(M, num)
                for m in rand_subset(M, num), n in rand_subset(M, num)
                    @test excitations_correct(add, (i, j, k), (l, m, n))
                end
            end
        end
    end
end
