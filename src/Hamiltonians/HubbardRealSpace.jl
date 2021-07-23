"""
    real_space_interaction(::AbstractFockAddress, u)
    real_space_interaction(::AbstractFockAddress, ::AbstractFockAddress, v)

Return the real space interaction between one or two fock states.
"""
real_space_interaction(b::BoseFS, u) = u * bose_hubbard_interaction(b) / 2
real_space_interaction(f::FermiFS, _) = 0
real_space_interaction(f::FermiFS, g::FermiFS, v) = v * count_ones(f.bs & g.bs)
real_space_interaction(f::FermiFS, b::BoseFS, v) = real_space_interaction(b, f, v)

function real_space_interaction(a::BoseFS, b::BoseFS, v)
    occ_a = occupied_orbitals(a)
    occ_b = occupied_orbitals(b)

    (n_a, i_a, _), st_a = iterate(occ_a)
    (n_b, i_b, _), st_b = iterate(occ_b)

    acc = 0
    while true
        if i_a > i_b
            # b is behind and needs to do a step
            iter = iterate(occ_b, st_b)
            isnothing(iter) && return acc * v
            (n_b, i_b, _), st_b = iter
        elseif i_a < i_b
            # a is behind and needs to do a step
            iter = iterate(occ_a, st_a)
            isnothing(iter) && return acc * v
            (n_a, i_a, _), st_a = iter
        else
            # a and b are at the same position
            acc += n_a * n_b
            # now both need to do a step
            iter = iterate(occ_a, st_a)
            isnothing(iter) && return acc * v
            (n_a, i_a, _), st_a = iter
            iter = iterate(occ_b, st_b)
            isnothing(iter) && return acc * v
            (n_b, i_b, _), st_b = iter
        end
    end
end
function real_space_interaction(b::BoseFS, f::FermiFS, v)
    acc = 0
    for (n, i) in occupied_orbitals(b)
        acc += is_occupied(f, i) * n
    end
    return acc * v
end
function real_space_interaction(fs::CompositeFS, m)
    M = num_components(fs)
    acc = 0
    for i in 1:M
        for j in i:M
            acc += real_space_interaction(fs.adds[i], fs.adds[j], m[i, j])
        end
    end
    return acc
end

struct HubbardRealSpace{D,C,A,S}
    address::A
    interactions::SMatrix{C,C,Float32}
    ts::SVector{C,Float32}
end

function HubbardRealSpace(
    address;
    interactions=ones(num_components(address), num_components(address)),
    ts=ones(num_components(address)),
    size=(num_modes(address),)
)
    C = num_components(address)
    prod(size) ≠ num_modes(address) && error("address is not compatible with lattice size")

    return HubbardRealSpace{length(size),C,typeof(address),size}(
        address, SMatrix{C,C,Float32}(interactions), SVector{C,Float32}(ts),
    )
end

starting_address(h::HubbardRealSpace) = h.address
lattice_size(::HubbardRealSpace{<:Any,<:Any,S}) where {S} = S
diagonal_element(h::HubbardRealSpace, address) = real_space_interaction(h, address)

function get_offdiagonal(h::HubbardRealSpace{D,C}, add, chosen) where {D,C}
    sz = lattice_size(h)
    # We have length(sz) dimensions. Moving in the first one is going to ±1, moving in the second is ±sz[1], nth ±prod(sz[1:n-1])
    length(sz)
end
