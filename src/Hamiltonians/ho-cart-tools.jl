"""
    get_all_blocks(h::Union{HOCartesianEnergyConserved{D},HOCartesianEnergyConservedPerDim{D}}; 
        target_energy = nothing, 
        max_energy = nothing, 
        max_blocks = nothing, 
        method = :vertices,
        kwargs...) -> df

Find all distinct blocks of `h`. Returns a `DataFrame` with columns 
* `block_id`: index of block in order found
* `block_E0`: noninteracting energy of all elements in the block
* `block_size`: number of elements in the block
* `addr`: first address that generates the block with e.g. [`build_basis`](@ref)
* `indices`: tuple of mode indices that allow recreation of the generating address 
    `addr`; in this case use e.g. `BoseFS(M; indices .=> 1)` This is useful when 
    the `DataFrame` is loaded from file since `Arrow.jl` converts custom
    types to `NamedTuple`s.
* `t_basis`: time to generate the basis for each block

Keyword arguments:
* `target_energy`: only blocks with this noninteracting energy are found
* `max_energy`: only blocks with noninteracting energy less than this are found
* `max_blocks`: exit after finding this many blocks
* `method`: Choose between `:vertices` and `:comb` for method of enumerating tuples of quantum numbers
* `save_to_file=nothing`: if set then the `DataFrame` recording blocks is saved after each new block is found
* additional `kwargs`: passed to `isapprox` for comparing block energies. Useful for anisotropic traps

If using option `block_by_level = false` in [`HOCartesianEnergyConserved`](@ref), it is better to use
[`parity_block_seed_addresses`](@ref) instead.
"""
function get_all_blocks(h::Union{HOCartesianEnergyConserved{D},HOCartesianEnergyConservedPerDim{D}}; 
        target_energy = nothing, 
        max_energy = nothing, 
        max_blocks = nothing, 
        method = :vertices,
        kwargs...
    ) where {D}

    add0 = starting_address(h)
    N = num_particles(add0)
    E0 = N * sum(h.aspect1) / 2  # starting address may not be ground state
    if !isnothing(target_energy) && target_energy - E0 > minimum(h.S .* h.aspect1) - 1
        @warn "target energy higher than single particle basis size, not all blocks may be found."
    end
    if !isnothing(max_energy) && max_energy < E0
        @warn "maximum requested energy lower than groundstate, not all blocks may be found."
    end
    if !isnothing(max_energy) && !isnothing(target_energy) && max_energy < target_energy
        @warn "maximum requested energy lower than target energy, not all blocks may be found."
    end

    if h isa HOCartesianEnergyConserved
        M = h.S[1] - 1
        if (isnothing(max_energy) && isnothing(target_energy)) ||
            (!isnothing(max_energy) && max_energy > E0 + M) ||
            (!isnothing(target_energy) && target_energy > E0 + M)
            throw(ArgumentError("requested energy range not accessible by given Hamiltonian"))
        end
    end

    if method == :vertices
        return get_all_blocks_vertices(h; target_energy, max_energy, max_blocks, kwargs...)
    elseif method == :comb
        return get_all_blocks_comb(h; target_energy, max_energy, max_blocks, kwargs...)
    else
        throw(ArgumentError("invalid method."))
    end
end

function get_all_blocks_vertices(h; 
        target_energy = nothing, 
        max_energy = nothing, 
        max_blocks = nothing, 
        save_to_file = nothing,
        kwargs...
    )
    add0 = starting_address(h)
    N = num_particles(add0)
    P = prod(h.S)

    # initialise
    df = DataFrame()
    block_id = 0
    known_basis = Set{typeof(add0)}()
    L = _binomial(P + N - 1, Val(N))
    idx_correction = reverse(ntuple(i -> i - 1, Val(N)))
    for i in 1:L
        t = vertices(i, Val(N)) .- idx_correction
        # check target energy
        block_E0 = noninteracting_energy(h, t)
        if !isnothing(target_energy)
            !isapprox(block_E0, target_energy; kwargs...) && continue
        end
        if !isnothing(max_energy)
            block_E0 > max_energy && continue
        end
        # check if known
        addr = BoseFS(P, t .=> ones(Int, N))
        if addr in known_basis
            continue
        end

        # new block found
        block_id += 1
        t_basis = @elapsed block_basis = build_basis(h, addr)
        for b in block_basis
            push!(known_basis, b)
        end
        push!(df, (; block_id, block_E0, block_size = length(block_basis), addr, indices = t, t_basis))
        !isnothing(save_to_file) && save_df(save_to_file, df)
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    return df
end

# old version - issues with GC due to allocating many small vectors
function get_all_blocks_comb(h; 
        target_energy = nothing, 
        max_energy = nothing, 
        max_blocks = nothing, 
        save_to_file = nothing,
        kwargs...
    )
    add0 = starting_address(h)
    N = num_particles(add0)
    P = prod(h.S)

    # initialise
    df = DataFrame()
    block_id = 0
    known_basis = Set{typeof(add0)}()
    tuples = with_replacement_combinations(1:P, N)
    for t in tuples
        # check target energy
        block_E0 = noninteracting_energy(h, t)
        if !isnothing(target_energy)
            !isapprox(block_E0, target_energy; kwargs...) && continue
        end
        if !isnothing(max_energy)
            block_E0 > max_energy && continue
        end
        # check if known
        addr = BoseFS(P, t .=> ones(Int, N))
        if addr in known_basis
            continue
        end

        # new block found
        block_id += 1
        t_basis = @elapsed block_basis = build_basis(h, addr)
        for b in block_basis
            push!(known_basis, b)
        end
        push!(df, (; block_id, block_E0, block_size = length(block_basis), addr, indices = tuple(t...), t_basis))
        !isnothing(save_to_file) && save_df(save_to_file, df)
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    return df
end

"""
    fock_to_cart(addr, S; zero_index = true)

Convert a Fock state address `addr` to Cartesian harmonic oscillator basis 
indices ``n_x,n_y,\\ldots``. These indices are bounded by `S` which is a 
tuple of the maximum number of states in each dimension. By default the 
groundstate in each dimension is indexed by `0`, but this can be changed 
by setting `zero_index = false`.
"""
function fock_to_cart(
    addr::SingleComponentFockAddress{N}, 
    S::NTuple{D,Int}; 
    zero_index = true
    ) where {N,D}
    prod(S) == num_modes(addr) || throw(ArgumentError("Specified cartesian states are incompatible with address"))
    states = CartesianIndices(S)

    cart = vcat(map(
        p -> [Tuple(states[p.mode]) .- Int(zero_index) for _ in 1:p.occnum], 
        OccupiedModeMap(addr))...)

    return SVector{N,NTuple{D,Int}}(cart)
end

"""
    parity_block_seed_addresses(H::HOCartesianEnergyConserved{D})

Get a vector of addresses that each have different parity with respect to 
the trap geometry defined by the Hamiltonian `H`. The result will have `2^D`
addresses for a `D`-dimensional trap.

This is designed for [`HOCartesianEnergyConserved`](@ref) with option
`block_by_level = false`. In this case it is better to use [`BasisSetRep`](@ref) 
with these seed addresses (and a filter) rather than [`get_all_blocks`](@ref).
"""
function parity_block_seed_addresses(H::HOCartesianEnergyConserved{D,A}) where {D,A}
    P = prod(H.S)
    N = num_particles(H.addr)
    breakpoints = accumulate(*, (1, H.S[1:end-1]...))
    seeds = A[]
    for c in with_replacement_combinations([0,1], D), p in multiset_permutations(c, D)
        index = 1 + dot(p, breakpoints)
        push!(seeds,
            BoseFS(P, index => 1, 1 => N-1)
        )
    end
    
    return seeds
end