"""
    get_all_blocks(h::Union{HOCartesian{D},HOCartesianSeparable{D}}; 
        target_energy = nothing, 
        max_energy = nothing, 
        max_blocks = nothing, 
        method = :vertices,
        kwargs...) -> df

Find all distinct blocks of `h`. Returns a `DataFrame`. 

Keyword arguments:
* `target_energy`: only blocks with this noninteracting energy are found
* `max_energy`: only blocks with noninteracting energy less than this are found
* `max_blocks`: exit after finding this many blocks
* `method`: Choose between `:vertices` and `:comb` for method of enumerating tuples of quantum numbers
* `save_to_file=nothing`: if set then the `DataFrame` recording blocks is saved after each new block is found
* additional `kwargs`: passed to `isapprox` for comparing block energies. Useful for anisotropic traps
"""
function get_all_blocks(h::Union{HOCartesian{D},HOCartesianSeparable{D}}; 
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
        @warn "target energy higher than single particle basis size; not all blocks may be found."
    end
    if !isnothing(max_energy) && max_energy < E0
        @warn "maximum requested energy lower than groundstate, not all blocks may be found."
    end
    if !isnothing(max_energy) && !isnothing(target_energy) && max_energy < target_energy
        @warn "maximum requested energy lower than target energy, not all blocks may be found."
    end

    if method == :vertices
        df = get_all_blocks_vertices(h; target_energy, max_energy, max_blocks, kwargs...)
    elseif method == :comb
        df = get_all_blocks_comb(h; target_energy, max_energy, max_blocks, kwargs...)
    else
        @error "invalid method."
    end

    # consistency check
    if isnothing(max_blocks) && isnothing(target_energy) && isnothing(max_energy) && sum(df[!,:block_size]) ≠ dimension(h)
        @warn "not all blocks were found"
    end
    return df
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
        block_basis = build_basis(h, addr)
        for b in block_basis
            push!(known_basis, b)
        end
        push!(df, (; block_id, block_E0, block_size = length(block_basis), addr, indices = t))
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
        block_basis = build_basis(h, addr)
        for b in block_basis
            push!(known_basis, b)
        end
        push!(df, (; block_id, block_E0, block_size = length(block_basis), addr, indices = tuple(t...)))
        !isnothing(save_to_file) && save_df(save_to_file, df)
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    return df
end

"""
    fock_to_cart(addr, S; zero_index = true)
    fock_to_cart(basis, S; zero_index = true)

Convert a Fock state address `addr` or `Vector` of addresses `basis` to Cartesian 
harmonic oscillator basis indices ``n_x,n_y,\\ldots``. These indices are bounded 
by `S` which is a tuple of the maximum number of states in each dimension. By default
the groundstate in each dimension is indexed by `0`, but this can be changed by setting 
`zero_index = false`.
"""
function fock_to_cart(addr::SingleComponentFockAddress, S; zero_index = true)
    prod(S) == num_modes(addr) || throw("Specified cartesian states are incompatible with address")
    states = CartesianIndices(S)

    cart = vcat(map(
        p -> [Tuple(states[p.mode]) .- Int(zero_index) for _ in 1:p.occnum], 
        OccupiedModeMap(addr))...)

    return Tuple(cart)
end
fock_to_cart(basis::Vector{<:SingleComponentFockAddress}, S; zero_index = true) = map(addr -> fock_to_cart(addr, S), basis; zero_index)

"""
    occupied_modes_list(addr)    
    occupied_modes_list(basis)

Output a `Tuple` of all occupied modes in `addr`. Can be applied to a vector 
of addresses. Multiply occupied modes are listed as duplicates so the length 
of the output is always the number of particles. This is a compact form of 
[`onr`](@ref).
"""
@inline function occupied_modes_list(a::SingleComponentFockAddress{N}) where {N}
    oml = [p.mode for p in OccupiedModeMap(a) for _ in 1:p.occnum]
    return Tuple(oml)
end
occupied_modes_list(basis::Vector{<:SingleComponentFockAddress}) = map(addr -> occupied_modes_list(addr), basis)
