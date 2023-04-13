"""
    AMzProjectionHO(S; z_dim = 3) <: AbstractHamiltonian

Implement angular momentum operators for application to Cartesian harmonic oscillator basis,
see [`HOCartesian`](@ref) and [`HOCartesianSeparable`](@ref).
Projection of angular momentum onto `z`-axis:
    ``\\hat{L}_z = i \\hbar \\left( a_x a_y^\\dag - a_y a_x^\\dag \\right)``

Returns ``\\hat{L}_z/ i \\hbar``. Argument `S` is a tuple defining the range of Cartesian 
modes in each dimension. 
If `S` indicates a 3D system the `z` dimension can be changed by setting `z_dim`; `S` should be 
be isotropic in the remaining `x`-`y` plane, i.e. must have `S[x_dim]==S[y_dim]`.
"""
struct AMzProjectionHO{D} <: AbstractHamiltonian{Float64}
    S::NTuple{D,Int64}
    xyz::NTuple{3,Int64}
end

function AMzProjectionHO(S; z_dim = 3)
    D = length(S)
    D < 2 && throw(ArgumentError("number of dimensions should be at least 2"))
    if D == 3 && z_dim ≠ 3
        if z_dim == 1
            x_dim, y_dim = (2,3)
        elseif z_dim == 2
            x_dim, y_dim = (1,3)
        else
            throw(ArgumentError("invalid choice of z dimension"))
        end
    else
        x_dim, y_dim = (1, 2)
    end
    S[x_dim] == S[y_dim] || throw(ArgumentError("angular momentum only defined for isotropic system"))
    return AMzProjectionHO{D}(S, (x_dim, y_dim, z_dim))
end

function Base.show(io::IO, ::AMzProjectionHO{D}) where {D}
    print(io, "AMzProjectionHO($D)")
end

LOStructure(::Type{<:AMzProjectionHO}) = IsDiagonal()

diagonal_element(L::AMzProjectionHO, addr::SingleComponentFockAddress) = 0.0

num_offdiagonals(::AMzProjectionHO, addr::SingleComponentFockAddress) = 2 * num_occupied_modes(addr)


function get_offdiagonal(L::AMzProjectionHO{D}, addr::SingleComponentFockAddress, chosen::Int; debug=false) where {D}
    S = L.S
    states = CartesianIndices(S)
    omm = OccupiedModeMap(addr)
    x, y, z = L.xyz

    # mode selects current mode, b = 0,1 selects left or right branch
    chosen_mode, b = divrem(chosen + 1, 2)
    mode_i = omm[chosen_mode].mode

    # Cartesian basis indices
    n_i = Tuple(states[mode_i])
    
    # only two indices change
    n_k = @. n_i[[x,y]] + (2b - 1) * (1, -1)

    # check bounds
    n_k[1] in 1:S[x] && n_k[2] in 1:S[y] || return addr, 0.0 #(0.0, 0.0, omm[mode].occnum)
    
    # prefactor based on Cartesian indices - this should use 0-indexing
    p = (1 - 2b) * sqrt((n_i[x] - 1 + b) * (n_i[y] - b))

    # n_k_rest = D == 2 ? n_k : (n_k..., n_i[3:end]...)
    if D == 2
        n_k_rest = n_k
    else
        n_k_rest = Vector{Int}(undef, 3)
        n_k_rest[x] = n_k[1]
        n_k_rest[y] = n_k[2]
        n_k_rest[z] = n_i[z]
    end
    mode_k = LinearIndices(states)[n_k_rest...]
    new = find_mode(addr, mode_k)

    new_add, val = excitation(addr, (new,), (omm[chosen_mode],))
    debug && println((chosen_mode, b, n_i, n_k, p, val))
    # return new_add, (p, val, omm[mode].occnum)
    return new_add, p * val
end

# not needed:
# starting_address(::AMzProjectionHO)