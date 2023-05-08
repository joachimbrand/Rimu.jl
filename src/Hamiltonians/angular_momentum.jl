"""
    AxialAngularMomentumHO(S; z_dim = 3) <: AbstractHamiltonian

Angular momentum operator for application to Cartesian harmonic oscillator basis,
see [`HOCartesianEnergyConserved`](@ref) or [`HOCartesianEnergyConservedPerDim`](@ref).
Represents the projection of angular momentum onto `z`-axis:
```math
\\hat{L}_z = i \\hbar \\sum_{j=1}^N \\left( b_{j_x} b_{j_y}^\\dag - b_{j_y} b_{j_x}^\\dag \\right),
```
where ``b_{j_x}^\\dag`` and ``b_{j_x}`` are raising and lowering (ladder) operators 
for the ``j^\\text{th}`` particle in a harmonic oscillator in the ``x`` dimension, 
and simlarly for ``y``. Their action on a ``N`` particle Fock state that represents 
a many-body harmonic oscillator is e.g.
```math
\\sum_{j=1}^N b_{j_x} b_{j_y}^\\dag = \\sum_{n_x=1}^{M_x} \\sum_{n_y=1}^{M_y} a_{n_x-1,n_y+1}^\\dag a_{n_x, n_y}
```
where ``a_n^\\dag`` and ``a_n`` are creation and annihilation operators for the 
``n^\\text{th}`` single-particle mode, indexed by a pair ``n = (n_x, n_y)``.

Argument `S` is a tuple defining the range of Cartesian modes in each dimension. 
If `S` indicates a 3D system the `z` dimension can be changed by setting `z_dim`; 
`S` should be be isotropic in the remaining `x`-`y` plane, i.e. must have 
`S[x_dim] == S[y_dim]`.

Note: This operator does not have a meaningful [`starting_address`](@ref), but will work with
[`BasisSetRep`](@ref)`(L, addr)` if an appropriate `addr` is supplied, that is, if
`prod(S) == num_modes(addr)`.
"""
struct AxialAngularMomentumHO{D} <: AbstractHamiltonian{ComplexF64}
    S::NTuple{D,Int64}
    xyz::NTuple{3,Int64}
end

function AxialAngularMomentumHO(S; z_dim = 3)
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
    return AxialAngularMomentumHO{D}(S, (x_dim, y_dim, z_dim))
end

function Base.show(io::IO, L::AxialAngularMomentumHO)
    print(io, "AxialAngularMomentumHO($(L.S); z_dim = $(L.xyz[3]))")
end

LOStructure(::Type{<:AxialAngularMomentumHO}) = IsHermitian()

# need this to work with `BasisSetRep`
starting_address(L::AxialAngularMomentumHO) = BoseFS(prod(L.S))
function check_address_type(L::AxialAngularMomentumHO, addr::A) where A
    prod(L.S) == num_modes(addr) || throw(ArgumentError("address type mismatch"))
end

diagonal_element(L::AxialAngularMomentumHO, addr::SingleComponentFockAddress) = 0.0

num_offdiagonals(::AxialAngularMomentumHO, addr::SingleComponentFockAddress) = 2 * num_occupied_modes(addr)

function get_offdiagonal(L::AxialAngularMomentumHO{D}, addr::SingleComponentFockAddress, chosen::Int) where {D}
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
    
    return new_add, p * val * im
end
