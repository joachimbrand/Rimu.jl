"""
    AxialAngularMomentumHO(S; z_dim = 3) <: AbstractHamiltonian

Angular momentum operator for application to Cartesian harmonic oscillator basis,
see [`HOCartesianEnergyConserved`](@ref) or [`HOCartesianEnergyConservedPerDim`](@ref).
Represents the projection of angular momentum onto `z`-axis:
```math
\\hat{L}_z = i \\hbar \\sum_{j=1}^N \\left( b_x b_y^\\dag - b_y b_x^\\dag \\right),
```
where ``b_x^\\dag`` and ``b_x`` are raising and lowering (ladder) operators 
for  a harmonic oscillator in the ``x`` dimension, and simlarly for ``y``. 

This is implemented for an ``N`` particle Fock space with creation and annihilation 
operators as
```math
\\frac{1}{\\hbar} \\hat{L}_z = i \\sum_{n_x=1}^{M_x} \\sum_{n_y=1}^{M_y} 
    \\left( a_{n_x-1,n_y+1}^\\dag - a_{n_x+1,n_y-1}^\\dag \\right) a_{n_x, n_y}.
```
in units of ``\\hbar``.

Argument `S` is a tuple defining the range of Cartesian modes in each dimension and
their mapping to Fock space modes in a `SingleComponentFockAddress`. If `S` indicates 
a 3D system the `z` dimension can be changed by setting `z_dim`; 
`S` should be be isotropic in the remaining `x`-`y` plane, i.e. must have 
`S[x_dim] == S[y_dim]`.
The starting address `addr` only needs to satisfy `num_modes(addr) == prod(S)`.

# Example
Calculate the overlap of two Fock addresses interpreted as harmonic oscillator states
in a 2D Cartesian basis

```jldoctest
julia> S = (2,2)
(2, 2)

julia> Lz = AxialAngularMomentumHO(BoseFS(prod(S)), S)
AxialAngularMomentumHO((2, 2); z_dim = 3)

julia> v = DVec(BoseFS(prod(S), 2 => 1) => 1.0)
DVec{BoseFS{1, 4, BitString{4, 1, UInt8}},Float64} with 1 entry, style = IsDeterministic{Float64}()
  fs"|0 1 0 0⟩" => 1.0

julia> w = DVec(BoseFS(prod(S), 3 => 1) => 1.0)
DVec{BoseFS{1, 4, BitString{4, 1, UInt8}},Float64} with 1 entry, style = IsDeterministic{Float64}()
  fs"|0 0 1 0⟩" => 1.0

julia> dot(w, Lz, v)
0.0 + 1.0im
```
"""
struct AxialAngularMomentumHO{A,D} <: AbstractHamiltonian{ComplexF64}
    addr::A
    S::NTuple{D,Int64}
    xyz::NTuple{3,Int64}
end

function AxialAngularMomentumHO(addr, S; z_dim = 3)
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
    prod(S) == num_modes(addr) || throw(ArgumentError("address type mismatch"))
    return AxialAngularMomentumHO{typeof(addr),D}(addr, S, (x_dim, y_dim, z_dim))
end

function Base.show(io::IO, L::AxialAngularMomentumHO)
    print(io, "AxialAngularMomentumHO($(L.addr), $(L.S); z_dim = $(L.xyz[3]))")
end

LOStructure(::Type{<:AxialAngularMomentumHO}) = IsHermitian()

starting_address(L::AxialAngularMomentumHO) = L.addr

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
