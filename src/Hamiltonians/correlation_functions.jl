using StaticArrays
# TO-DO: add geometry for higher dimensions.
"""
    G2RealCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Two-body operator for density-density correlation between sites separated by `d`
with `0 ≤ d < M`.
```math
    \\hat{G}^{(2)}(d) = \\frac{1}{M} \\sum_i^M \\hat{n}_i (\\hat{n}_{i+d} - \\delta_{0d}).
```
Assumes a one-dimensional lattice with periodic boundary conditions where
```math
    \\hat{G}^{(2)}(-M/2 \\leq d < 0) = \\hat{G}^{(2)}(|d|),
```
```math
    \\hat{G}^{(2)}(M/2 < d < M) = \\hat{G}^{(2)}(M - d),
```
and normalisation
```math
    \\sum_{d=0}^{M-1} \\langle \\hat{G}^{(2)}(d) \\rangle = \\frac{N (N-1)}{M}.
```

For multicomponent basis, calculates correlations between all particles equally,
equivalent to stacking all components into a single Fock state.

# Arguments
- `d::Integer`: distance between sites.

# See also

* [`HubbardReal1D`](@ref)
* [`G2MomCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2RealCorrelator{D} <: AbstractHamiltonian{Float64}
end

G2RealCorrelator(d::Int) = G2RealCorrelator{d}()

function Base.show(io::IO, ::G2RealCorrelator{D}) where {D}
    print(io, "G2RealCorrelator($D)")
end

LOStructure(::Type{<:G2RealCorrelator}) = IsDiagonal()

function diagonal_element(::G2RealCorrelator{0}, add::SingleComponentFockAddress)
    M = num_modes(add)
    v = onr(add)
    return dot(v, v .- 1) / M
end
function diagonal_element(::G2RealCorrelator{D}, add::SingleComponentFockAddress) where {D}
    M = num_modes(add)
    d = mod(D, M)
    v = onr(add)
    result = 0
    for i in eachindex(v)
        result += v[i] * v[mod1(i + d, M)]
    end
    return result / M
end

function diagonal_element(::G2RealCorrelator{0}, add::CompositeFS)
    M = num_modes(add)
    v = sum(map(onr, add.components))
    return dot(v, v .- 1) / M
end
function diagonal_element(::G2RealCorrelator{D}, add::CompositeFS) where {D}
    M = num_modes(add)
    d = mod(D, M)
    v = sum(map(onr, add.components))
    result = 0
    for i in eachindex(v)
        result += v[i] * v[mod1(i + d, M)]
    end
    return result / M
end

num_offdiagonals(::G2RealCorrelator, ::SingleComponentFockAddress) = 0
num_offdiagonals(::G2RealCorrelator, ::CompositeFS) = 0

# not needed:
# get_offdiagonal(::G2RealCorrelator, add)
# starting_address(::G2RealCorrelator)

# Methods that need to be implemented:
#   •  starting_address(::AbstractHamiltonian) - not needed
"""
    G2MomCorrelator(d::Int,c=:cross) <: AbstractHamiltonian{ComplexF64}

Two-body correlation operator representing the density-density
correlation at distance `d` of a two component system
in a momentum-space Fock-state basis.
It returns a `Complex` value.

Correlation across two components:
```math
\\hat{G}^{(2)}(d) = \\frac{1}{M}\\sum_{spqr=1}^M e^{-id(p-q)2π/M} a^†_{s} b^†_{p}  b_q a_r δ_{s+p,q+r}
```
Correlation within a single component:
```math
\\hat{G}^{(2)}(d) = \\frac{1}{M}\\sum_{spqr=1}^M e^{-id(p-q)2π/M} a^†_{s} a^†_{p}  a_q a_r δ_{s+p,q+r}
```

The diagonal element, where `(p-q)=0`, is
```math
\\frac{1}{M}\\sum_{k,p=1}^M a^†_{k} b^†_{p}  b_p a_k .
```

# Arguments
- `d::Integer`: the distance between two particles.
- `c`: possible instructions: `:cross`: default instruction, computing correlation between
  particles across two components;
  `:first`: computing correlation between particles within the first component;
  `:second`: computing correlation between particles within the second component.
  These are the only defined instructions, using anything else will produce errors.

# To use on a one-component system

For a system with only one component, e.g. with `BoseFS`, the second argument `c` is
irrelevant and can be any of the above instructions, one could simply skip this argument
and let it be the default value.

# See also

* [`BoseHubbardMom1D2C`](@ref)
* [`BoseFS2C`](@ref)
* [`G2RealCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2MomCorrelator{C} <: AbstractHamiltonian{ComplexF64}
    d::Int

    function G2MomCorrelator(d,c=:cross)
        if c == :first
            return new{1}(d)
        elseif c == :second
            return new{2}(d)
        elseif c == :cross
            return new{3}(d)
        else
            throw(ArgumentError("Unknown instruction for G2MomCorrelator!"))
        end
    end
end
@deprecate G2Correlator(d,c) G2MomCorrelator(d,c)
@deprecate G2Correlator(d) G2MomCorrelator(d)

function Base.show(io::IO, g::G2MomCorrelator{C}) where C
    if C == 1
        print(io, "G2MomCorrelator($(g.d),:first)")
    elseif C == 2
        print(io, "G2MomCorrelator($(g.d),:second)")
    elseif C == 3
        print(io, "G2MomCorrelator($(g.d),:cross)")
    end
end


num_offdiagonals(g::G2MomCorrelator{1},add::BoseFS2C) = num_offdiagonals(g, add.bsa)
num_offdiagonals(g::G2MomCorrelator{2},add::BoseFS2C) = num_offdiagonals(g, add.bsb)

function num_offdiagonals(g::G2MomCorrelator{3}, add::BoseFS2C)
    m = num_modes(add)
    sa = num_occupied_modes(add.bsa)
    sb = num_occupied_modes(add.bsb)
    return sa*(m-1)*sb
    # number of excitations that can be made
end

function num_offdiagonals(g::G2MomCorrelator, add::SingleComponentFockAddress)
    m = num_modes(add)
    singlies, doublies = num_singly_doubly_occupied_sites(add)
    return singlies*(singlies-1)*(m - 2) + doublies*(m - 1)
end

diagonal_element(g::G2MomCorrelator{1}, add::BoseFS2C) = diagonal_element(g, add.bsa)
diagonal_element(g::G2MomCorrelator{2}, add::BoseFS2C) = diagonal_element(g, add.bsb)

function diagonal_element(g::G2MomCorrelator{3}, add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    onrep_a = onr(add.bsa)
    onrep_b = onr(add.bsb)
    gd = 0
    for p in 1:M
        iszero(onrep_b[p]) && continue
        for k in 1:M
            gd += onrep_a[k]*onrep_b[p] # b†_p b_p a†_k a_k
        end
    end
    return ComplexF64(gd/M)
end

function diagonal_element(g::G2MomCorrelator, add::SingleComponentFockAddress)
    M = num_modes(add)
    onrep = onr(add)
    gd = 0
    for p in 1:M
        iszero(onrep[p]) && continue
        for k in 1:M
            gd += onrep[k]*onrep[p] # a†_p a_p a†_k a_k
        end
    end
    return ComplexF64(gd/M)
end

function get_offdiagonal(g::G2MomCorrelator{1}, add::A, chosen)::Tuple{A,ComplexF64} where A<:BoseFS2C
    new_bsa, elem = get_offdiagonal(g, add.bsa, chosen)
    return A(new_bsa,add.bsb), elem
end

function get_offdiagonal(g::G2MomCorrelator{2}, add::A, chosen)::Tuple{A,ComplexF64} where A<:BoseFS2C
    new_bsb, elem = get_offdiagonal(g, add.bsb, chosen)
    return A(add.bsa, new_bsb), elem
end

function get_offdiagonal(
    g::G2MomCorrelator{3},
    add::A,
    chosen,
    ma=OccupiedModeMap(add.bsa),
    mb=OccupiedModeMap(add.bsb),
)::Tuple{A,ComplexF64} where {A<:BoseFS2C}

    m = num_modes(add)
    new_bsa, new_bsb, gamma, _, _, Δp = momentum_transfer_excitation(
        add.bsa, add.bsb, chosen, ma, mb
    )
    gd = exp(-im*g.d*Δp*2π/m)*gamma
    return A(new_bsa, new_bsb), ComplexF64(gd/m)
end

function get_offdiagonal(
    g::G2MomCorrelator,
    add::A,
    chosen,
)::Tuple{A,ComplexF64} where {A<:SingleComponentFockAddress}
    M = num_modes(add)
    new_add, gamma, Δp = momentum_transfer_excitation(add, chosen, OccupiedModeMap(add))
    gd = exp(-im*g.d*Δp*2π/M)*gamma
    return new_add, ComplexF64(gd/M)
end

"""
    SuperfluidCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Operator for extracting superfluid correlation between sites separated by a distance `d` with `0 ≤ d < M`:

```math
    \\hat{C}_{\\text{SF}}(d) = \\frac{1}{M} \\sum_{i}^{M} a_{i}^{\\dagger} a_{i + d}
```
Assumes a one-dimensional lattice with ``M`` sites and periodic boundary conditions. ``M`` is also the number of modes in the Fock state address.

# Usage
Superfluid correlations can be extracted from a Monte Carlo calculation by wrapping `SuperfluidCorrelator` with
[`AllOverlaps`](@ref) and passing into [`lomc!`](@ref) with the `replica` keyword argument. For an example with a
similar use of [`G2RealCorrelator`](@ref) see
[G2 Correlator Example](https://joachimbrand.github.io/Rimu.jl/previews/PR227/generated/G2-example.html).


See also [`HubbardReal1D`](@ref), [`G2RealCorrelator`](@ref), [`AbstractHamiltonian`](@ref),
and [`AllOverlaps`](@ref).
"""
struct SuperfluidCorrelator{D} <: AbstractHamiltonian{Float64}
end

SuperfluidCorrelator(d::Int) = SuperfluidCorrelator{d}()

function Base.show(io::IO, ::SuperfluidCorrelator{D}) where {D}
    print(io, "SuperfluidCorrelator($D)")
end

function num_offdiagonals(::SuperfluidCorrelator, add::SingleComponentFockAddress)
    return num_occupied_modes(add)
end

function get_offdiagonal(::SuperfluidCorrelator{D}, add::SingleComponentFockAddress, chosen) where {D}
    src = find_occupied_mode(add, chosen)
    dst = find_mode(add, mod1(src.mode + D, num_modes(add)))
    address, value = excitation(add, (dst,), (src,))
    return address, value / num_modes(add)
end

function diagonal_element(::SuperfluidCorrelator{0}, add::SingleComponentFockAddress)
    return num_particles(add) / num_modes(add)
end
function diagonal_element(::SuperfluidCorrelator{D}, add::SingleComponentFockAddress) where {D}
    return 0.0
end


"""
    StringCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Operator for extracting string correlation between lattice sites on a one-dimensional Hubbard lattice
separated by a distance `d` with `0 ≤ d < M`

```math
    \\hat{C}_{\\text{string}}(d) = \\frac{1}{M} \\sum_{j}^{M} \\delta n_j (e^{i \\pi \\sum_{j \\leq k < j + d} \\delta n_k}) \\delta n_{j+d}
```
Here, ``\\delta \\hat{n}_j = \\hat{n}_j - \\bar{n}`` is the boson number deviation from the mean filling
number and ``\\bar{n} = N/M`` is the mean filling number of lattice sites with ``N`` particles and
``M`` lattice sites (or modes).

Assumes a one-dimensional lattice with periodic boundary conditions. For usage
see [`SuperfluidCorrelator`](@ref) and [`AllOverlaps`](@ref).

See also [`HubbardReal1D`](@ref), [`G2RealCorrelator`](@ref), [`SuperfluidCorrelator`](@ref),
[`AbstractHamiltonian`](@ref), and [`AllOverlaps`](@ref).
"""
struct StringCorrelator{D} <: AbstractHamiltonian{Float64}
end

StringCorrelator(d::Int) = StringCorrelator{d}()

function Base.show(io::IO, ::StringCorrelator{D}) where {D}
    print(io, "StringCorrelator($D)")
end

LOStructure(::Type{<:StringCorrelator}) = IsDiagonal()

function diagonal_element(::StringCorrelator{0}, add::SingleComponentFockAddress)
    M = num_modes(add)
    N = num_particles(add)
    n̄ = N/M
    v = onr(add)

    result = 0.0
    for i in eachindex(v)
        result += (v[i] - n̄)^2
    end

    return result / M
end

num_offdiagonals(::StringCorrelator, ::SingleComponentFockAddress) = 0

function diagonal_element(::StringCorrelator{D}, add::SingleComponentFockAddress) where {D}
    M = num_modes(add)
    N = num_particles(add)
    d = mod(D, M)

    if !ismissing(N) && iszero(N % M)
        return _string_diagonal_real(d, add)
    else
        return _string_diagonal_complex(d, add)
    end
end

function _string_diagonal_complex(d, add)
    M = num_modes(add)
    N = num_particles(add)
    n̄ = N/M
    v = onr(add)

    result = ComplexF64(0)
    for i in eachindex(v)
        phase_sum = sum((v[mod1(k, M)] - n̄) for k in i:1:(i+d-1))

        result += (v[i] - n̄) * exp(pi * im * phase_sum) * (v[mod1(i + d, M)] - n̄)
    end

    return result / M
end
function _string_diagonal_real(d, add)
    M = num_modes(add)
    N = num_particles(add)
    n̄ = N ÷ M
    v = onr(add)

    result = 0.0
    for i in eachindex(v)
        phase_sum = sum((v[mod1(k, M)] - n̄) for k in i:1:(i+d-1))

        result += (v[i] - n̄) * (-1)^phase_sum * (v[mod1(i + d, M)] - n̄)
    end

    return result / M
end

"""
    G2RealSpace(g::Geometry) <: AbstractHamiltonian{SArray}

Two-body operator for density-density correlation for all displacements.

```math
    \\hat{G}^{(2)}(d) = \\frac{1}{M} ∑_i^M∑_v \\hat{n}_i (\\hat{n}_{i+v} - \\delta_{0d}).
```

# See also

* [`HubbardReal1D`](@ref)
* [`G2MomCorrelator`](@ref)
* [`G2RealCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2RealSpace{A,B,G<:Geometry,S} <: AbstractHamiltonian{S}
    geometry::G
    init::S
end
function G2RealSpace(geometry::Geometry, source=1, target=source)
    init = zeros(SArray{Tuple{size(geometry)...}})
    return G2RealSpace{source,target,typeof(geometry),typeof(init)}(geometry, init)
end

LOStructure(::Type{<:G2RealSpace}) = IsDiagonal()

num_offdiagonals(g2::G2RealSpace, _) = 0

#=
function diagonal_element(g2::G2RealSpace{1,1,D}, addr::SingleComponentFockAddress) where {D}
    geo = g2.geometry
    result = g2.init
    v = onr(addr)

    for i in eachindex(result)
        res_i = 0.0
        δ_vec = Offsets(geo)[i]
        for p in eachindex(v)
            p_vec = geo[p]
            q = geo[add(p_vec, δ_vec)]
            if q ≠ 0
                res_i += v[p] * v[q] - (p == q)
            end
        end
        result = setindex(result, res_i, i)
    end
    return result ./ length(geo)
end
=#

function diagonal_element2(
    g2::G2RealSpace{A,B}, addr1::SingleComponentFockAddress, addr2::SingleComponentFockAddress
) where {A,B}
    geo = g2.geometry
    result = g2.init
    if addr1 ≡ addr2
        v1 = v2 = onr(addr1)
    else
        v1 = onr(addr1)
        v2 = onr(addr2)
    end

    for i in eachindex(result)
        res_i = 0.0
        δ_vec = Offsets(geo)[i]
        for p in eachindex(v1)
            p_vec = geo[p]
            q = geo[add(p_vec, δ_vec)]
            if q ≠ 0
                # A = B implies addr1 and addr2 are the same address, in which case we need
                # to subtract δ_{p,q}
                res_i += v1[p] * v2[q] - (A == B) * (p == q)
            end
        end
        result = setindex(result, res_i, i)
    end
    return result ./ length(geo)
end

function diagonal_element(
    g2::G2RealSpace{A,B}, addr1::SingleComponentFockAddress, addr2::SingleComponentFockAddress
) where {A, B}
    geo = g2.geometry
    result = g2.init
    if addr1 ≡ addr2
        v1 = v2 = onr(addr1, geo)
    else
        v1 = onr(geo, addr1)
        v2 = onr(geo, addr2)
    end

    @inbounds for i in eachindex(result)
        res_i = 0.0
        δ_vec = Offsets(geo)[i]

        # Case of n_i(n_i - 1)
        if A == B && all(==(0), δ_vec)
            v2_offset = max.(v2 .- 1, 0)
            result = setindex(result, dot(v1, v2_offset), i)
        else
            #circshift!(v2_offset, v2, δ_vec)
            result = setindex(result, csh(v2, δ_vec), i)
        end
    end
    return result ./ length(geo)
end

function diagonal_element(g2::G2RealSpace{A,A}, addr::SingleComponentFockAddress) where {A}
    return diagonal_element(g2, addr, addr)
end
function diagonal_element(g2::G2RealSpace{A,B}, addr::CompositeFS) where {A,B}
    return diagonal_element(g2, addr.components[A], addr.components[B])
end


# TODO: clean up!
function csh(arr, inds)
    _circshift_dot!((), arr, (), axes(arr), inds)
end


@inline function _circshift_dot!(rdest, src, rsrc,
                             inds::Tuple{AbstractUnitRange,Vararg{Any}},
                             shiftamt::Tuple{Integer,Vararg{Any}})
    ind1, d = inds[1], shiftamt[1]
    s = mod(d, length(ind1))
    sf, sl = first(ind1)+s, last(ind1)-s
    r1, r2 = first(ind1):sf-1, sf:last(ind1)
    r3, r4 = first(ind1):sl, sl+1:last(ind1)
    tinds, tshiftamt = Base.tail(inds), Base.tail(shiftamt)
    _circshift_dot!((rdest..., r1), src, (rsrc..., r4), tinds, tshiftamt) +
    _circshift_dot!((rdest..., r2), src, (rsrc..., r3), tinds, tshiftamt)
end
@inline function _circshift_dot!(rdest, src, rsrc, inds, shiftamt)
    dot(view(src, rdest...), view(src, rsrc...))
    #copyto!(CartesianIndices(rdest), src, CartesianIndices(rsrc))
end
