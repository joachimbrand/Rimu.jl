"""
    FroehlichPolaron1D{T}(address::BoseFS{missing,M}; kwargs...) <: AbstractHamiltonian{T}

The Froehlich polaron Hamiltonian for a 1D lattice with `M` momentum modes is given by

```math
H = (p̂_f - p)^2/2m + ωN̂ - v Σₖ(âₖ^† + â₋ₖ)
```

where ``p`` is the total momentum, ``p̂_f = Σ_k k âₖ^† âₖ`` is the momentum operator for the
bosons, and ``k`` part of the momentum lattice with separation ``2π/l``. ``N̂`` is the number
operator for the bosons.

Setting the type parameter `T` is optional and will be inferred from the keyword arguments
if not provided. Set `T` to `Float32` for single precision, e.g. when using GPUs.

# Keyword Arguments

* `p = 0.0`: the total momentum ``p``.
* `v = 1.0`: the coupling strength ``v``.
* `alpha = nothing`: the dimensionless coupling strength ``α``.
    If provided, this will override the value of `v` using the relation
    ``v = \\sqrt{2α ω²/(l \\sqrt{2m ω})}``.
* `two_m = 1.0`: twice the particle mass ``2m``.
* `omega = 1.0`: the oscillation frequency of the phonons ``ω``.
* `l = 1.0`: the box size in real space ``l``. Provides scale parameter of the momentum
    lattice.
* `momentum_cutoff = nothing`: the maximum boson momentum allowed for an address.
* `mode_cutoff`: the maximum number of bosons in each momentum mode. Defaults to the maximum
    value supported by the address type [`BoseFS{missing}`](@ref).
    [`maximum_mode_occupation(hamiltonian)`](@ref Main.Interfaces.maximum_mode_occupation)
    will return this value.

# Examples
```jldoctest
julia> fs = BoseFS{missing}(0,0,0)
BoseFS{missing}(0, 0, 0)

julia> ham = FroehlichPolaron1D(fs; v=0.5)
FroehlichPolaron1D(fs"|0 0 0⟩{}"; v=0.5, two_m=1.0, omega=1.0, l=1.0, p=0.0, mode_cutoff=255)

julia> dimension(ham)
16777216

julia> dimension(FroehlichPolaron1D(fs; v=0.5, mode_cutoff=5))
216

julia> maximum_mode_occupation(FroehlichPolaron1D(fs; v=0.5, mode_cutoff=5))
5
```
!!! warning
    This type is retained for testing purposes only and may be removed in a future version.
    Use [`FroehlichPolaron`](@ref) instead.

See also [`BoseFS`](@ref), [`dimension`](@ref), [`maximum_mode_occupation`](@ref),
[`AbstractHamiltonian`](@ref), [`FroehlichPolaron`](@ref).
"""
struct FroehlichPolaron1D{
    T, # eltype
    M, # number of modes
    A<:BoseFS{missing,M}, # address type
    MC # momentum cutoff indicating type
} <: AbstractHamiltonian{T}
    addr::A
    v::T
    two_m::T
    omega::T
    l::T
    p::T
    ks::SVector{M,T} # values for k
    momentum_cutoff::MC
    mode_cutoff::Int
end

function FroehlichPolaron1D{T}(
    addr::BoseFS{missing,M,SVector{M,AT}};
    v=1,
    two_m=1,
    omega=1,
    l=1,
    p=0,
    momentum_cutoff=nothing,
    mode_cutoff=nothing,
    mass=nothing, # deprecated keyword, use `two_m` instead
    alpha=nothing,
) where {T,M,AT}
    if l ≤ 0
        throw(ArgumentError("l must be positive"))
    end
    T <: AbstractFloat || throw(ArgumentError("T must be a subtype of AbstractFloat"))

    if !isnothing(mass)
        @warn "The keyword argument `mass` is deprecated. Use `two_m` instead."
        two_m = mass
    end

    v, p, two_m, omega, l = T.((v, p, two_m, omega, l))
    if !isnothing(alpha)
        v = sqrt(2 * T(alpha) * omega^2 / (l * sqrt(two_m * omega)))::T
    end

    step = T(2π/M)
    if isodd(M)
        start = -π * T(1 + 1/M) + step
    else
        start = -π + step
    end
    kr = (M/l)*range(start; step = step, length = M)
    ks = SVector{M,T}(kr)

    if !isnothing(momentum_cutoff)
        momentum_cutoff = T(momentum_cutoff)
        momentum = dot(ks,onr(addr))
        if abs(momentum) > momentum_cutoff
            throw(ArgumentError("Starting address has momentum $momentum which cannot exceed momentum_cutoff $momentum_cutoff"))
        end
    end

    if isnothing(mode_cutoff)
        mode_cutoff = Int(typemax(AT))
    end
    mode_cutoff = floor(Int, mode_cutoff)::Int
    if _exceed_mode_cutoff(mode_cutoff, addr)
        throw(ArgumentError("Starting address cannot have occupations that exceed mode_cutoff"))
    end
    return FroehlichPolaron1D(addr, v, two_m, omega, l, p, ks, momentum_cutoff, mode_cutoff)
end
function FroehlichPolaron1D(
    addr::BoseFS{missing};
    v=1,
    two_m=1,
    omega=1,
    l=1,
    p=0,
    mass=nothing, # deprecated keyword, use `two_m` instead
    kwargs...
)
    if !isnothing(mass)
        @warn "The keyword argument `mass` is deprecated. Use `two_m` instead."
        two_m = mass
    end

    T = float(promote_type(typeof(v), typeof(two_m), typeof(omega), typeof(l), typeof(p)))
    return FroehlichPolaron1D{T}(addr; v=v, two_m=two_m, omega=omega, l=l, p=p, kwargs...)
end

function Base.show(io::IO, h::FroehlichPolaron1D)
    compact_addr = repr(h.addr, context=:compact => true) # compact print address
    print(io, "FroehlichPolaron1D")
    eltype(h) === Float64 || print(io, "{$(eltype(h))}")
    print(io, "($compact_addr; ")
    print(io, "v=$(h.v), two_m=$(h.two_m), omega=$(h.omega), l=$(h.l), p=$(h.p), ")
    isnothing(h.momentum_cutoff) || print(io, "momentum_cutoff=$(h.momentum_cutoff), ")
    print(io, "mode_cutoff=$(h.mode_cutoff))")
end

function starting_address(h::FroehlichPolaron1D)
    return h.addr
end
function Interfaces.maximum_mode_occupation(h::FroehlichPolaron1D)
    return h.mode_cutoff
end

LOStructure(::Type{<:FroehlichPolaron1D{<:Real}}) = IsHermitian()

function diagonal_element(h::FroehlichPolaron1D{<:Any,M}, addr::BoseFS{missing,M}) where {M}
    map = onr(addr)
    p_f = dot(h.ks, map)
    return h.omega * num_particles(addr) + (h.p - p_f)^2 / h.two_m
end

function num_offdiagonals(::FroehlichPolaron1D{<:Any,M}, ::BoseFS{missing,M}) where {M}
    return 2M #num_occupied_modes
end

function get_offdiagonal(h::FroehlichPolaron1D{<:Any,M,<:Any,Nothing}, addr::BoseFS{missing,M},chosen) where {M}
    # branch that bypasses momentum cutoff
    return _froehlich_offdiag(h, addr, chosen)
end

function get_offdiagonal(h::FroehlichPolaron1D{T,M,<:Any,T}, addr::BoseFS{missing,M}, chosen) where {M,T}
    # branch for checking momentum cutoff
    naddress, value = _froehlich_offdiag(h, addr, chosen)

    new_p_tot = dot(h.ks, onr(naddress))
    if abs(new_p_tot) > h.momentum_cutoff # check if momentum of new address exceeds momentum_cutoff
        return addr, zero(T)
    else
        return naddress, value
    end
end

function _froehlich_offdiag(h, addr::BoseFS{missing,M},chosen) where {M}
    T = eltype(h)
    if chosen ≤ M # assign first M indices to creations
        if onr(addr)[chosen] ≥ h.mode_cutoff # check whether occupation exceeds cutoff
            return addr, zero(T)
        else
            naddress, value = excitation(T, addr, (chosen,), ())
            return naddress, - h.v * value
        end
    else # remaining indices are destructions

        naddress, value = excitation(T, addr, (), (chosen - M,))
        return naddress, - h.v * value
    end
end

function _exceed_mode_cutoff(mode_cutoff, addr::BoseFS{missing,M}) where {M}
    return any(x -> x > mode_cutoff, onr(addr))
end

function dimension(h::FroehlichPolaron1D, address)
    # takes into account `mode_cutoff` but not `momentum_cutoff`
    M = num_modes_check_equal(address)
    n = h.mode_cutoff
    return BigInt(n + 1)^BigInt(M)
end
