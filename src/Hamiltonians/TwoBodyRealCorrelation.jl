"""
    G2RealCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Two-body operator for density-density correlation between sites separated by `d`
with `0 ≤ d < M`. Can be applied to a vector in any [`SingleComponentFockAddress`](@ref) basis.
```math
    \\hat{G}^{(2)}(d) = \\frac{1}{M} \\sum_i^M \\hat{n}_i (\\hat{n}_{i+d} - \\delta_{0d}).
```
Assumes periodic boundary conditions where
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

# Arguments
- `d::Integer`: distance between sites.

# See also

* [`HubbardReal1D`](@ref)
* [`G2MomCorrelator`](@ref)
* [`AbstractHamiltonian`](@ref)
"""
struct G2RealCorrelator{D} <: AbstractHamiltonian{Float64}
end

G2RealCorrelator(d::Int) = G2RealCorrelator{d}()

function Base.show(io::IO, ::G2RealCorrelator{D}) where {D}
    print(io, "G2RealCorrelator($D)")
end

LOStructure(::Type{<:G2RealCorrelator}) = IsDiagonal()

function diagonal_element(::G2RealCorrelator{D}, add::SingleComponentFockAddress{N,M}) where {D,N,M}
    d = mod(D, M)
    if d == 0
        v = onr(add)
        return dot(v, v .- 1) / M
    else
        v = onr(add)
        result = 0
        for i in eachindex(v)
            result += v[i] * v[mod1(i + d, M)]
        end
        return result / M
    end    
end

num_offdiagonals(::G2RealCorrelator, ::SingleComponentFockAddress) = 0

# not needed:
# get_offdiagonal(::G2RealCorrelator, add)
# starting_address(::G2RealCorrelator)