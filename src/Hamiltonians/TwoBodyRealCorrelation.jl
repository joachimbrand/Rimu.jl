"""
    G2RealCorrelator(d::Int) <: AbstractHamiltonian{Float64}

Two-body operator for density-density correlation between sites separated by `d`.
```math
    G_2(d) = \\sum_i^M \\hat{n}_i (\\hat{n}_{i+d} - \\delta_{0d}).
```
Can be applied to any [`SingleComponentFockAddress`](@ref).
Assumes periodic boundary conditions where
```math
    G_2(-M/2 \\leq d < 0) = G_2(|d|).
```
and normalisation
```math
    \\sum_{d=0}^{M-1} G_2(d) = N (N-1).
```

# Arguments
- `d::Integer`: distance between sites, must be 0 ≤ d ≤ M/2.

# See also

* [`HubbardReal1D`](@ref)
* [`G2Correlator`](@ref)
* [`AbstractHamiltonian`](@ref)
"""
struct G2RealCorrelator{D} <: AbstractHamiltonian{Float64}
end

function G2RealCorrelator(d)
    d::Int
    return G2RealCorrelator{d}()
end

function Base.show(io::IO, ::G2RealCorrelator{D}) where {D}
    print(io, "G2RealCorrelator($D)")
end

LOStructure(::Type{<:G2RealCorrelator}) = IsDiagonal()

function diagonal_element(::G2RealCorrelator{D}, add::BoseFS{N,M}) where {D,N,M}
    if D == 0
        v = onr(add)
        return float(dot(v, v .- 1))
    elseif 0 < D < M ÷ 2 + 1
        v = onr(add)
        result = 0
        for i in eachindex(v)
            result += v[i] * v[mod1(i + D, M)]
        end
        return float(result)
    else
        throw(DomainError(D, "Bad input for G2RealCorrelator. Requires: 0 ≤ d ≤ M ÷ 2"))
    end    
end

num_offdiagonals(g::G2RealCorrelator, add::AbstractFockAddress) = 0

# not needed:
# get_offdiagonal(::G2RealCorrelator, add)
# starting_address(::G2RealCorrelator)