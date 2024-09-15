
"""
    SpectralStrategy{S}

Abstract type for spectral strategies. The spectral strategy is used to control the number
of spectral states used in the simulation.

## Implemented Strategies

* [`GramSchmidt`](@ref): Orthogonalize the spectral states using the Gram-Schmidt procedure.
"""
abstract type SpectralStrategy{S} end

"""
    num_spectral_states(state_or_strategy)

Return the number of spectral states used in the simulation.
"""
num_spectral_states(::SpectralStrategy{S}) where {S} = S

"""
    GramSchmidt(S; orthogonalization_interval = 1) <: SpectralStrategy{S} 
Use the Gram-Schmidt procedure to orthogonalize the excited states. A total of `S` spectral
states are used in the simulation, and they are orthogonalized every 
`orthogonalization_interval` steps.

Use with the keyword argument `spectral_strategy` in [`ProjectorMonteCarloProblem`](@ref).
"""
struct GramSchmidt{S} <: SpectralStrategy{S}
    orthogonalization_interval::Int
end

function GramSchmidt(S = 1; orthogonalization_interval = 1)
    return GramSchmidt{S}(orthogonalization_interval)
end

function Base.show(io::IO, gs::GramSchmidt{S}) where {S}
    print(io, "GramSchmidt(", S, "; orthogonalization_interval=")
    print(io, gs.orthogonalization_interval, ")")
end
