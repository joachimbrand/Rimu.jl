
"""
    SpectralStrategy{S}

Abstract type for spectral strategies. The spectral strategy is used to control the number
of spectral states used in the simulation.
"""
abstract type SpectralStrategy{S} end

"""
    num_spectral_states(state_or_strategy)

Return the number of spectral states used in the simulation.
"""
num_spectral_states(::SpectralStrategy{S}) where {S} = S

"""
    GramSchmidt{S} <: SpectralStrategy{S}

Use the Gram-Schmidt procedure to orthogonalize the excited states. A total of `S` spectral
states are used in the simulation.
"""
struct GramSchmidt{S} <: SpectralStrategy{S} end

function GramSchmidt(num_spectral_states = 1)
    num_spectral_states > 1 && throw(ArgumentError("num_spectral_states > 1 is currently not supported."))
    GramSchmidt{num_spectral_states}()
end
