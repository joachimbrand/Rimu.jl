
"""
    SpectralStrategy{S}
Abstract type for spectral strategies. The spectral strategy is used to control the number
of spectral states used in the simulation.
"""
abstract type SpectralStrategy{S} end

"""
    num_spectral_states(S::SpectralStrategy)
Return the number of spectral states used in the simulation.
"""
num_spectral_states(::SpectralStrategy{S}) where {S} = S

"""
    GramSchmidt{S} <: SpectralStrategy{S}
Use the Gram-Schmidt procedure to orthogonalize the excited states. A total of `S` spectral
states are used in the simulation.
"""
struct GramSchmidt{S} <: SpectralStrategy{S} end
GramSchmidt(num_spectral_states = 1) = GramSchmidt{num_spectral_states}()
