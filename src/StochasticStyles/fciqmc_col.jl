# third level: `fciqmc_col!()`
###
### `fciqmc_col!` implementations
###
# only use real part of the shift if the coefficients are real

# otherwise, pass on complex shift in generic method
fciqmc_col!(w::Union{AbstractArray,AbstractDVec}, args...) = fciqmc_col!(StochasticStyle(w), w, args...)

# generic method for unknown trait: throw error
fciqmc_col!(::Type{T}, args...) where T = throw(TypeError(:fciqmc_col!,
    "first argument: trait not recognised",StochasticStyle,T))

###
### Defaults
###
function step_stats(s::StochasticStyle{T}) where {T}
    spawning_strat = SpawningStrategy(s)
    # This gets the default name => type pairs.
    pairs = step_stats(s, T)
    names = (first.(pairs)..., :deaths, :clones, :zombies)
    values = MultiScalar(zero.(last.(paris))..., z, z, z)
    return names, values
end

###
### Concrete implementations
###




function step_stats(::IsDynamicSemistochastic)
    return (
        (:exact_steps, :inexact_steps, :spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
end
function fciqmc_col!(
    s::IsDynamicSemistochastic,
    w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    clones, deaths, zombies, ann_d = diagonal_step!(w, ham, add, val, dτ, shift)
    exact, inexact, spawns, ann_o = semistochastic_spawns!(w, ham, add, val, dτ, 0, s.strength)
    return (exact, inexact, spawns, deaths, clones, zombies, ann_d + ann_o)
end

function step_stats(::IsExplosive)
    return (
        (:explosions, :ticks, :normal_steps,
         :explosive_spawns, :normal_spawns,
         :clones, :deaths, :zombies
         ),
        MultiScalar(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
end
function fciqmc_col!(
    s::IsExplosive, w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    explosions = normal_steps = ticks = 0
    explosive_spawns = normal_spawns = 0.0
    clones = deaths = zombies = 0.0

    pd = dτ * (diagonal_element(ham, add) - shift) * s.delay_factor
    if abs(val) ≤ s.explosion_threshold && 0 ≤ pd < 1
        if cRand() < pd
            _, _, explosive_spawns, _ = semistochastic_spawns!(
                w, ham, add, val / pd, dτ, 0, s.splatter_factor
            )
            explosions = 1
        else
            deposit!(w, add, val, add => val)
            ticks = 1
        end
    else
        clones, deaths, zombies, _ = diagonal_step!(w, ham, add, val, dτ, shift)
        _, _, normal_spawns, _ = semistochastic_spawns!(w, ham, add, val, dτ)
        normal_steps = 1
    end

    return (
        explosions, ticks, normal_steps,
        explosive_spawns, normal_spawns,
        clones, deaths, zombies,
    )
end
