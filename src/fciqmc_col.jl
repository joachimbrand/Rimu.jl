# third level: `fciqmc_col!()`
using StatsBase

"""
    step_stats(::StochasticStyle)

Return a tuple of names (`Symbol` or `String`) and a zeros of values of the same length.
These will be reported as columns in the `DataFrame` returned by [`lomc!`](@ref).
"""
step_stats(::StochasticStyle)

step_stats(v, n) = step_stats(StochasticStyle(v), n)
function step_stats(s::StochasticStyle, ::Val{N}) where N
    if N == 1
        return step_stats(s)
    else
        names, stats = step_stats(s)
        return names, MVector(ntuple(_ -> stats, Val(N)))
    end
end

"""
    projected_deposit!(w, add, val, parent, threshold=0)

Like [`deposit!`](@ref), but performs threshold projection before spawning. If `eltype(w)` is
an `Integer`, values are stochastically rounded.

Returns the value deposited and the number of annihilations.
"""
function projected_deposit!(w, add, val, parent, threshold=0)
    projected_deposit!(valtype(w), w, add, val, parent, threshold)
end
# Non-integer
function projected_deposit!(::Type{T}, w, add, val, parent, threshold) where T
    absval = abs(val)
    if absval < threshold
        if cRand() < abs(val) / threshold
            val = sign(val) * threshold
        else
            val = zero(val)
        end
    end
    annihilations = zero(T)
    if !iszero(val)
        prev = w[add]
        if sign(prev) ≠ sign(val)
            annihilations = min(abs(prev), abs(val))
        end
        deposit!(w, add, val, parent)
    end

    return abs(val), annihilations
end
# Round to integer
function projected_deposit!(::Type{T}, w, add, val, parent, _) where {T<:Integer}
    intval = T(sign(val)) * floor(T, abs(val) + cRand())
    annihilations = zero(T)
    if !iszero(intval)
        prev = w[add]
        if sign(prev) ≠ sign(intval)
            annihilations = min(abs(prev), abs(intval))
        end
        deposit!(w, add, intval, parent)
    end
    return abs(intval), annihilations
end

"""
    diagonal_step!(w, ham, add, val, dτ, shift, proj=0)

Perform diagonal step on a walker `add => val`. Optional argument `proj` sets the
projection threshold. If `eltype(w)` is an `Integer`, the `val` is rounded stochastically.
"""
function diagonal_step!(w, ham, add, val, dτ, shift, proj=0)
    clones = deaths = zombies = zero(valtype(w))

    pd = dτ * (diagonal_element(ham, add) - shift)
    new_val = (1 - pd) * val
    res, annihilations = projected_deposit!(w, add, new_val, add => val, proj)
    if pd < 0
        clones = abs(res - val)
    elseif pd < 1
        deaths = abs(res - val)
    else
        deaths = abs(val)
        zombies = abs(res)
    end
    return (clones, deaths, zombies, annihilations)
end

"""
    spawns!(w, ham, add, val, dτ, proj=0, strength=1)

Perform spawns from walker `add => val`. The number of spawns performed is always equal to
`ceil(abs(val))`. `proj` sets the projection threshold which should be zero for integer
spawns. `strength` controls the amount of spawning performed without changing the expected
value; a `strength` of 2 will perform twice the number of spawns with half the magnuitude.

Returns the number of spawns and the number of annihilations.
"""
function spawns!(w, ham, add, val::T, dτ, proj=0, strength=1) where {T}
    hops = offdiagonals(ham, add)
    spawns = annihilations = zero(valtype(w))
    num_spawns = floor(Int, abs(val) * strength)
    magnitude = val / num_spawns

    for _ in 1:num_spawns
        new_add, prob, mat_elem = random_offdiagonal(hops)
        new_val = -mat_elem * magnitude / prob * dτ
        spw, ann = projected_deposit!(w, new_add, new_val, add => val, proj)
        spawns += spw
        annihilations += ann
    end

    return (spawns, annihilations)
end

"""
    exact_spawns!(w, ham, add, val, dτ, proj=0)

Perform exact spawning step from a walker `add => val`. `proj` sets the projection threshold.

Returns the number of spawns and annihilations.
"""
function exact_spawns!(w, ham, add, val, dτ, proj=0)
    hops = offdiagonals(ham, add)
    spawns = annihilations = zero(valtype(w))
    factor = -dτ * val
    for (new_add, mat_elem) in hops
        spw, ann = projected_deposit!(w, new_add, factor * mat_elem, add => val, proj)
        spawns += spw
        annihilations += ann
    end
    return (spawns, annihilations)
end

"""
    semistochastic_spawns!(w, ham, add, val, dτ, proj=0, strength=1)

Perform semistochastic spawns from a walker `add => val`. `proj` sets the projection
threshold. `strength` sets the number of spawns to perform, e.g. if `val=5` and
`strength=2`, 10 spawns will be performed.

Unlike [`spawns!`](@ref), this function calls [`exact_spawns!`](@ref) when the number of
spawns exceeds the number of offdiagonals.

Returns the numbers of exact and inexact steps, the number of spawns, and the number of
annihilations.
"""
function semistochastic_spawns!(w, ham, add, val, dτ, proj=0, strength=1)
    nhops = num_offdiagonals(ham, add)
    num_spawns = strength * abs(val)
    if num_spawns ≥ nhops
        # Exact multiplication.
        spawns, annihilations = exact_spawns!(w, ham, add, val, dτ, proj)
        return (1, 0, spawns, annihilations)
    else
        # Regular spawns.
        spawns, annihilations = spawns!(w, ham, add, val, dτ, proj, strength)
        return (0, 1, spawns, annihilations)
    end
    #=
    TODO statstools thing
    else
        spawns = floor(Int, num_spawns)
        selected = sample(1:length(hops), spawns; replace=false)
        # new_val = sign(val) * rem_factor * dτ * mat_elem / (gen_prob * strength)
        α = sign(val) * (num_spawns / spawns) * dτ * length(hops) / strength
        for i in selected
            new_add, mat_elem = hops[i]
            new_val = α * mat_elem
            projected_deposit!(w, new_add, -new_val, add => val, proj)
        end

        return (0, 1, spawns)
    end
    =#
end

"""
    fciqmc_col!(w, ham, add, num, shift, dτ)
    fciqmc_col!(::Type{T}, args...)
    -> spawns, deaths, clones, zombies, annihilations
Spawning and diagonal step of FCIQMC for single column of `ham`. In essence it
computes

`w .+= (1 .+ dτ.*(shift .- ham[:,add])).*num`.

Depending on `T == `[`StochasticStyle(w)`](@ref), a stochastic or deterministic algorithm will
be chosen. The possible values for `T` are:

- [`IsDeterministic()`](@ref) deteministic algorithm
- [`IsStochasticInteger()`](@ref) stochastic version where the changes added to `w` are purely integer, according to the FCIQMC algorithm
- [`IsStochasticWithThreshold(c)`](@ref) stochastic algorithm with floating point walkers.
"""
function fciqmc_col!(w::Union{AbstractArray{T},AbstractDVec{K,T}},
    ham, add, num, shift, dτ
) where  {K,T<:Real}
    return fciqmc_col!(StochasticStyle(w), w, ham, add, num, real(shift), dτ)
end
# only use real part of the shift if the coefficients are real

# otherwise, pass on complex shift in generic method
fciqmc_col!(w::Union{AbstractArray,AbstractDVec}, args...) = fciqmc_col!(StochasticStyle(w), w, args...)

# generic method for unknown trait: throw error
fciqmc_col!(::Type{T}, args...) where T = throw(TypeError(:fciqmc_col!,
    "first argument: trait not recognised",StochasticStyle,T))

###
### Concrete implementations
###

function step_stats(::IsDeterministic)
    return (:exact_steps,), MultiScalar(0,)
end
function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    for i in axes(ham, 1) # iterate through off-diagonal rows of `ham`
        i == add && continue
        deposit!(w, i, -dτ * ham[i, add] * num, add => num)
    end
    deposit!(w, add, (1 + dτ * (shift - ham[add, add])) * num, add => num) # diagonal
    return (1,)
end
function fciqmc_col!(::IsDeterministic, w, ham::AbstractHamiltonian, add, val, shift, dτ)
    diagonal_step!(w, ham, add, val, dτ, shift)
    exact_spawns!(w, ham, add, val, dτ)
    return (1,)
end

function step_stats(::IsStochasticInteger)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0, 0, 0, 0, 0),
    )
end
function fciqmc_col!(
    ::IsStochasticInteger, w, ham::AbstractHamiltonian, add, num::Real, shift, dτ
)
    clones, deaths, zombies, ann_diag = diagonal_step!(w, ham, add, num, dτ, shift)
    spawns, ann_offdiag = spawns!(w, ham, add, num, dτ)
    return (spawns, deaths, clones, zombies, ann_diag + ann_offdiag)
end

function step_stats(::DictVectors.IsStochastic2Pop)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(ntuple(_ -> 0 + 0im, Val(5))),
    )
end
function fciqmc_col!(::DictVectors.IsStochastic2Pop, w, ham::AbstractHamiltonian, add,
                        cnum::Complex, cshift, dτ
)
    # version for complex integer psips
    # off-diagonal: spawning psips
    spawns::typeof(cnum) = deaths = clones = zombies = annihilations = zero(cnum)
    # stats reported are complex, for each component separately
    hops = offdiagonals(ham,add)
    # real psips first
    num = real(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(Int, pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(real(w[naddress])) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(real(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            deposit!(w, naddress, nspawns, add => cnum)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # now imaginary psips
    num = imag(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(Int, pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = im*convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(imag(w[naddress])) * sign(imag(nspawns)) < 0 # record annihilations
            annihilations += min(abs(imag(w[naddress])),abs(imag(nspawns)))
        end
        if !iszero(nspawns)
            deposit!(w, naddress, nspawns, add => cnum)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += im*abs(nspawns)
        end
    end

    # diagonal death / clone
    shift = real(cshift) # use only real part of shift for now
    dME = diagonal_element(ham,add)
    pd = dτ * (dME - shift) # real valued so far
    cnewdiagpop = (1-pd)*cnum # now it's complex
    # treat real part
    newdiagpop = real(cnewdiagpop)
    num = real(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) + 0im # now real integer type
    if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    end
    deposit!(w, add, ndiags, add => num) # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += abs(real(ndiags) - num)
    elseif pd < 1
        deaths += abs(real(ndiags) - num)
    else
        zombies += abs(real(ndiags))
    end
    # treat imaginary part
    newdiagpop = imag(cnewdiagpop)
    num = imag(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = im*convert(typeof(num),ndiag) # now complex integer type
    if sign(imag(w[add])) ≠ sign(imag(ndiag)) # record annihilations
        annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    end
    deposit!(w, add, ndiags, add => cnum)
    if  pd < 0 # record event statistics
        clones += im*abs(imag(ndiags) - num)
    elseif pd < 1
        deaths += im*abs(imag(ndiags) - num)
    else
        zombies += im*abs(imag(ndiags))
    end

    # imaginary part of shift leads to spawns across populations
    cspawn = im*dτ*imag(cshift)*cnum # to be spawned as complex number with signs

    # real part - to be spawned into real walkers
    rspawn = real(cspawn) # float with sign
    nspawn = trunc(Int, rspawn) # deal with integer part separately
    cRand() < abs(rspawn - nspawn) && (nspawn += Int(sign(rspawn))) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn)
    if sign(real(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(real(w[add])),abs(nspawn))
    end
    deposit!(w, add, cnspawn, add => cnum)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    # imag part - to be spawned into imaginary walkers
    ispawn = imag(cspawn) # float with sign
    nspawn = trunc(Int, ispawn) # deal with integer part separately
    cRand() < abs(ispawn - nspawn) && (nspawn += Int(sign(ispawn))) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
    if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(imag(w[add])),abs(nspawn))
    end
    deposit!(w, add, cnspawn, add => cnum)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    return (spawns, deaths, clones, zombies, annihilations)
    # note that w is not returned
end

function step_stats(::IsStochasticWithThreshold)
    return (
        (:spawns, :deaths, :clones, :zombies, :annihilations),
        MultiScalar(0.0, 0.0, 0.0, 0.0, 0.0)
    )
end
function fciqmc_col!(
    s::IsStochasticWithThreshold, w, ham::AbstractHamiltonian, add, val, shift, dτ
)
    clones, deaths, zombies, ann_d = diagonal_step!(w, ham, add, val, dτ, shift, s.threshold)
    spawns, ann_o = spawns!(w, ham, add, val, dτ, s.threshold)
    return (spawns, deaths, clones, zombies, ann_d + ann_o)
end

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
    exact, inexact, spawns, ann_o = semistochastic_spawns!(w, ham, add, val, dτ)
    return (exact, inexact, spawns, deaths, clones, zombies, ann_d + ann_o)
end

function step_stats(::IsExplosive)
    return (
        (:explosive_steps,
         :explosive_spawns,
         :unevents,
         :normal_steps,
         :normal_spawns,
         :clones, :deaths, :zombies
         ),
        MultiScalar(0, 0, 0, 0, 0, 0.0, 0.0, 0.0),
    )
end
function fciqmc_col!(
    s::IsExplosive, w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    explosive_steps = normal_steps = unevents = 0
    explosive_spawns = normal_spawns = 0
    clones = deaths = zombies = 0.0

    pd = dτ * (diagonal_element(ham, add) - shift)
    if abs(val) ≤ s.explosion_threshold && 0 ≤ pd < 1
        if cRand() < pd
            _, _, explosive_spawns = semistochastic_spawns!(
                w, ham, add, val / pd, dτ, 0, s.splatter_factor, true
            )
            explosions = 1
        else
            deposit!(w, add, val, add => val)
            unevents = 1
        end
    else
        clones, deaths, zombies = diagonal_step!(w, ham, add, val, dτ, shift)
        _, _, normal_spawns = semistochastic_spawns!(w, ham, add, val, dτ, 0, 1, true)
        normal_steps = 1
    end

    return (
        explosive_steps, explosive_spawns, unevents, normal_steps, normal_spawns,
        clones, deaths, zombies,
    )
end
