# third level: `fciqmc_col!()`

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
    fciqmc_col!(w, ham, add, num, shift, dτ)
    fciqmc_col!(::Type{T}, args...)
    -> spawns, deaths, clones, antiparticles, annihilations
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
    return (:exact_steps,), SVector(0,)
end
function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    for i in axes(ham, 1) # iterate through off-diagonal rows of `ham`
        i == add && continue
        deposit!(w, i, -dτ * ham[i, add] * num, add => num)
    end
    deposit!(w, add, (1 + dτ * (shift - ham[add, add])) * num, add => num) # diagonal
    return (1,)
end
function fciqmc_col!(::IsDeterministic, w, ham::AbstractHamiltonian, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in offdiagonals(ham, add)
        deposit!(w, nadd, -dτ * elem * num, add => num)
    end
    # diagonal death or clone
    deposit!(w, add, (1 + dτ * (shift - diagonal_element(ham,add))) * num, add => num)
    return (1,)
end

function step_stats(::IsStochasticInteger)
    return (
        (:spawns, :deaths, :clones, :antiparticles, :annihilations),
        SVector(0, 0, 0, 0, 0),
    )
end
function fciqmc_col!(
    ::IsStochasticInteger, w, ham::AbstractHamiltonian, add, num::Real, shift, dτ
)
    # version for single population of integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = offdiagonals(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn + 1 - cRand()) # project to integer
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            deposit!(w, naddress, nspawns, add => num)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagonal_element(ham,add)
    pd = dτ * (dME - shift)
    newdiagpop = (1-pd)*num # new diag population but not yet integer
    ndiag = floor(newdiagpop + 1 - cRand()) # project to integer
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    deposit!(w, add, ndiags, add => num) # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function step_stats(::DictVectors.IsStochastic2Pop)
    return (
        (:spawns, :deaths, :clones, :antiparticles, :annihilations),
        SVector{5,Complex{Int}}(0, 0, 0, 0, 0),
    )
end
function fciqmc_col!(::DictVectors.IsStochastic2Pop, w, ham::AbstractHamiltonian, add,
                        cnum::Complex, cshift, dτ
)
    # version for complex integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(cnum)
    # stats reported are complex, for each component separately
    hops = offdiagonals(ham,add)
    # real psips first
    num = real(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
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
        nspawn = floor(pspawn) # deal with integer part separately
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
    ndiags = convert(typeof(num),ndiag) # now real integer type
    if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    end
    deposit!(w, add, ndiags, add => num) # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += abs(real(ndiags) - num)
    elseif pd < 1
        deaths += abs(real(ndiags) - num)
    else
        antiparticles += abs(real(ndiags))
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
        antiparticles += im*abs(imag(ndiags))
    end

    # imaginary part of shift leads to spawns across populations
    cspawn = im*dτ*imag(cshift)*cnum # to be spawned as complex number with signs

    # real part - to be spawned into real walkers
    rspawn = real(cspawn) # float with sign
    nspawn = trunc(rspawn) # deal with integer part separately
    cRand() < abs(rspawn - nspawn) && (nspawn += sign(rspawn)) # random spawn
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
    nspawn = trunc(ispawn) # deal with integer part separately
    cRand() < abs(ispawn - nspawn) && (nspawn += sign(ispawn)) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
    if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(imag(w[add])),abs(nspawn))
    end
    deposit!(w, add, cnspawn, add => cnum)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function step_stats(::IsStochasticWithThreshold)
    return (:spawns, :deaths), SVector(0, 0)
end
function fciqmc_col!(s::IsStochasticWithThreshold, w, ham::AbstractHamiltonian,
        add, val::N, shift, dτ) where N <: Real
    deaths = 0
    spawns = 0

    # diagonal death or clone: deterministic fomula
    # w[add] += (1 + dτ*(shift - diagonal_element(ham,add)))*val
    # projection to threshold should be applied after all colums are evaluated
    new_val = (1 + dτ*(shift - diagonal_element(ham,add)))*val
    # apply threshold if necessary
    if abs(new_val) < s.threshold
        # project stochastically to threshold
        # w[add] += (abs(new_val)/s.threshold > cRand()) ? sign(new_val)*s.threshold : 0
        new_val = ifelse(
            cRand() < abs(new_val)/s.threshold, sign(new_val) * s.threshold, zero(new_val)
        )
        if iszero(new_val)
            deaths += 1
        else
            deposit!(w, add, new_val, add => val)
        end
    else
        deposit!(w, add, new_val, add => val)
    end

    # off-diagonal: spawning psips stochastically
    # only integers are spawned!!
    hops = offdiagonals(ham, add)
    # first deal with integer psips
    for n in 1:floor(abs(val)) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if !iszero(nspawns)
            deposit!(w, naddress, nspawns, add => val)
            spawns += 1
        end
    end
    # deal with non-integer remainder: attempt to spawn
    rval =  abs(val%1) # non-integer part reduces probability for spawning
    naddress, pgen, matelem = random_offdiagonal(hops)
    pspawn = rval * dτ * abs(matelem) /pgen # non-negative Float64
    nspawn = floor(pspawn) # deal with integer part separately
    cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
    # at this point, nspawn is non-negative
    # now converted to correct type and compute sign
    nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
    # - because Hamiltonian appears with - sign in iteration equation
    if !iszero(nspawns)
        deposit!(w, naddress, nspawns, add => val)
        spawns += 1
    end
    # done with stochastic spawning
    return (spawns, deaths)
end

"""
    threshold_projected_deposit!
This function performs threshold projection before spawning, but only for
`IsDynamicSemistochastic` with the `project_later` parameter set to `false`.
"""
function threshold_projected_deposit!(s::IsDynamicSemistochastic{<:Any,false}, w, add, val, p)
    threshold = s.proj_threshold
    absval = abs(val)
    if absval < threshold
        if cRand() < abs(val) / threshold
            deposit!(w, add, sign(val) * threshold, p)
        end
    else
        deposit!(w, add, val, p)
    end
    return nothing
end
function threshold_projected_deposit!(::IsDynamicSemistochastic{<:Any,true}, w, add, val, p)
    deposit!(w, add, val, p)
end

function step_stats(::IsDynamicSemistochastic)
    return (:exact_steps, :inexact_steps, :spawns), SVector(0, 0, 0)
end
function fciqmc_col!(
    s::IsDynamicSemistochastic,
    w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    absval = abs(val)
    spawns = 0

    # Diagonal step:
    new_val = (1 + dτ*(shift - diagonal_element(ham, add))) * val
    threshold_projected_deposit!(s, w, add, new_val, add => val)

    hops = offdiagonals(ham, add)
    if s.rel_threshold * val ≥ length(hops) || val ≥ s.abs_threshold
        # Exact multiplication when conditions are met.
        factor = dτ * val
        for (new_add, mat_elem) in hops
            threshold_projected_deposit!(s, w, new_add, -factor * mat_elem, add => val)
        end
        spawns = length(hops)
        return (1, 0, spawns)
    else
        remainder = absval % 1
        hasrem = !iszero(remainder)
        spawns = ceil(Int, absval)
        for i in 1:spawns
            new_add, gen_prob, mat_elem = random_offdiagonal(hops)
            rem_factor = ifelse(i == 1 & hasrem, remainder, 1.0)
            new_val = sign(val) * rem_factor * dτ * mat_elem / gen_prob
            threshold_projected_deposit!(s, w, new_add, -new_val, add => val)
        end

        return (0, 1, spawns)
    end
end

function step_stats(::IsSemistochasticWithList)
    return (:exact_steps, :inexact_steps, :spawns), SVector(0, 0, 0)
end

function fciqmc_col!(
    s::IsSemistochasticWithList,
    w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    absval = abs(val)
    spawns = 0

    # Diagonal step:
    new_val = (1 + dτ*(shift - diagonal_element(ham, add))) * val
    deposit!(w, add, new_val, add => val)

    hops = offdiagonals(ham, add)
    if add in s.preserve
        # Exact multiplication when conditions are met.
        factor = dτ * val
        for (new_add, mat_elem) in hops
            deposit!(w, new_add, -factor * mat_elem, add => val)
        end
        spawns = length(hops)
        return (1, 0, spawns)
    else
        remainder = absval % 1
        hasrem = !iszero(remainder)
        spawns = ceil(Int, absval)
        for i in 1:spawns
            new_add, gen_prob, mat_elem = random_offdiagonal(hops)
            rem_factor = ifelse(i == 1 & hasrem, remainder, 1.0)
            new_val = sign(val) * rem_factor * dτ * mat_elem / gen_prob
            deposit!(w, new_add, -new_val, add => val)
        end

        return (0, 1, spawns)
    end
end
