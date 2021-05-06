# third level: `fciqmc_col!()`

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
- [`IsStochastic()`](@ref) stochastic version where the changes added to `w` are purely integer, according to the FCIQMC algorithm
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

function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    w[add] += (1 + dτ * shift) * num # diagonal without Hamiltonian contribution
    w .+= -dτ .* ham[:, add] .* num # full matrix column
    # todo: return something sensible
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(::IsDeterministic, w, ham::AbstractHamiltonian, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in offdiagonals(ham, add)
        w[nadd] += -dτ * elem * num
    end
    # diagonal death or clone
    w[add] += (1 + dτ*(shift - diagonal_element(ham,add)))*num
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(::IsStochastic, w, ham::AbstractHamiltonian, add, num::Real,
                        shift, dτ)
    # version for single population of integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = offdiagonals(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagonal_element(ham,add)
    pd = dτ * (dME - shift)
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    w[add] += ndiags # should carry to correct sign
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

function fciqmc_col!(::DictVectors.IsStochastic2Pop, w, ham::AbstractHamiltonian, add,
                        cnum::Complex, cshift, dτ)
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
            w[naddress] += nspawns
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
            annihilations += min(abs(imag(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
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
    ndiags = convert(typeof(num),ndiag) # now complex integer type
    if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    end
    w[add] += ndiags # should carry the correct sign
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
    if sign(imag(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    end
    w[add] += ndiags # should carry the correct sign
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
    w[add] += cnspawn
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
    w[add] += cnspawn
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end


# function fciqmc_col!(::IsStochastic, w, ham::AbstractHamiltonian, add,
#                         tup::Tuple{Real,Real},
#                         shift, dτ)
#     # trying out Ali's suggestion with occupation ratio of neighbours
#     # off-diagonal: spawning psips
#     num = tup[1] # number of psips on configuration
#     occ_ratio= tup[2] # ratio of occupied vs total number of neighbours
#     spawns = deaths = clones = antiparticles = annihilations = zero(num)
#     hops = offdiagonals(ham,add)
#     for n in 1:abs(num) # for each psip attempt to spawn once
#         naddress, pgen, matelem = random_offdiagonal(hops)
#         pspawn = dτ * abs(matelem) /pgen # non-negative Float64
#         nspawn = floor(pspawn) # deal with integer part separately
#         cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
#         # at this point, nspawn is non-negative
#         # now converted to correct type and compute sign
#         nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
#         # - because Hamiltonian appears with - sign in iteration equation
#         wnapsips, wnaflag = w[naddress]
#         if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
#             annihilations += min(abs(wnapsips),abs(nspawns))
#         end
#         if !iszero(nspawns)
#             w[naddress] = (wnapsips+nspawns, wnaflag)
#             # perform spawn (if nonzero): add walkers with correct sign
#             spawns += abs(nspawns)
#         end
#     end
#     # diagonal death / clone
#     dME = diagonal_element(ham,add)
#     # modify shift locally according to occupation ratio of neighbouring configs
#     mshift = occ_ratio > 0 ? shift*occ_ratio : shift
#     pd = dτ * (dME - mshift) # modified
#     newdiagpop = (1-pd)*num
#     ndiag = trunc(newdiagpop)
#     abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
#     # only treat non-integer part stochastically
#     ndiags = convert(typeof(num),ndiag) # now appropriate type
#     wapsips, waflag = w[add]
#     if sign(wapsips) ≠ sign(ndiags) # record annihilations
#         annihilations += min(abs(wapsips),abs(ndiags))
#     end
#     w[add] = (wapsips + ndiags, waflag) # should carry to correct sign
#     if  pd < 0 # record event statistics
#         clones += abs(ndiags - num)
#     elseif pd < 1
#         deaths += abs(ndiags - num)
#     else
#         antiparticles += abs(ndiags)
#     end
#     return (spawns, deaths, clones, antiparticles, annihilations)
#     # note that w is not returned
# end # inner_step!


function fciqmc_col!(s::IsStochasticWithThreshold, w, ham::AbstractHamiltonian,
        add, val::N, shift, dτ) where N <: Real

    # diagonal death or clone: deterministic fomula
    # w[add] += (1 + dτ*(shift - diagonal_element(ham,add)))*val
    # projection to threshold should be applied after all colums are evaluated
    new_val = (1 + dτ*(shift - diagonal_element(ham,add)))*val
    # apply threshold if necessary
    if abs(new_val) < s.threshold
        # project stochastically to threshold
        # w[add] += (abs(new_val)/s.threshold > cRand()) ? sign(new_val)*s.threshold : 0
        w[add] += ifelse(cRand() < abs(new_val)/s.threshold, sign(new_val)*s.threshold, 0)
    else
        w[add] += new_val
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
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
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
        w[naddress] += nspawns
        # perform spawn (if nonzero): add walkers with correct sign
    end
    # done with stochastic spawning
    return (0, 0, 0, 0, 0)
end

# TODO this is here for testing pupropses. Should be deleted later on.
function fciqmc_col!(
    s::DictVectors.IsStochasticWithThresholdAndInitiator,
    w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    # Diagonal step:
    new_val = (1 + dτ*(shift - diagonal_element(ham,add))) * val
    w[add] += new_val
    absval = abs(val)

    # Initiator:
    if absval > s.initiator_threshold
        hops = offdiagonals(ham, add)
        if s.rel_threshold * val ≥ length(hops)
            # Exact multiplication
            factor = dτ * val
            for (new_add, mat_elem) in hops
                w[new_add] -= factor * mat_elem
            end
            return (1, 0, 0, 0, 0)
        else
            # Stochastic - do the integer part first
            for n in 1:floor(Int, absval)
                new_add, gen_prob, mat_elem = random_offdiagonal(hops)
                # if pspawn > 1, do exactly, otherwise round to 1
                pspawn = dτ * abs(mat_elem) / gen_prob
                sg = sign(val) * sign(mat_elem)
                if pspawn ≥ 1
                    w[new_add] -= pspawn * sg
                elseif cRand() < pspawn
                    w[new_add] -= sg
                end
            end
            # Take care of the leftovers
            rem_val = absval % 1
            new_add, gen_prob, mat_elem = random_offdiagonal(hops)
            pspawn = rem_val * dτ * abs(mat_elem) / gen_prob
            sg = sign(val) * sign(mat_elem)
            if pspawn ≥ 1
                w[new_add] -= pspawn * sg
            elseif cRand() < pspawn
                w[new_add] -= sg
            end
            return (0, 1, 0, 0, 0)
        end
    end

    return (0, 0, 1, 0, 0)
end

function fciqmc_col!(
    s::IsDynamicSemistochastic,
    w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    absval = abs(val)

    # Diagonal step:
    new_val = (1 + dτ*(shift - diagonal_element(ham, add))) * val
    w[add] += new_val

    hops = offdiagonals(ham, add)
    if s.rel_threshold * val ≥ length(hops) || val ≥ s.abs_threshold
        # Exact multiplication when conditions are met.
        factor = dτ * val
        for (new_add, mat_elem) in hops
            w[new_add] -= factor * mat_elem
        end
        return (1, 0, 0, 0, 0)
    else
        # Spawning without projection. It is done later on.
        remainder = absval % 1
        for i in 1:ceil(Int, absval)
            new_add, gen_prob, mat_elem = random_offdiagonal(hops)
            new_val = sign(val) * ifelse(i == 1, remainder, 1.0) * dτ * mat_elem / gen_prob
            w[new_add] -= new_val
        end
        return (0, 1, 0, 0, 0)
    end
end
