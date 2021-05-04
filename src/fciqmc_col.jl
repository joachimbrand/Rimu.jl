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
function fciqmc_col!(w, ham, add, num, shift, dτ)
    if valtype(w) <: Real
        return fciqmc_col!(StochasticStyle(w), w, ham, add, num, real(shift), dτ)
    else
        return fciqmc_col!(StochasticStyle(w), w, ham, add, num, shift, dτ)
    end
end

function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    for (i, val) in enumerate(ham[:, add])
        spawn!(w, i, (1 + dτ * (shift - val)) * num, add => num)
    end
    # todo: return something sensible
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(::IsDeterministic, w, ham::AbstractHamiltonian, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in offdiagonals(ham, add)
        spawn!(w, nadd, -dτ * elem * num, add => num)
    end
    # diagonal death or clone
    spawn!(w, add, (1 + dτ*(shift - diagonal_element(ham,add)))*num, add => num)
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
        # TODO: recording stats should be done at the WorkingMemory level
        #if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
        #    annihilations += min(abs(w[naddress]),abs(nspawns))
        #end
        if !iszero(nspawns)
            spawn!(w, naddress, nspawns, add => num)
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
    #if sign(w[add]) ≠ sign(ndiags) # record annihilations
    #    annihilations += min(abs(w[add]),abs(ndiags))
    #end
    spawn!(w, add, ndiags, add => num) # should carry to correct sign
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
        #if sign(real(w[naddress])) * sign(nspawns) < 0 # record annihilations
        #    annihilations += min(abs(real(w[naddress])),abs(nspawns))
        #end
        if !iszero(nspawns)
            spawn!(w, naddress, nspawns, add => num)
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
        #if sign(imag(w[naddress])) * sign(imag(nspawns)) < 0 # record annihilations
        #    annihilations += min(abs(imag(w[naddress])),abs(nspawns))
        #end
        if !iszero(nspawns)
            spawn!(w, naddress, nspawns, add => num)
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
    #if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
    #    annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    #end
    spawn!(w, add, ndiags, add => num) # should carry the correct sign
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
    #if sign(imag(w[add])) ≠ sign(ndiag) # record annihilations
    #    annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    #end
    spawn!(w, add, ndiags, add => num) # should carry the correct sign
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
    #if sign(real(w[add])) * sign(nspawn) < 0 # record annihilations
    #    annihilations += min(abs(real(w[add])),abs(nspawn))
    #end
    spawn!(w, add, cnspawn, add => num)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    # imag part - to be spawned into imaginary walkers
    ispawn = imag(cspawn) # float with sign
    nspawn = trunc(ispawn) # deal with integer part separately
    cRand() < abs(ispawn - nspawn) && (nspawn += sign(ispawn)) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
    #if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
    #    annihilations += min(abs(imag(w[add])),abs(nspawn))
    #end
    spawn!(w, add, cnspawn, add => num)
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function fciqmc_col!(::DictVectors.IsStochastic2PopInitiator, w, ham::AbstractHamiltonian,
                        add, cnum::Complex, cshift, dτ)
    # version for complex integer psips with initiator approximation
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
        #if sign(real(w[naddress])) * sign(nspawns) < 0 # record annihilations
        #    annihilations += min(abs(real(w[naddress])),abs(nspawns))
        #end
        if !iszero(nspawns)
            spawn!(w, naddress, nspawns, add => num)
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
        #if sign(imag(w[naddress])) * sign(imag(nspawns)) < 0 # record annihilations
        #    annihilations += min(abs(imag(w[naddress])),abs(nspawns))
        #end
        if !iszero(nspawns)
            spawn!(w, naddress, nspawns, add => num)
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
    #if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
    #    annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    #end
    spawn!(w, add, ndiags, add => num) # should carry the correct sign
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
    #if sign(imag(w[add])) ≠ sign(ndiag) # record annihilations
    #    annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    #end
    spawn!(w, add, ndiags, add => num) # should carry the correct sign
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
    # TODO: this is broken
    if real(w[add]) ≠ 0 # only spawn into occupied sites (initiator approximation)
        rspawn = real(cspawn) # float with sign
        nspawn = trunc(rspawn) # deal with integer part separately
        cRand() < abs(rspawn - nspawn) && (nspawn += sign(rspawn)) # random spawn
        # at this point, nspawn has correct sign
        # now convert to correct type
        cnspawn = convert(typeof(cnum), nspawn)
        if sign(real(w[add])) * sign(nspawn) < 0 # record annihilations
            annihilations += min(abs(real(w[add])),abs(nspawn))
        end
        spawn!(w, add, cnspawn, add => num)
        # perform spawn (if nonzero): add walkers with correct sign
        spawns += abs(nspawn)
    end

    # imag part - to be spawned into imaginary walkers
    if imag(w[add]) ≠ 0 # only spawn into occupied sites (initiator approximation)
        ispawn = imag(cspawn) # float with sign
        nspawn = trunc(ispawn) # deal with integer part separately
        cRand() < abs(ispawn - nspawn) && (nspawn += sign(ispawn)) # random spawn
        # at this point, nspawn has correct sign
        # now convert to correct type
        cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
        if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
            annihilations += min(abs(imag(w[add])),abs(nspawn))
        end
        spawn!(w, add, cnspawn, add => num)
        # perform spawn (if nonzero): add walkers with correct sign
        spawns += abs(nspawn)
    end

    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function fciqmc_col!(nl::DictVectors.IsStochasticNonlinear, w, ham::AbstractHamiltonian,
                        add, num::Real,
                        shift, dτ)
    # version for single population of integer psips
    # Nonlinearity in diagonal death step according to Ali's suggestion
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
            spawn!(w, naddress, nspawns, add => num)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagonal_element(ham,add)
    shifteff = shift*(1 - exp(-num/nl.c))
    pd = dτ * (dME - shifteff)
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    spawn!(w, add, ndiags, add => num) # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end # inner_step!

function fciqmc_col!(::IsStochastic, w, ham::AbstractHamiltonian, add,
                        tup::Tuple{Real,Real},
                        shift, dτ)
    # trying out Ali's suggestion with occupation ratio of neighbours
    # off-diagonal: spawning psips
    num = tup[1] # number of psips on configuration
    occ_ratio= tup[2] # ratio of occupied vs total number of neighbours
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
        wnapsips, wnaflag = w[naddress]
        if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(wnapsips),abs(nspawns))
        end
        if !iszero(nspawns)
            spawn!(w, naddress, (wnapsips+nspawns, wnaflag), add => num)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagonal_element(ham,add)
    # modify shift locally according to occupation ratio of neighbouring configs
    mshift = occ_ratio > 0 ? shift*occ_ratio : shift
    pd = dτ * (dME - mshift) # modified
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now appropriate type
    wapsips, waflag = w[add]
    if sign(wapsips) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(wapsips),abs(ndiags))
    end
    spawn!(w, add, (wapsips + ndiags, waflag), add => num) # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end # inner_step!

function fciqmc_col!(s::DictVectors.IsSemistochastic, w, ham::AbstractHamiltonian, add,
         val_flag_tuple::Tuple{N, F}, shift, dτ) where {N<:Number, F<:Integer}
    (val, flag) = val_flag_tuple
    deterministic = flag & one(F) # extract deterministic flag
    # diagonal death or clone
    new_val = w[add][1] + (1 + dτ*(shift - diagonal_element(ham,add)))*val
    if deterministic
        spawn!(w, add, (new_val, flag), add => num) # new tuple
    else
        if new_val < s.threshold
            if new_val/s.threshold > cRand()
                new_val = convert(N,s.threshold)
                spawn!(w, add, (new_val, flag), add => num) # new tuple
            end
            # else # do nothing, stochastic space and rounded to zero
        else
            spawn!(w, add, (new_val, flag), add => num) # new tuple
        end
    end
    # off-diagonal: spawning psips
    if deterministic
        for (nadd, elem) in offdiagonals(ham, add)
            wnapsips, wnaflag = w[nadd]
            if wnaflag & one(F) # new address `nadd` is also in deterministic space
                spawn!(w, nadd, (wnapsips - dτ * elem * val, wnaflag), add => num)
            else
                # TODO: det -> sto
                pspawn = abs(val * dτ * matelem) # non-negative Float64, pgen = 1
                nspawn = floor(pspawn) # deal with integer part separately
                cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
                # at this point, nspawn is non-negative
                # now converted to correct type and compute sign
                nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
                # - because Hamiltonian appears with - sign in iteration equation
                if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
                    annihilations += min(abs(wnapsips),abs(nspawns))
                end
                if !iszero(nspawns) # successful attempt to spawn
                    spawn!(w, naddress, (wnapsips+nspawns, wnaflag), add => num)
                    # perform spawn (if nonzero): add walkers with correct sign
                    spawns += abs(nspawns)
                end
            end
        end
    else
        # TODO: stochastic
        hops = offdiagonals(ham, add)
        for n in 1:floor(abs(val)) # abs(val÷s.threshold) # for each psip attempt to spawn once
            naddress, pgen, matelem = random_offdiagonal(hops)
            pspawn = dτ * abs(matelem) /pgen # non-negative Float64
            nspawn = floor(pspawn) # deal with integer part separately
            cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
            # at this point, nspawn is non-negative
            # now converted to correct type and compute sign
            nspawns = convert(typeof(val), -nspawn * sign(val) * sign(matelem))
            # - because Hamiltonian appears with - sign in iteration equation
            wnapsips, wnaflag = w[naddress]
            if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
                annihilations += min(abs(wnapsips),abs(nspawns))
            end
            if !iszero(nspawns)
                spawn!(w, naddress, (wnapsips+nspawns, wnaflag), add => num)
                # perform spawn (if nonzero): add walkers with correct sign
                spawns += abs(nspawns)
            end
        end
        # deal with non-integer remainder
        rval =  abs(val%1) # abs(val%threshold)
        naddress, pgen, matelem = random_offdiagonal(hops)
        pspawn = rval * dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(val), -nspawn * sign(val) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        wnapsips, wnaflag = w[naddress]
        if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(wnapsips),abs(nspawns))
        end
        if !iszero(nspawns)
            spawn!(w, naddress, (wnapsips+nspawns, wnaflag), add => num)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
        # done with stochastic spawning
    end
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(s::IsStochasticWithThreshold, w, ham::AbstractHamiltonian,
        add, val::N, shift, dτ) where N <: Real

    # diagonal death or clone: deterministic fomula
    # spawn!(w, add,  (1 + dτ*(shift - diagonal_element(ham,add)))*val, add => val)
    # projection to threshold should be applied after all colums are evaluated
    new_val = (1 + dτ*(shift - diagonal_element(ham,add)))*val
    # apply threshold if necessary
    if abs(new_val) < s.threshold
        # project stochastically to threshold
        # spawn!(w, add, (abs(new_val)/s.threshold > cRand()) ? sign(new_val)*s.threshold : 0, add => val)
        spawn!(w, add, ifelse(cRand() < abs(new_val)/s.threshold, sign(new_val)*s.threshold, 0), add => val)
    else
        spawn!(w, add, new_val, add => val)
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
            spawn!(w, naddress, nspawns, add => val)
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
        spawn!(w, naddress, nspawns, add => val)
        # perform spawn (if nonzero): add walkers with correct sign
    end
    # done with stochastic spawning
    return (0, 0, 0, 0, 0)
end

"""
    threshold_projected_spawn!
This function performs threshold projection before spawning, but only for
`IsDynamicSemistochastic` with the `project_later` parameter set to `false`.
"""
function threshold_projected_spawn!(s::IsDynamicSemistochastic{false}, w, add, val, p)
    threshold = s.proj_threshold
    absval = abs(val)
    if absval < threshold
        if cRand() < abs(val) / threshold
            spawn!(w, add, sign(val) * threshold, p)
        end
    else
        spawn!(w, add, val, p)
    end
    return nothing
end
function threshold_projected_spawn!(::IsDynamicSemistochastic{true}, w, add, val, p)
    spawn!(w, add, val, p)
end

function fciqmc_col!(
    s::IsDynamicSemistochastic,
    w, ham::AbstractHamiltonian, add, val, shift, dτ,
)
    absval = abs(val)

    # Diagonal step:
    new_val = (1 + dτ*(shift - diagonal_element(ham, add))) * val
    threshold_projected_spawn!(s, w, add, new_val, add => val)

    hops = offdiagonals(ham, add)
    if s.rel_threshold * val ≥ length(hops) || val ≥ s.abs_threshold
        # Exact multiplication when conditions are met.
        factor = dτ * val
        for (new_add, mat_elem) in hops
            threshold_projected_spawn!(s, w, new_add, -factor * mat_elem, add => val)
        end
        return (1, 0, 0, 0, 0)
    else
        # Spawning without projection. It is done later on.
        remainder = absval % 1
        for i in 1:ceil(Int, absval)
            new_add, gen_prob, mat_elem = random_offdiagonal(hops)
            new_val = sign(val) * ifelse(i == 1, remainder, 1.0) * dτ * mat_elem / gen_prob
            threshold_projected_spawn!(s, w, new_add, -new_val, add => val)
        end
        return (0, 1, 0, 0, 0)
    end
end
