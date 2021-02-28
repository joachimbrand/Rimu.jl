struct HubbardMom1D{TT,U,T,N,M,AD} <: AbstractHamiltonian{TT}
    add::AD # default starting address, should have N particles and M modes
    ks::SVector{M,TT} # values for k
    kes::SVector{M,TT} # values for kinetic energy
end

@doc """
    HubbardMom1D(add::BoseFS; u=1.0, t=1.0)
Implements a one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = -t \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}\\\\
ϵ_k = - 2 t \\cos(k)
```

# Arguments
- `add::BoseFS`: bosonic starting address, defines number of particles and sites
- `u::Float64`: the interaction parameter
- `t::Float64`: the hopping strength

# Functor use:
    w = ham(v)
    ham(w, v)
Compute the matrix - vector product `w = ham * v`. The two-argument version is
mutating for `w`.

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" HubbardMom1D

# constructors
function HubbardMom1D(add::BoseFS{N,M,A}; u::TT=1.0, t::TT=1.0) where {N, M, TT, A}
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    kes = SVector{M}(-2*cos.(kr))
    return HubbardMom1D{TT,u,t,N,M,BoseFS{N,M,A}}(add, ks, kes)
end
# allow passing the N and M parameters for compatibility with show()
function HubbardMom1D{N,M}(add::BoseFS{N,M,A}; u::TT=1.0, t::TT=1.0) where {N, M, TT, A}
    return HubbardMom1D(add; u=u, t=t)
end

# display in a way that can be used as constructor
function Base.show(io::IO, h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD}
    print(io, "HubbardMom1D{$N,$M}(")
    show(io, h.add)
    print(io, "; u=$U, t=$T)")
end

Base.eltype(::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD} = TT

# set the `LOStructure` trait
LOStructure(::Type{HubbardMom1D{TT,U,T,N,M,AD}}) where {TT<:Real,U,T,N,M,AD} = HermitianLO()

# functor definitions need to be done separately for each concrete type
function (h::HubbardMom1D)(s::Symbol)
  if s == :dim # attempt to compute dimension as `Int`
      return hasIntDimension(h) ? dimensionLO(h) : nothing
  elseif s == :fdim
      return fDimensionLO(h) # return dimension as floating point
  end
  return nothing
end
# should be all that is needed to make the Hamiltonian a linear map:
ks(h::HubbardMom1D) = h.ks

# standard interface function
function numOfHops(ham::HubbardMom1D, add)
  nSandD = numSandDoccupiedsites(add)
  return numOfHops(ham, add, nSandD)
end

# 3-argument version
@inline function numOfHops(ham::HubbardMom1D{TT,U,T,N,M,AD}, add, nSandD) where {TT,U,T,N,M,AD}
  singlies, doublies = nSandD
  return singlies*(singlies-1)*(M - 2) + doublies*(M - 1)
  # number of excitations that can be made
end

@inline function interaction_energy_diagonal(h::HubbardMom1D{TT,U,T,N,M,AD},
        onrep::StaticVector) where {TT,U,T,N,M,AD<:BoseFS}
    # now compute diagonal interaction energy
    onproduct = 0 # Σ_kp < c^†_p c^†_k c_k c_p >
    for p = 1:M
        @inbounds onproduct += onrep[p] * (onrep[p] - 1)
        @inbounds @simd for k = 1:p-1
            onproduct += 4*onrep[k]*onrep[p]
        end
    end
    # @show onproduct
    return U / 2M * onproduct
end

function kinetic_energy(h::HubbardMom1D, add::AbstractFockAddress)
    onrep = BitStringAddresses.m_onr(add) # get occupation number representation
    return kinetic_energy(h, onrep)
end

@inline function kinetic_energy(h::HubbardMom1D, onrep::StaticVector)
    return h.kes⋅onrep # safe as onrep is Real
end

@inline function diagME(h::HubbardMom1D, add)
    onrep = BitStringAddresses.m_onr(add) # get occupation number representation
    return diagME(h, onrep)
end

@inline function diagME(h::HubbardMom1D, onrep::StaticVector)
    return kinetic_energy(h, onrep) + interaction_energy_diagonal(h, onrep)
end

@inline function hop(ham::HubbardMom1D{TT,U,T,N,M,AD}, add::AD, chosen::Number) where {TT,U,T,N,M,AD}
    hop(ham, add, chosen, numSandDoccupiedsites(add))
end

@inline function hop_old(ham::HubbardMom1D{TT,U,T,N,M,AD}, add::AD, chosen::Number, nSD) where {TT,U,T,N,M,AD}
  onrep =  BitStringAddresses.m_onr(add)
  # get occupation number representation as a mutable array
  singlies, doublies = nSD # precomputed `numSandDoccupiedsites(add)`
  onproduct = 1
  k = p = q = 0
  double = chosen - singlies*(singlies-1)*(M - 2)
  # start by making holes as the action of two annihilation operators
  if double > 0 # need to choose doubly occupied site for double hole
    # c_p c_p
    double, q = fldmod1(double, M-1)
    # double is location of double
    # q is momentum transfer
    for (i, occ) in enumerate(onrep)
      if occ > 1
        double -= 1
        if double == 0
          onproduct *= occ*(occ-1)
          onrep[i] = occ-2 # annihilate two particles in onrep
          p = k = i # remember where we make the holes
          break # should break out of the for loop
        end
      end
    end
  else # need to punch two single holes
    # c_k c_p
    pair, q = fldmod1(chosen, M-2) # floored integer division and modulus in ranges 1:(m-1)
    first, second = fldmod1(pair, singlies-1) # where the holes are to be made
    if second < first # put them in ascending order
      f_hole = second
      s_hole = first
    else
      f_hole = first
      s_hole = second + 1 # as we are counting through all singlies
    end
    counter = 0
    for (i, occ) in enumerate(onrep)
      if occ > 0
        counter += 1
        if counter == f_hole
          onproduct *= occ
          onrep[i] = occ -1 # punch first hole
          p = i # location of first hole
        elseif counter == s_hole
          onproduct *= occ
          onrep[i] = occ -1 # punch second hole
          k = i # location of second hole
          break
        end
      end
    end
    # we have p<k and 1 < q < ham.m - 2
    if q ≥ k-p
      q += 1 # to avoid putting particles back into the holes
    end
  end # if double > 0 # we're done punching holes

  # now it is time to deal with two creation operators
  # c^†_k-q
  kmq = mod1(k-q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[kmq]
  onproduct *= occ + 1
  onrep[kmq] = occ + 1
  # c^†_p+q
  ppq = mod1(p+q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[ppq]
  onproduct *= occ + 1
  onrep[ppq] = occ + 1

  return AD(onrep), U/(2*M)*sqrt(onproduct)
  # return new address and matrix element
end

# a non-allocating version of hop()
@inline function hop(ham::HubbardMom1D{TT,U,T,N,M,AD}, add::AD, chosen::Number, nSD) where {TT,U,T,N,M,AD}
  onrep = onr(add)
  # get occupation number representation as a static array
  singlies, doublies = nSD # precomputed `numSandDoccupiedsites(add)`
  onproduct = 1
  k = p = q = 0
  double = chosen - singlies*(singlies-1)*(M - 2)
  # start by making holes as the action of two annihilation operators
  if double > 0 # need to choose doubly occupied site for double hole
    # c_p c_p
    double, q = fldmod1(double, M-1)
    # double is location of double
    # q is momentum transfer
    for (i, occ) in enumerate(onrep)
      if occ > 1
        double -= 1
        if double == 0
          onproduct *= occ*(occ-1)
          onrep = @set onrep[i] = occ-2
          # annihilate two particles in onrep
          p = k = i # remember where we make the holes
          break # should break out of the for loop
        end
      end
    end
  else # need to punch two single holes
    # c_k c_p
    pair, q = fldmod1(chosen, M-2) # floored integer division and modulus in ranges 1:(m-1)
    first, second = fldmod1(pair, singlies-1) # where the holes are to be made
    if second < first # put them in ascending order
      f_hole = second
      s_hole = first
    else
      f_hole = first
      s_hole = second + 1 # as we are counting through all singlies
    end
    counter = 0
    for (i, occ) in enumerate(onrep)
      if occ > 0
        counter += 1
        if counter == f_hole
          onproduct *= occ
          onrep = @set onrep[i] = occ-1
          # punch first hole
          p = i # location of first hole
        elseif counter == s_hole
          onproduct *= occ
          onrep = @set onrep[i] = occ-1
          # punch second hole
          k = i # location of second hole
          break
        end
      end
    end
    # we have p<k and 1 < q < ham.m - 2
    if q ≥ k-p
      q += 1 # to avoid putting particles back into the holes
    end
  end # if double > 0 # we're done punching holes

  # now it is time to deal with two creation operators
  # c^†_k-q
  kmq = mod1(k-q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[kmq]
  onproduct *= occ + 1
  onrep = @set onrep[kmq] = occ + 1
  # c^†_p+q
  ppq = mod1(p+q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[ppq]
  onproduct *= occ + 1
  onrep = @set onrep[ppq] = occ + 1

  return AD(onrep), U/(2*M)*sqrt(onproduct)
  # return new address and matrix element
end

function hasIntDimension(h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD<:BoseFS}
  try
    binomial(N + M - 1, N)# formula for boson Hilbert spaces
    return true
  catch
    false
  end
end

function dimensionLO(h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD<:BoseFS}
    return binomial(N + M - 1, N) # formula for boson Hilbert spaces
end

function fDimensionLO(h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD<:BoseFS}
  fbinomial(N + M - 1, N) # formula for boson Hilbert spaces
  # NB: returns a Float64
end #dimHS

function Hops(ham::O, add::AD) where {TT,U,T,N,M,AD, O<:HubbardMom1D{TT,U,T,N,M,AD}}
    nSandD = numSandDoccupiedsites(add)::Tuple{Int64,Int64}
    # store this information for reuse
    nH = numOfHops(ham, add, nSandD)
    return Hops{TT,AD,O,Tuple{Int64,Int64}}(ham, add, nH, nSandD)
end

function Base.getindex(s::Hops{T,A,O,I}, i::Int) where {T,A,O<:HubbardMom1D,I}
    nadd, melem = hop(s.h, s.add, i, s.info)
    return (nadd, melem)
end #  returns tuple (newaddress, matrixelement)
