"""
    MomentumMom1D(ham::AbstractHamiltonian) <: AbstractHamiltonian
Momentum as a linear operator in Fock space. Pass a Hamiltonian `ham` in order to convey information about the Fock basis.

Example use:
```julia
add = BoseFS((1,0,2,1,2,1,1,3)) # address for a Fock state (configuration) with 11 bosons in 8 modes
ham = BoseHubbardMom1D(add; u = 2.0, t = 1.0)
mom = Momentum(ham) # create an instance of the momentum operator
diagME(mom, add) # 10.996 - to calculate the momentum of a single configuration
v = DVec(Dict(add => 10), 1000)
rayleigh_quotient(mom, v) # 10.996 - momentum expectation value for state vector `v`
```
"""
struct MomentumMom1D{H,T} <: AbstractHamiltonian{T}
    ham::H
end
LOStructure(::Type{MomentumMom1D{H,T}}) where {H,T <: Real} = HermitianLO()
numOfHops(ham::MomentumMom1D, add) = 0
diagME(mom::MomentumMom1D, add) = mod1(onr(add)⋅ks(mom.ham) + π, 2π) - π # fold into (-π, π]

# TODO
MomentumMom1D(ham::AbstractHamiltonian{T}) where T = MomentumMom1D{typeof(ham),T}(ham)
