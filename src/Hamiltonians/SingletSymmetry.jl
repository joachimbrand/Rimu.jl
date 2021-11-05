"""
    SingletSymmetry(ham::AbstractHamiltonian{T}; even=true) <: AbstractHamiltonian{T}

Impose even or odd parity on all states and the Hamiltonian `ham` as controlled by the
keyword argument `even`. For some Hamiltonians, this reduces the size of the Hilbert space
by half.

`SingletSymmetry` changes the basis set to states with defined parity. Effectively, a
non-even address ``|α⟩`` is replaced by ``\\frac{1}{√2}(|α⟩ ± |ᾱ⟩)`` for even and odd
parity, respectively, where `ᾱ == reverse(α)`.

# Notes

* This modifier currently only works on [`starting_address`](@ref)s with an odd number of
  modes.
* For odd parity, the [`starting_address`](@ref) of the underlying Hamiltonian cannot be
  symmetric.
* If parity is not a symmetry of the Hamiltonian `ham` then the result is undefined.
* `SingletSymmetry` works by modifying the [`offdiagonals`](@ref) iterator.

```jldoctest
julia> ham = HubbardReal1D(BoseFS((0,2,1)))
HubbardReal1D(BoseFS{3,3}((0, 2, 1)); u=1.0, t=1.0)

julia> size(Matrix(ham))
(10, 10)

julia> size(Matrix(SingletSymmetry(ham)))
(6, 6)

julia> size(Matrix(SingletSymmetry(ham; odd=true)))
(4, 4)

julia> eigvals(Matrix(ham))[1] ≈ eigvals(Matrix(SingletSymmetry(ham)))[1]
true
```
"""
struct SingletSymmetry{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    hamiltonian::H
    even::Bool
end

function SingletSymmetry(hamiltonian; odd=false, even=!odd)
    address = starting_address(hamiltonian)
    if !isodd(num_modes(address))
        throw(ArgumentError("Starting address must have an odd number of modes"))
    end
    if !even && address == reverse(address)
        throw(ArgumentError("Even starting address can't be used with odd `SingletSymmetry`"))
    end
    return SingletSymmetry(hamiltonian, even)
end

function Base.show(io::IO, h::SingletSymmetry)
    print(io, "SingletSymmetry(", h.hamiltonian, ", even=", h.even, ")")
end

LOStructure(h::SingletSymmetry) = LOStructure(h.hamiltonian)
function starting_address(h::SingletSymmetry)
    add = starting_address(h.hamiltonian)
    return min(add, reverse(add))
end

function Base.adjoint(h::SingletSymmetry)
    return SingletSymmetry(adjoint(h.hamiltonian); even=h.even)
end

get_offdiagonal(h::SingletSymmetry, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::SingletSymmetry, add) = num_offdiagonals(h.hamiltonian, add)

struct ParitySymmetryOffdiagonals{
    A,T,O<:AbstractVector{Tuple{A,T}}
} <: AbstractOffdiagonals{A,T}
    od::O
    even::Bool
end
Base.size(o::ParitySymmetryOffdiagonals) = size(o.od)

function offdiagonals(h::SingletSymmetry, add)
    return ParitySymmetryOffdiagonals(offdiagonals(h.hamiltonian, add), h.even)
end

function Base.getindex(o::ParitySymmetryOffdiagonals, i)
    add, val = o.od[i]
    can_add, sign = singlet_canonify(add)
    return can_add, val*sign
end
function diagonal_element(h::SingletSymmetry, add)
    return diagonal_element(h.hamiltonian, add)
end

function singlet_canonify(c::FermiFS2C{N,N,<:Andy,<:Any,F,F}) where {N, F}
    f1, f2 = c.components
    bs_and = f1.bs & f2.bs # extract modes with two fermions in it (already singlets)
    bs1_x = f1.bs ⊻ bs_and # extract loose fermions
    bs2_x = f2.bs ⊻ bs_and

    om1_x = FermiOccupiedModes{N, typeof(bs1_x)}(bs1_x) # N for type stability; ≤N particles
    om2_x = FermiOccupiedModes{N, typeof(bs2_x)}(bs2_x) # N for type stability; ≤N particles

    # om1, om2 = occupied_modes.(c.components)
    # # omm1, omm2 = OccupiedModeMap.(c.components)
    # T = eltype(om1)
    # onr1 = MVector{min(M),T}(undef)
    # onr2 = MVector{min(M),T}(undef)
    bs1 = bs2 = bs_and
    sign = 1
    for (mode1, mode2) in zip(om1_x, om2_x)
        if mode1 < mode2 # fermions already in canonical order
            bs1 |= mode1 # add back into bitstring
            bs2 |= mode2
        else
            bs1 |= mode2 # swap
            bs2 |= mode1
            sign *= -1 # and remember sign change
        end
    end

    return FermiFS2C(F(bs1), F(bs2)), sign
end
