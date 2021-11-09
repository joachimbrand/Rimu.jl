"""
    SingletSymmetry(ham::AbstractHamiltonian{T}; even=true) <: AbstractHamiltonian{T}

Impose singlet symmetry on all states and the Hamiltonian `ham`.

Does not yet work as expected:
```jldoctest
julia> ham=HubbardMom1D(FermiFS2C((0,1,1,1,0), (0,1,1,1,0));u=-1);

julia> size(Matrix(SingletSymmetry(ham)))
(10, 10)

julia> size(Matrix(ham))
(20, 20)

julia> E_0 = eigvals(Matrix(ham))[1]
-8.34874140660577

julia> E_0_SS = eigvals(Matrix(SingletSymmetry(ham)))[1]
-8.347120813182073

julia> E_0 ≈ E_0_SS
false
```

"""
struct SingletSymmetry{T,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{T}
    hamiltonian::H
end

function SingletSymmetry(hamiltonian)
    address = starting_address(hamiltonian)
    if !(address isa FermiFS2C)
        throw(ArgumentError("Starting address must be a `FermiFS2C`"))
    end
    f1, f2 = address.components
    if num_particles(f1) ≠ num_particles(f2)
        throw(ArgumentError("Starting address must have equal number of particles"))
    end
    return SingletSymmetry(hamiltonian)
end

function Base.show(io::IO, h::SingletSymmetry)
    print(io, "SingletSymmetry(", h.hamiltonian, ")")
end

LOStructure(h::Type{<:SingletSymmetry}) = AdjointUnknown()

function starting_address(h::SingletSymmetry)
    add, _ = singlet_canonify(starting_address(h.hamiltonian))
    return add
end

get_offdiagonal(h::SingletSymmetry, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::SingletSymmetry, add) = num_offdiagonals(h.hamiltonian, add)

struct SingletSymmetryOffdiagonals{
    A,T,O<:AbstractVector{Tuple{A,T}}
} <: AbstractOffdiagonals{A,T}
    od::O
end
Base.size(o::SingletSymmetryOffdiagonals) = size(o.od)

function offdiagonals(h::SingletSymmetry, add)
    return SingletSymmetryOffdiagonals(offdiagonals(h.hamiltonian, add))
end

function Base.getindex(o::SingletSymmetryOffdiagonals, i)
    add, val = o.od[i]
    can_add, _ = singlet_canonify(add)
    return can_add, val
end
function diagonal_element(h::SingletSymmetry, add)
    return diagonal_element(h.hamiltonian, add)
end

function singlet_canonify(c::FermiFS2C{N,N,<:Any,<:Any,F,F}) where {N, F}
    f1, f2 = c.components
    bs_and = f1.bs & f2.bs # extract modes with two fermions in it (already singlets)
    bs1_x = f1.bs ⊻ bs_and # extract loose fermions
    bs2_x = f2.bs ⊻ bs_and

    NN = num_particles(f1)
    om1_x = FermiOccupiedModes{NN, typeof(bs1_x)}(bs1_x) # N for type stability; ≤N particles
    om2_x = FermiOccupiedModes{NN, typeof(bs2_x)}(bs2_x) # N for type stability; ≤N particles

    bs1 = bs2 = bs_and
    sign = 1
    for (mode1, mode2) in zip(om1_x, om2_x)
        if mode1 < mode2 # fermions already in canonical order
            @inbounds bs1 |= mode1 # add back into bitstring
            @inbounds bs2 |= mode2
        else
            @inbounds bs1 |= mode2 # swap
            @inbounds bs2 |= mode1
            sign *= -1 # and remember sign change
        end
    end

    return FermiFS2C(F(bs1), F(bs2)), sign
end
