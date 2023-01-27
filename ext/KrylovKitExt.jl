module KrylovKitExt

using Rimu
using KrylovKit
using LinearAlgebra

const U = Union{Symbol,EigSorter}

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, vec::PDVec, howmany::Int=1, which::U=:LR; kwargs...
)
    prop = DictVectors.OperatorMulPropagator(ham, vec)
    return eigsolve(
        prop, vec, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, vec::AbstractDVec, howmany::Int=1, which::U=:LR; kwargs...
)
    return @invoke eigsolve(
        ham::Any, vec::Any, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

end
