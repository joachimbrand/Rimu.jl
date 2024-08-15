using LinearAlgebra
using Rimu
using Test

@testset "Interface basics" begin
    @test eltype(StyleUnknown{String}()) == String
    @test StochasticStyle(['a', 'b']) == StyleUnknown{Char}()

    vector = [1, 2, 3]
    deposit!(vector, 1, 1, 1 => 1)
    @test vector == [2, 2, 3]
    @test storage(vector) ≡ vector
    @test localpart(vector) ≡ vector

    zerovector!(vector)
    @test vector == [0, 0, 0]

    ham = [1 0 0; 2 3 0; 5 6 0]
    @test offdiagonals(ham, 1) == [(2, 2), (3, 5)]
    @test offdiagonals(ham, 2) == [(3, 6)]
    @test offdiagonals(ham, 3) == []

    @test num_offdiagonals(ham, 1) == 2
    @test num_offdiagonals(ham, 2) == 1
    @test num_offdiagonals(ham, 3) == 0

    @test get_offdiagonal(ham, 1, 1) == (2, 2)
    @test get_offdiagonal(ham, 1, 2) == (3, 5)
    @test get_offdiagonal(ham, 2, 1) == (3, 6)

    @test starting_address(ham) == 3

    @test LOStructure(ham) == AdjointKnown()
    @test has_adjoint(ham)

    @test_throws ArgumentError Interfaces.dot_from_right(1, 2, 3)
end

# using lomc! with a matrix was removed in Rimu.jl v0.12.0
@testset "lomc! with matrix" begin
    ham = [1 1 2 3 2;
           2 0 2 2 3;
           0 0 0 3 2;
           0 0 1 1 2;
           0 1 0 1 0]
    vector = ones(5)
    @test_throws ArgumentError lomc!(ham, vector; laststep=10_000)

    # rephrase with MatrixHamiltonian
    mh = MatrixHamiltonian(ham)
    sv = DVec(pairs(vector))
    post_step_strategy = ProjectedEnergy(mh, sv)

    # solve with new API
    p = ProjectorMonteCarloProblem(mh; start_at=sv, last_step=10_000, post_step_strategy)
    sm = solve(p)
    last_shift = DataFrame(sm).shift[end]

    # solve with old API
    df, _ = lomc!(mh, sv; laststep=10_000, post_step_strategy)
    eigs = eigen(ham)

    @test eigs.values[1] ≈ last_shift rtol = 0.01
    @test df.shift[end] ≈ eigs.values[1] rtol=0.01
    @test df.hproj[end] / df.vproj[end] ≈ eigs.values[1] rtol=0.01
    @test normalize(state_vectors(sm)[1]) ≈ DVec(pairs(eigs.vectors[:, 1])) rtol = 0.01
end
