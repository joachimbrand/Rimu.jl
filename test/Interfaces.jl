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
    @test offdiagonals(ham, 1) == [2 => 2, 3 => 5]
    @test offdiagonals(ham, 2) == [3 => 6]
    @test offdiagonals(ham, 3) == []

    @test num_offdiagonals(ham, 1) == 2
    @test num_offdiagonals(ham, 2) == 1
    @test num_offdiagonals(ham, 3) == 0

    @test get_offdiagonal(ham, 1, 1) == (2 => 2)
    @test get_offdiagonal(ham, 1, 2) == (3 => 5)
    @test get_offdiagonal(ham, 2, 1) == (3 => 6)

    @test starting_address(ham) == 3

    @test LOStructure(ham) == AdjointKnown()
    @test has_adjoint(ham)
end
@testset "lomc! with matrix" begin
    ham = [1 1 2 3 2;
           2 0 2 2 3;
           0 0 0 3 2;
           0 0 1 1 2;
           0 1 0 1 0]
    vector = ones(5)
    post_step = ProjectedEnergy(ham, vector)
    df, st = lomc!(ham, vector; laststep=10_000, post_step)
    eigs = eigen(ham)
    @test df.shift[end] ≈ eigs.values[1] rtol=0.01
    @test df.hproj[end] / df.vproj[end] ≈ eigs.values[1] rtol=0.01
    @test normalize(st.replicas[1].v) ≈ eigs.vectors[:, 1] rtol=0.01
end
