using KrylovKit
using Rimu
using Test

"""
    function test_krylov_eigsolve(
        dvec_type, address_type;
        n=6, m=6, u=6.0, t=1.0, dvec_kwargs=(;)
    )

Run `eigsolve` from KrylovKit.jl on BoseHubbardReal1D and selected `dvec_type` and
`address_type`.

# Example

```jldoctest
julia> test_krylov_eigsolve(DVec, BSAdd64)
Test Passed
```
"""
function test_krylov_eigsolve(
    dvec_type, address_type; n=6, m=6, u=6.0, t=1.0, dvec_kwargs=(;)
)
    ham = BoseHubbardReal1D(; n=n, m=m, u=u, t=t, AT=address_type)

    a_init = nearUniform(ham)
    c_init = dvec_type(a_init => 1.0; capacity=ham(:dim), dvec_kwargs...)

    all_results = eigsolve(ham, c_init, 1, :SR; issymmetric = true)
    energy = all_results[1][1]

    @test energy ≈ -4.0215 atol=0.0001
end

for dvec_type in (DVec,), address_type in (BSAdd64, BSAdd128, BoseFS{6,6,BSAdd64})
    test_krylov_eigsolve(dvec_type, address_type)
end

BoseFS{6,6}
