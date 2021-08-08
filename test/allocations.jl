using Rimu
using Test

@testset "Allocations" begin
    # The purpose of these tests is to find type instabilities that might appear as the
    # Julia compiler changes. If allocations suddenly increase by a lot, there is a good
    # chance that dynamic dispatch is happening somewhere in the code.
    b1 = nearUniform(BoseFS{10,10})
    b2 = nearUniform(BoseFS{50,50})
    b3 = nearUniform(BoseFS{100,100})

    for H in (
        HubbardReal1D(b1),
        HubbardReal1D(b2),
        HubbardReal1D(b3),

        HubbardMom1D(b1),
        HubbardMom1D(b2),
        HubbardMom1D(b3),

        ExtendedHubbardReal1D(b1),
        ExtendedHubbardReal1D(b2),
        ExtendedHubbardReal1D(b3),

        BoseHubbardReal1D2C(BoseFS2C(b1, b1)),
        BoseHubbardReal1D2C(BoseFS2C(b2, b2)),
        BoseHubbardReal1D2C(BoseFS2C(b3, b3)),

        BoseHubbardMom1D2C(BoseFS2C(b1, b1)),
        BoseHubbardMom1D2C(BoseFS2C(b2, b2)),
        BoseHubbardMom1D2C(BoseFS2C(b3, b3)),
    )
        @testset "Allocations for $(typeof(H))" begin
            for dv_type in (DVec, InitiatorDVec)
                dv = dv_type(starting_address(H) => 1.0, style=IsDynamicSemistochastic())

                lomc!(H, dv; dτ=1e-6)
                allocs = @allocated lomc!(H, dv; dτ=1e-6, laststep=200)
                @test allocs < 5e8 # 500MiB
            end
        end
    end
end
