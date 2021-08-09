using Rimu
using Rimu: fciqmc_step!
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
        add = starting_address(H)
        for dv_type in (DVec, InitiatorDVec)
            for style in (
                IsDeterministic(),
                IsStochasticInteger(),
                IsStochasticWithThreshold(),
                IsDynamicSemistochastic(),
            )
                hamname = string(
                    nameof(typeof(H)), "(", num_modes(add), ")/", nameof(typeof(style))
                )
                @testset "Allocations for $(hamname)" begin
                    dτ = if num_modes(add) == 10
                        1e-4
                    elseif num_modes(add) == 50
                        1e-4
                    else
                        1e-6
                    end

                    dv = dv_type(add => 1.0, style=IsDynamicSemistochastic())
                    sizehint!(dv, 500_000)

                    # Warmup for lomc!
                   params = RunTillLastStep(shift = diagonal_element(H, add), dτ)
                    _, st = lomc!(H, dv; params, threading=false, maxlength=10_000, laststep=1)

                    r = only(st.replicas)
                    p = r.params

                    # Warmup for step!
                    fciqmc_step!(H, r.v, p.shift, dτ, r.pnorm, r.w)
                    fciqmc_step!(H, r.v, p.shift, dτ, r.pnorm, r.w)
                    fciqmc_step!(H, r.v, p.shift, dτ, r.pnorm, r.w)
                    fciqmc_step!(H, r.v, p.shift, dτ, r.pnorm, r.w)

                    allocs_step = @allocated fciqmc_step!(H, r.v, p.shift, dτ, r.pnorm, r.w)
                    @test allocs_step ≤ 512

                    dv = dv_type(add => 1.0, style=IsDynamicSemistochastic())
                    allocs_full = @allocated lomc!(
                        H, dv; dτ, laststep=200, threading=false, maxlength=10_000
                    )
                    @test allocs_full ≤ 1e8 # 100MiB

                    # Print out the results to make it easier to find problems.
                    print(rpad(hamname, 50))
                    print(": per step ", allocs_step, ", full ", allocs_full/(1024^2), "M\n")
                end
            end
        end
    end
end
