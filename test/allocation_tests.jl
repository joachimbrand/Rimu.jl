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
        hamname = string(nameof(typeof(H)), "(", num_modes(add), "): ")
        @testset "Allocations for $(hamname)" begin
            for dv_type in (DVec, InitiatorDVec)
                dτ = if num_modes(add) == 10
                    1e-3
                elseif num_modes(add) == 50
                    1e-4
                else
                    1e-6
                end

                dv = dv_type(add => 1.0, style=IsDynamicSemistochastic())
                sizehint!(dv, 50_000)

                # Warmup for lomc!
                _, st = lomc!(H, dv; dτ, threading=false, maxlength=10_000)

                r = only(st.replicas)
                p = r.params

                # Warmup for step!
                fciqmc_step!(H, r.v, p.shift, 1e-4, r.pnorm, r.w)
                fciqmc_step!(H, r.v, p.shift, 1e-4, r.pnorm, r.w)
                fciqmc_step!(H, r.v, p.shift, 1e-4, r.pnorm, r.w)
                fciqmc_step!(H, r.v, p.shift, 1e-4, r.pnorm, r.w)

                allocs_step = @allocated fciqmc_step!(H, r.v, p.shift, 1e-4, r.pnorm, r.w)
                @test allocs_step ≤ 512

                allocs_full = @allocated lomc!(
                    H, dv; dτ, laststep=200, threading=false, maxlength=10_000
                )
                @test allocs_full ≤ 1e8 # 100MiB

                # Print out the results to make it easier to find problems.
                print(rpad(hamname, 40))
                print("per step ", allocs_step, ", full ", allocs_full/(1024^2), "M\n")
            end
        end
    end
end
