"""
    growth_witness(shift::AbstractArray, norm::AbstractArray, dt, [b]; skip=0)

Compute the growth witness
```math
G^{(n)} = S^{(n)} - \\frac{\\vert\\mathbf{c}^{(n+1)}\\vert -
          \\vert\\mathbf{c}^{(n)}\\vert}{\\vert\\mathbf{c}^{(n)}\\vert d\\tau},
```
where `S` is the `shift` and \$\\vert\\mathbf{c}^{(n)}\\vert ==\$ `norm[n, 1]`.
Setting `b ≥ 1` a sliding average over `b` time steps is computed using
[`smoothen()`](@ref). The first `skip` time steps are skipped.
`mean(growth_witness)` is approximately the same as [`growth_estimator`](@ref) with `h=0`.

See also [`growth_estimator`](@ref).
"""
function growth_witness(shift::AbstractArray, norm::AbstractArray, dt; skip = 0)
    l = length(norm)
    @assert length(shift) == l "`shift` and `norm` arrays need to have the same length."
    l =  l-skip
    @assert l > 0 "`skip` must be larger than length of `shift` and `norm`"
    n = l - 1
    g = Vector{eltype(shift)}(undef, l)
    for i in 1:n
        g[i] = shift[skip+i] - (norm[skip+i+1] - norm[skip+i])/(dt*norm[skip+i])
    end
    # pad the vector g at the end
    g[n+1] = @views mean(g[1:n])
    return g
end
function growth_witness(shift::AbstractArray, norm::AbstractArray, dt, b; kwargs...)
    g_raw = growth_witness(shift, norm, dt; kwargs...)
    return smoothen(g_raw, b)
end
"""
    growth_witness(df::DataFrame, [b];
        shift=:shift,
        norm=:norm,
        time_step=determine_constant_time_step(df),
        skip=0
    )
    growth_witness(sim::PMCSimulation, [b]; kwargs...)

Calculate the growth witness directly from the result (`DataFrame` or
[`PMCSimulation`](@ref Main.Rimu.PMCSimulation)) of
[`solve`](@ref CommonSolve.solve(::ProjectorMonteCarloProblem))ing a
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem). The keyword arguments
`shift` and `norm` can be used to change the names of the relevant columns.
"""
function growth_witness(
    sim, b=Val(0);
    shift=:shift, norm=:norm, time_step=nothing, kwargs...
)
    df = DataFrame(sim)
    time_step = determine_constant_time_step(df)

    shift_vec = getproperty(df, Symbol(shift))
    norm_vec = getproperty(df, Symbol(norm))
    return growth_witness(shift_vec, norm_vec, time_step, b; kwargs...)
end

"""
    smoothen(noisy::AbstractVector, b)
Smoothen the array `noisy` by averaging over a sliding window of length `b` and
wrapping `noisy` periodically. The `mean(noisy)` is preserved.
"""
function smoothen(noisy::AbstractVector, b::Integer)
    l = length(noisy)
    @assert 1 ≤ b ≤ l
    smooth = zeros(float(eltype(noisy)), l)
    offset = b÷2 + 1
    for i in 1:l
        for j = 1:b # average over `b` elements of `noisy`
            smooth[i] += noisy[mod1(i-offset+j, l)]/b
        end
    end
    return smooth
end
smoothen(noisy::AbstractVector, ::Val{0}) = noisy
