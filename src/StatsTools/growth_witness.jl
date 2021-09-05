# growth_witness()
# smoothen()

"""
    growth_witness(shift::AbstractArray, norm::AbstractArray, dt, [b]; skip=0) -> g
    growth_witness(df::DataFrame, [b]; skip=0) -> g
Compute the growth witness
```math
G^{(n)} = S^{(n)} - \\frac{\\vert\\mathbf{c}^{(n+1)}\\vert - \\vert\\mathbf{c}^{(n)}\\vert}{\\vert\\mathbf{c}^{(n)}\\vert d\\tau},
```
where `S` is the `shift` and \$\\vert\\mathbf{c}^{(n)}\\vert ==\$ `norm[n, 1]`.
Setting `b ≥ 1` a sliding average over `b` time steps is computed using
[`smoothen()`](@ref). The first `skip` time steps are skipped.

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
function growth_witness(df::DataFrame; kwargs...)
    return growth_witness(df.shift, df.norm, df.dτ[end]; kwargs...)
end
function growth_witness(df::DataFrame, b; kwargs...)
    return growth_witness(df.shift, df.norm, df.dτ[1], b; kwargs...)
end

"""
    smoothen(noisy::AbstractVector, b)
Smoothen the array `noisy` by averaging over a sliding window of length `b` and
wrapping `noisy` periodically. The [`mean(noisy)`](@ref) is preserved.
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
