# growth_witness()
# smoothen()

"""
    growth_witness(norm::AbstractArray, shift::AbstractArray, dt, [b]; pad = :true) -> g
    growth_witness(df::DataFrame, [b]; pad = :true) -> g
Compute the growth witness
```math
G^{(n)} = S^{(n)} - \\frac{\\vert\\mathbf{c}^{(n+1)}\\vert - \\vert\\mathbf{c}^{(n)}\\vert}{\\vert\\mathbf{c}^{(n)}\\vert d\\tau},
```
where `S` is the `shift` and \$\\vert\\mathbf{c}^{(n)}\\vert ==\$ `norm[n, 1]`.
Setting `b ≥ 1` a sliding average over `b` time steps is computed.

If `pad` is set to `:false` then the returned array `g` has the length `length(norm) - b`.
If set to `:true` then `g` will be padded up to the same length as `norm` and `shift`.
"""
function growth_witness(norm::AbstractArray, shift::AbstractArray, dt)
    l = length(norm)
    @assert length(shift) == l "`norm` and `shift` arrays need to have the same length."
    n = l - 1
    g = Vector{eltype(shift)}(undef, l)
    for i in 1:n
        g[i] = shift[i] - (norm[i+1] - norm[i])/(dt*norm[i])
    end
    # pad the vector g at the end
    g[n+1] = @views mean(g[1:n])
    return g
end
function growth_witness(norm::AbstractArray, shift::AbstractArray, dt, b)
    g_raw = growth_witness(norm, shift, dt)
    return smoothen(g_raw, b)
end
growth_witness(df::DataFrame) = growth_witness(df.norm, df.shift, df.dτ[end])
function growth_witness(df::DataFrame, b)
    return growth_witness(df.norm, df.shift, df.dτ[1], b)
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
