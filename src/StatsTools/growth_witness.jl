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
    g[n+1] = g[n]
    return g
end
function growth_witness(norm::AbstractArray, shift::AbstractArray, dt, b; pad = :true)
    g_raw = growth_witness(norm, shift, dt)
    return smoothen(g_raw, b; pad)
end
growth_witness(df::DataFrame) = growth_witness(df.norm, df.shift, df.dτ[1])
function growth_witness(df::DataFrame, b; pad = :true)
    return growth_witness(df.norm, df.shift, df.dτ[1], b; pad=pad)
end

"""
    smoothen(noisy::AbstractVector, b; pad = :true)
Smoothen the array `noisy` by averaging over a sliding window of length `b`.
Pad to `length(noisy)` if `pad == true`. Otherwise, the returned array will have
the length `length(noisy) - b`.
"""
function smoothen(noisy::AbstractVector, b; pad = :true)
    l = length(noisy)
    n = l - b
    smooth = Vector{promote_type(eltype(noisy),Float64)}(undef, pad ? l : n)
    offset = pad ? b÷2 : 0 # use offset only if pad == :true
    for i in 1:n
        smooth[i + offset] = 1/b * sum(noisy[i:i+b-1])
    end
    if pad # pad the vector g at both ends
        smooth[1:offset] .= smooth[offset+1]
        smooth[offset+n+1 : end] .= smooth[offset+n]
    end
    return smooth
end
