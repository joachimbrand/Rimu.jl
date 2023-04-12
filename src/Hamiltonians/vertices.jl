using TupleTools

"""
    _binomial(n, ::Val{K})

Binomial coefficients for small, statically known values of `K`, where `n` and `K` are
always positive. Using `Val(K)` allows the compiler to optimize away the loop.
"""
_binomial(::I, ::Val{0}) where {I} = one(I)
_binomial(n, ::Val{1}) = n
function _binomial(n::I, ::Val{K}) where {K,I}
    x = nn = I(n - K + 1)
    nn += one(I)
    for rr in I(2):I(K)
        x = div(x * nn, rr)
        nn += one(I)
    end
    return I(x)
end

"""
    _first_vertex(index, ::Val{K})

Use binary search to find index of first vertex in `(K - 1)`-dimensional simplex with index
`index`.
"""
function _first_vertex(index::I, ::Val{K}) where {I,K}
    lo = I(K - 1)
    hi = I(K + 100)
    while _binomial(hi, Val(K)) ≤ index
        lo = hi
        hi <<= 0x01
        hi + one(I) < lo && throw(OverflowError("simplex overflowed! This is a bug"))
    end
    return _first_vertex(index, Val(K), hi + one(I), lo)
end
function _first_vertex(index::I, ::Val{K}, hi::I, lo::I=I(K - 1)) where {I,K}
    while lo < hi - one(I)
        m = lo + ((hi - lo) >>> 0x01)
        if _binomial(m, Val(K)) ≤ index
            lo = m
        else
            hi = m
        end
    end
    return lo
end
function _first_vertex(index::I, ::Val{1}, ::I=I(0), ::I=I(0)) where {I}
    return index
end
function _first_vertex(index::I, ::Val{2}, ::I=I(0), ::I=I(0)) where {I}
    # This is https://oeis.org/A002024
    return floor(I, (√(8 * index + 1) + 1) / 2)
end

"""
    vertices(index::I, ::Val{N})::NTuple{N,I}

Get the vertices of simplex represented by index.
"""
@inline function vertices(index::I, ::Val{K}) where {I,K}
    index -= one(I)
    vk = _first_vertex(index, Val(K))
    index -= _binomial(vk, Val(K))
    return tuple(vk + one(I), vertices(index, Val(K - 1), vk)...)
end
@inline function vertices(index::I, ::Val{K}, prev) where {I,K}
    vk = _first_vertex(index, Val(K), prev)
    index -= _binomial(vk, Val(K))
    return tuple(vk + one(I), vertices(index, Val(K - 1), vk)...)
end
@inline vertices(index::I, ::Val{1}, _) where {I} = (index + one(I),)
@inline vertices(index, ::Val{1}) = (index,)

"""
    index(vertices)

Calculate the index from tuple or static vector of vertices. The index is equal to

```math
(i_d, i_{d-1}, ..., i_1) \\mapsto \\sum_{k=1}^{d+1} \\binom{i_k - 1}{k},
```

where ``i_k`` are the simplex vertex indices.

```jldoctest
julia> index((6,2,1))
11
```
"""
index(vertices::NTuple) = _index(vertices, one(eltype(vertices)))

@inline _index(::NTuple{0}, acc) = acc
@inline function _index(vertices::NTuple{N,I}, acc::I) where {N,I}
    acc += _binomial(first(vertices) - one(I), Val(N))
    return _index(TupleTools.tail(vertices), acc)
end
