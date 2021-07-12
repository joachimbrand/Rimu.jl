"""
    StochasticStyle(v)

`StochasticStyle` specifies the native style of the generalised vector `v` that determines
how simulations are to proceed. This can be fully stochastic (with
[`IsStochasticInteger`](@ref)), fully deterministic (with [`IsDeterministic`](@ref)), or
stochastic with floating point walker numbers and threshold (with
[`IsStochasticWithThreshold`](@ref), [`IsDynamicSemistochastic`](@ref)).

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.

For it to work with FCIQMC, a `StochasticStyle` must define the following:

* [`fciqmc_col!(::StochasticStyle, w, H, address, value, shift, dτ)`](@ref)
* [`step_stats(::StochasticStyle)`](@ref)

Optionally, it can also define:

* [`update_dvec!`](@ref): perform arbitrary transformations on the `dvec` after the spawning
step is complete.
* [`CompressionStrategy`](@ref): if `update_dvec!` is not defined, this provides an
  implementation that performs vector compression based on the returned
  `CompressionStrategy`.

"""
abstract type StochasticStyle{T} end

Base.eltype(::Type{<:StochasticStyle{T}}) where {T} = T

StochasticStyle(::AbstractArray{T}) where {T} = default_style(T)

"""
    default_style(::Type)

Pick a [`StochasticStyle`](@ref) based on the value type.
"""
default_style(::Type{T}) where T = StyleUnknown{T}()

"""
    StyleUnknown{T}() <: StochasticStyle
Trait for value types not (currently) compatible with FCIQMC. This style makes it possible to
construct dict vectors with unsupported `valtype`s.

See also [`StochasticStyle`](@ref).
"""
struct StyleUnknown{T} <: StochasticStyle{T} end

"""
    update_dvec!([::StochasticStyle,] dvec) -> dvec, nt

Perform an arbitrary transformation on `dvec` after the spawning step is completed and
report statistics to the `DataFrame`.

Returns the new `dvec` and a `NamedTuple` `nt` of statistics to be reported.

When extending this function for a custom [`StochasticStyle`](@ref), define a method
for the two-argument call signature!

The default implementation uses [`CompressionStrategy`](@ref) to compress the vector.
"""
function update_dvec!(s::StochasticStyle, v)
    len_before = length(v)
    return compress!(CompressionStrategy(s), v), (; len_before)
end
update_dvec!(v) = update_dvec!(StochasticStyle(v), v)

"""
    step_stats(::StochasticStyle)

Return a tuple of names (`Symbol` or `String`) and a tuple of zeros of values of the same
length. These will be reported as columns in the `DataFrame` returned by [`lomc!`](@ref).
"""
step_stats(::StochasticStyle)

step_stats(v, n) = step_stats(StochasticStyle(v), n)
function step_stats(s::StochasticStyle, ::Val{N}) where N
    if N == 1
        return step_stats(s)
    else
        names, stats = step_stats(s)
        return names, MVector(ntuple(_ -> stats, Val(N)))
    end
end

"""
    fciqmc_col!(w, ham, add, num, shift, dτ)
    fciqmc_col!(::StochasticStyle, args...)

Spawning and diagonal step of FCIQMC for single column of `ham`. In essence it computes

`w .+= (1 .+ dτ.*(shift .- ham[:,add])).*num`.

The [`StochasticStyle(w)`](@ref), picks the algorithm used.
"""
function fciqmc_col!(w, ham, add, num, shift, dτ)
    return fciqmc_col!(StochasticStyle(w), w, ham, add, num, shift, dτ)
end

# Generic Hamiltonian-free function definitions.
"""
    deposit!(w::AbstractDVec, add, val, parent::Pair)

Add `val` into `w` at address `add`, taking into account initiator rules if applicable.
`parent` contains the `address => value` pair from which the pair `add => val`
was created. [`InitiatorDVec`](@ref) can intercept this and add its own functionality.

Return the old value and the new value.
"""
function deposit!(w, add, val, _)
    old = w[add]
    new = old + convert(valtype(w), val)
    w[add] = new
    return (old, new)
end

"""
    random_offdiagonal(offdiagonals::AbstractVector)
    random_offdiagonal(ham::AbstractHamiltonian, add)

Generate a single random excitation, i.e. choose from one of the accessible off-diagonal
elements in the column corresponding to address `add` of the Hamiltonian matrix represented
by `ham`. Alternatively, pass as argument an iterator over the accessible matrix elements.

Return the chosen address, the probability of choosing said address, and the new value.
"""
function random_offdiagonal(offdiagonals::AbstractVector)
    nl = length(offdiagonals)
    chosen = cRand(1:nl)
    add, val = offdiagonals[chosen]
    return add, 1.0/nl, val
end
function random_offdiagonal(ham, add)
    return random_offdiagonal(offdiagonals(ham, add))
end

function offdiagonals(matrix, add)
    column = map(=>, 1:size(matrix, 1), matrix[:, add])
    return deleteat!(column, add)
end

"""
    diagonal_element(m::AbstractMatrix, i)

Get the diagonal element of `m` at index `i`.
"""
diagonal_element(m::AbstractMatrix, i) = m[i, i]
