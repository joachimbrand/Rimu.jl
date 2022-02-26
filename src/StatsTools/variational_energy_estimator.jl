"""
    variational_energy_estimator(shifts, overlaps; skip=0, kwargs...)
    variational_energy_estimator(df::DataFrame; skip=0, kwargs...)
Compute the variational energy estimator from the replica time series of the `shifts` and
coefficient vector `overlaps` by blocking analysis.
Keyword arguments are passed on to [`ratio_of_means()`](@ref).
Returns a [`RatioBlockingResult`](@ref).

An estimator for the variational energy
```math
\\frac{⟨\\mathbf{c}⟩^† \\mathbf{H}⟨\\mathbf{c}⟩}{⟨\\mathbf{c}⟩^†⟨\\mathbf{c}⟩}
```
is calculated from
```math
Ē_{v}  =  \\frac{\\sum_{a<b}^Ω \\overline{(S_a+S_b) \\mathbf{c}_a^† \\mathbf{c}_b}}
               {2\\sum_{a<b}^Ω \\overline{\\mathbf{c}_a^† \\mathbf{c}_b}} ,
```
where ``Ω`` is the number of replicas. See
[arXiv:2103.07800](http://arxiv.org/abs/2103.07800).

The `DataFrame` version can extract the relevant information from the result of
[`lomc!`](@ref). Set up [`lomc!`](@ref) with the keyword argument
`replica = AllOverlaps(Ω)` and `Ω ≥ 2`.

See [`AllOverlaps`](@ref).
"""
function variational_energy_estimator(
    shifts::AbstractArray{<:AbstractArray},
    overlaps::AbstractArray{<:AbstractArray};
    kwargs...
)
    num_replicas = length(shifts)
    if length(overlaps) ≠ binomial(num_replicas,2)
        throw(ArgumentError(
            "The number of `overlaps` needs to be `binomial(length(shifts),2)`."
        ))
    end
    denominator = sum(overlaps)
    numerator = zero(shifts[1])
    count_overlaps = 0
    for i in 1:num_replicas, j in i+1:num_replicas
        count_overlaps += 1
        @. numerator += 1/2*(shifts[i] + shifts[j])*overlaps[count_overlaps]
    end
    return ratio_of_means(numerator, denominator; kwargs...)
end

function variational_energy_estimator(df::DataFrame; kwargs...)
    if "shift" in names(df)
        throw(ArgumentError(
            "`DataFrame` looks like a non-initiator output. Use keyword \
            `replica=AllOverlaps(n)` with n≥2 in `lomc!()` to set up replicas!"
        ))
    end
    num_replicas = length(filter(startswith("norm_"), names(df))) # number of replicas
    if iszero(num_replicas)
        throw(ArgumentError(
            "No replicas found. Use keyword \
            `replica=AllOverlaps(n)` with n≥2 in `lomc!()` to set up replicas!"
        ))
    end
    @assert num_replicas ≥ 2 "At least two replicas are needed, found $num_replicas"

    num_overlaps = length(filter(startswith(r"c._dot"), names(df)))
    @assert num_overlaps == binomial(num_replicas, 2) "Unexpected number of overlaps."

    shiftnames = [Symbol("shift_$i") for i in 1:num_replicas]
    shifts = map(name -> getproperty(df, name), shiftnames)
    @assert length(shifts) == num_replicas

    overlap_names = Symbol[]
    for i in 1:num_replicas, j in i+1:num_replicas
        push!(overlap_names, Symbol("c$(i)_dot_c$(j)"))
    end
    overlaps = map(name -> getproperty(df, name), overlap_names)
    @assert length(overlaps) == num_overlaps

    return variational_energy_estimator(shifts, overlaps; kwargs...)
end
