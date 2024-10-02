# small functions supporting fciqmc!()
# versions without dependence on MPI.jl
"""
    MultiScalar

Wrapper over a tuple that supports `+`, `*`, `min`, and `max`. Used with MPI communication
because `SVector`s are treated as arrays by `MPI.Allreduce` and `Tuples` do not support
scalar operations.

# Example

Suppose you want to compute the sum of a vector `dv` and also get the number of positive
elements it has in a single pass. You can use `MultiScalar`:

```julia
julia> dv = DVec(:a => 1, :b => -2, :c => 1);

julia> s, p = mapreduce(+, values(dv)) do v
    Rimu.MultiScalar(v, Int(sign(v) == 1))
end;

julia> s, p
(0, 2)
```

Note that only `MultiScalar`s with the same types can be operated on. This is a feature, as
it forces type stability.
"""
struct MultiScalar{T<:Tuple}
    tuple::T
end
MultiScalar(args...) = MultiScalar(args)
MultiScalar(v::SVector) = MultiScalar(Tuple(v))
MultiScalar(m::MultiScalar) = m
MultiScalar{T}(m::MultiScalar{T}) where T<:Tuple = m
MultiScalar(arg) = MultiScalar((arg,))

Base.getindex(m::MultiScalar, i) = m.tuple[i]

const SVecOrTuple = Union{SVector,Tuple}

for op in (:+, :*, :max, :min)
    @eval function Base.$op(a::MultiScalar{T}, b::MultiScalar{T}) where {T}
        return MultiScalar($op.(a.tuple, b.tuple))
    end
    @eval function Base.$op(a::MultiScalar, b::SVecOrTuple)
        return $op(a, MultiScalar(b))
    end
    @eval function Base.$op(a::SVecOrTuple, b::MultiScalar)
        return $op(MultiScalar(a), b)
    end
end

Base.iterate(m::MultiScalar, args...) = iterate(m.tuple, args...)
Base.length(m::MultiScalar) = length(m.tuple)

# this is run during Rimu initialisation and sets the default
"""
    smart_logger(args...)
Enable terminal progress bar during interactive use (i.e. unless running on CI or HPC).
Arguments are passed on to the logger. This is run once during `Rimu` startup. Undo with
[`default_logger`](@ref) or by setting `Base.global_logger()`.
"""
function smart_logger(args...; kwargs...)
    if isdefined(Main, :IJulia) && Main.IJulia.inited # are we running in Jupyter?
        # need for now as TerminalLoggers currently does not play nice with Jupyter
        # install a bridge to use ProgressMeter under the hood
        # may become unneccessary in the future
        ConsoleProgressMonitor.install_logger(; kwargs...) # use ProgressMeter for Jupyter
    elseif isa(stderr, Base.TTY) && (get(ENV, "CI", nothing) ≠ true) # running in terminal?
        Base.global_logger(TerminalLogger(args...; kwargs...)) # enable progress bar
    end
    return Base.global_logger()
end
"""
    default_logger(args...)
Reset the `global_logger` to `Logging.ConsoleLogger`. Undoes the effect of
[`smart_logger`](@ref). Arguments are passed on to `Logging.ConsoleLogger`.
"""
function default_logger(args...; kwargs...)
    Base.global_logger(ConsoleLogger(args...; kwargs...)) # disable terminal progress bar
    return Base.global_logger()
end

# small functions for handling keyword arguments
"""
    replace_keys(nt::NamedTuple, (:old1 => :new1, :old2 => :new2, ...))

Replace keys in a `NamedTuple` with new keys. This is useful for renaming fields in a
`NamedTuple`. Ignores keys that are not present in the `NamedTuple`.
"""
function replace_keys(nt::NamedTuple, pairs)
    for (old, new) in pairs
        if isdefined(nt, old)
            nt = (; nt..., namedtuple((new,), (nt[old],))...)
            nt = delete(nt, old)
        end
    end
    return nt
end

"""
    delete_and_warn_if_present(nt::NamedTuple, keys)

Delete keys from a `NamedTuple` and issue a warning if they are present. This is useful for
removing unused keyword arguments.
"""
function delete_and_warn_if_present(nt::NamedTuple{names}, keys) where {names}
    unused = names ∩ keys
    if !isempty(unused)
        @warn "The keyword(s) \"$(join(unused, "\", \""))\" are unused and will be ignored."
    end
    return delete(nt, keys)
end

"""
    clean_and_warn_if_others_present(nt::NamedTuple{names}, keys) where {names}

Remove keys from a `NamedTuple` that are not in `keys` and issue a warning if they are
present.
"""
function clean_and_warn_if_others_present(nt::NamedTuple{names}, keys) where {names}
    unused = setdiff(names, keys)
    if !isempty(unused)
        @warn "The keyword(s) \"$(join(unused, "\", \""))\" are unused and will be ignored."
    end
    return NamedTuple{filter(x -> x ∈ keys, names)}(nt)
end
