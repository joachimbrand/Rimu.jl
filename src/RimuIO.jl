"""
Module to provide file inut and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load(filename)`](@ref) Load Arrow file into dataframe.
"""
module RimuIO

using ..DictVectors

using Arrow, DataFrames, JLSO

export save_df, load_df

const COMPLEX = Symbol("JuliaLang.Complex")
ArrowTypes.arrowname(::Type{<:Complex}) = COMPLEX
ArrowTypes.ArrowType(::Type{Complex{T}}) where {T} = Tuple{T,T}
ArrowTypes.JuliaType(::Val{COMPLEX}, ::Type{Tuple{T,T}}) where {T} = Complex{T}
ArrowTypes.toarrow(a::Complex) = (a.re, a.im)
ArrowTypes.fromarrow(::Type{Complex{T}}, t::Tuple{T,T}) where {T} = Complex{T}(t...)

"""
    RimuIO.save_df(filename, df::DataFrame)
Save dataframe in Arrow format.
"""
save_df(filename, df::DataFrame) = Arrow.write(filename, df)

"""
    RimuIO.load_df(filename) -> DataFrame
Load Arrow file into dataframe.
"""
load_df(filename) = DataFrame(Arrow.Table(filename))

function serialize_state(filename, state)
    # Extract pairs from DVecs for better storage
    temp_storage = map(r -> r.v, state.replicas)
    dvec_pairs = map(r -> collect(pairs(storage(r.v))), state.replicas)

    # Empty working memory and replace vectors with empty ones.
    for r in state.replicas
        if r.w isa Tuple
            map(empty!, r.w)
        else
            empty!(r.w)
        end
        r.v = empty(r.v)
    end

    # Serialize both
    JLSO.save(filename, :state => state, :dvec_pairs => dvec_pairs)

    # Regenerate the state
    for (r, v) in zip(state.replicas, temp_storage)
        r.v = v
    end
    return filename
end

function deserialize_state(filename)
    d = JLSO.load(filename)
    state = d[:state]
    for (r, ps) in zip(state.replicas, d[:dvec_pairs])
        for (k, v) in ps
            storage(r.v)[k] = v
        end
    end
    return state
end

end # module RimuIO
