# const HOCartBorC = Union{HOCartesianBasis, HOCartesianEnergyConserved}


abstract type AbstractHOCartesian{D,T} <: AbstractHamiltonian{T} end

struct HOCartesianBasis{D,S,A,T,IT} <: AbstractHOCartesian{D,IT}
    # S::NTuple{D,Int64}  modes per dimension; `D` is number of dimensions
    addr::A
    aspect::NTuple{D,T}         # aspect ratio from `S`
    aspect1::NTuple{D,Float64}  # (possibly custom) ratios of trapping frequencies
    # for single-particle energy scale
    energies::Vector{Float64}   # noninteracting single particle energies
    vtable::Array{Float64,4}    # interaction coefficients
    u::IT                       # bare inetraction strength
end

function HOCartesianBasis(
        addr::BoseFS;
        S = (num_modes(addr),),
        η = nothing,
        g = 1.0,
        interaction_only = false
    )
    # x dimension defines the energy scale, and should be the smallest frequency i.e.
    # largest spatial dimension
    D = length(S)
    S_sort = tuple(sort(collect(S); rev = true)...)
    S == S_sort || @warn("dimensions have been reordered")
    S_sort isa NTuple{D,Int64} || throw(ArgumentError("S must be a tuple of integers"))

    P = prod(S)
    P == num_modes(addr) || throw(ArgumentError("number of modes does not match starting address"))

    aspect = box_to_aspect(S)
    if all(denominator.(aspect) .== 1)
        aspect = Int.(aspect)
    end

    if isnothing(η)
        aspect1 = float.(aspect)
    elseif length(η) == D
        aspect1 = S == S_sort ? η : η[sortperm(collect(S); rev = true)]
    elseif length(η) == 1 && isreal(η) && η ≥ 1
        aspect1 = ntuple(d -> (d == 1 ? 1.0 : η), D)
    else
        throw(ArgumentError("Invalid aspect ratio parameter η."))
    end

    # set up the single-particle energies
    if interaction_only
        energies = zeros(P)
    else
        states = CartesianIndices(S)    # 1-indexed
        energies = reshape(map(x -> dot(aspect1, Tuple(x) .- 1/2), states), P)
    end

    # the aspect ratio appears from a change of variable when calculating the interaction
    # integrals
    u = g / (2 * sqrt(prod(aspect1)))

    # calculate the interaction integrals
    max_level = S_sort[1] - 1
    r = 0:max_level
    vmat = [four_oscillator_integral_general(i, j, k, l; max_level)
        for i in r, j in r, k in r, l in r
    ]

    return HOCartesianBasis{D, S_sort,typeof(addr),eltype(aspect),typeof(u)}(
        addr, aspect, aspect1, energies, vmat, u
    )
end

# enable accessing `S` as a field
Base.getproperty(h::HOCartesianBasis, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HOCartesianBasis, ::Val{:addr}) = getfield(h, :addr)
Base.getproperty(h::HOCartesianBasis, ::Val{:aspect}) = getfield(h, :aspect)
Base.getproperty(h::HOCartesianBasis, ::Val{:aspect1}) = getfield(h, :aspect1)
Base.getproperty(h::HOCartesianBasis, ::Val{:energies}) = getfield(h, :energies)
Base.getproperty(h::HOCartesianBasis, ::Val{:vtable}) = getfield(h, :vtable)
Base.getproperty(h::HOCartesianBasis, ::Val{:u}) = getfield(h, :u)
Base.getproperty(::HOCartesianBasis{<:Any,S}, ::Val{:S}) where {S} = S

function Base.show(io::IO, h::HOCartesianBasis)
    flag = iszero(h.energies)
    # invert the scaling of u parameter
    g = h.u * 2 * sqrt(prod(h.aspect1))
    print(io, "HOCartesianBasis($(h.addr); S=$(h.S), η=$(h.aspect1), g=$g, interaction_only=$flag)")
end

function Base.:(==)(H::HOCartesianBasis, G::HOCartesianBasis)
    return all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))
end

function starting_address(h::HOCartesianBasis)
    return h.addr
end

function LOStructure(::Type{<:HOCartesianBasis})
    return eltype(h) <: Real ? IsHermitian() : AdjointUnknown()
end

# diagonal elements are taken care of in `HOCartesianEnergyConserved.jl`

### OFFDIAGONAL ELEMENTS ###

function even_parity_excitations(S::NTuple, pairs_tuple)
    return map(even_parity_excitations, zip(S, pairs_tuple))
end

function num_even_parity_excitations(S::NTuple, pairs::OccupiedPairsMap)
    even_pairs = num_even_pairs.(S)
    odd_pairs = S .* (S .+ 1) .÷ 2 .- even_pairs
    ix = CartesianIndices(S)
    count = 0
    for (ii, jj) in pairs
        @inbounds i = ix[ii.mode]
        @inbounds j = ix[jj.mode]
        for d in 1:length(S)
            @inbounds count += ifelse(iseven(i[d] + j[d]), even_pairs[d], odd_pairs[d])
        end
    end
    return count
end
