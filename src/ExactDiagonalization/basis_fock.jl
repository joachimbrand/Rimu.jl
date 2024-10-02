"""
    build_basis(addr::AbstractFockAddress)
    build_basis(::Type{<:AbstractFockAddress}) -> basis

Return all possible Fock states of a given type as a vector. This method is _much_ faster
than `build_basis(::AbstractHamiltonian, ...)`, but does not take matrix blocking into
account. This version of `build_basis` accepts no additional arguments.

All address types except [`OccupationNumberFS`](@ref Main.Rimu.OccupationNumberFS) are
supported.

Returns a sorted vector of length equal to the [`dimension`](@ref) of `addr`.
"""
function build_basis(addr::AbstractFockAddress)
    return build_basis(typeof(addr))
end

###
### BoseFS
###
# this is equivalent to `dimension(BoseFS{n,m})`, but does not return a `BigInt`.
_bose_dimension(n, m) = binomial(n + m - 1, n)

function build_basis(::Type{<:BoseFS{N,M}}) where {N,M}
    T = typeof(near_uniform(BoseFS{N,M}))
    result = Vector{T}(undef, _bose_dimension(N, M))
    _bose_basis!(result, (), 1, N, Val(M), 2 * Threads.nthreads())
    return result
end

# Multithreaded version - attempts to spawn `tasks` tasks.
@inline function _bose_basis!(
    result::Vector, prefix, index, remaining_n, ::Val{M}, tasks::Int
) where {M}
    @sync if M < 5 || remaining_n ≤ 1 || tasks ≤ 0
        _bose_basis!(result, prefix, index, remaining_n, Val(M))
    else
        tasks ÷= 2
        Threads.@spawn begin
            _bose_basis!(result, $(0, prefix...), $index, $remaining_n, Val(M - 1), $tasks)
        end
        index += _bose_dimension(remaining_n, M - 1)
        Threads.@spawn begin
            _bose_basis!(
                result, $(1, prefix...), $index, $(remaining_n - 1), Val(M - 1), $tasks
            )
        end
        index += _bose_dimension(remaining_n - 1, M - 1)
        for n in 2:remaining_n
            _bose_basis!(result, (n, prefix...), index, remaining_n - n, Val(M - 1))
            index += _bose_dimension(remaining_n - n, M - 1)
        end
    end
end

@inline function _bose_basis!(
    result::Vector{T}, prefix, index, remaining_n, ::Val{M}
) where {M,T}
    if remaining_n == 0
        @inbounds result[index] = T((ntuple(Returns(0), Val(M))..., prefix...))
    elseif remaining_n == 1
        _basis_basecase_N1!(result, prefix, index, Val(M))
    elseif M == 1
        @inbounds result[index] = T((remaining_n, prefix...))
    elseif M == 2
        _bose_basis_basecase_M2!(result, prefix, index, remaining_n)
    elseif M == 3
        _bose_basis_basecase_M3!(result, prefix, index, remaining_n)
    else
        for n in 0:remaining_n
            _bose_basis!(result, (n, prefix...), index, remaining_n - n, Val(M - 1))
            index += _bose_dimension(remaining_n - n, M - 1)
        end
    end
    return nothing
end

###
### FermiFS
###
# this is equivalent to `dimension(FermiFS{n,m})`, but does not return a `BigInt`.
_fermi_dimension(n, m) = binomial(m, n)

function build_basis(::Type{<:FermiFS{N,M}}) where {N,M}
    T = typeof(near_uniform(FermiFS{N,M}))
    result = Vector{T}(undef, _fermi_dimension(N, M))
    _fermi_basis!(result, (), 1, N, Val(M), 2 * Threads.nthreads())
    return result
end

# Multithreaded version - attempts to spawn `tasks` tasks.
@inline function _fermi_basis!(
    result::Vector, prefix, index, remaining_n, ::Val{M}, tasks
) where {M}
    @sync if M < 5 || remaining_n ≤ M || remaining_n == 1 || tasks ≤ 0
        _fermi_basis!(result, prefix, index, remaining_n, Val(M))
    else
        tasks ÷= 2
        Threads.@spawn begin
            _fermi_basis!(result, $(0, prefix...), $index, $remaining_n, Val(M - 1), $tasks)
        end
        index += _fermi_dimension(remaining_n, M - 1)
        _fermi_basis!(result, (1, prefix...), index, remaining_n - 1, Val(M - 1), tasks)
    end
end

@inline function _fermi_basis!(
    result::Vector{T}, prefix, index, remaining_n, ::Val{M}
) where {M,T}
    @assert M ≥ remaining_n
    if remaining_n == 0
        @inbounds result[index] = T((ntuple(Returns(0), Val(M))..., prefix...))
    elseif remaining_n == M
        @inbounds result[index] = T((ntuple(Returns(1), Val(M))..., prefix...))
    elseif remaining_n == 1
        _basis_basecase_N1!(result, prefix, index, Val(M))
    else
        _fermi_basis!(result, (0, prefix...), index, remaining_n, Val(M - 1))
        index += _fermi_dimension(remaining_n, M - 1)
        _fermi_basis!(result, (1, prefix...), index, remaining_n - 1, Val(M - 1))
    end
    return nothing
end

###
### CompositeFS
###
function build_basis(::Type{C}) where {T,C<:CompositeFS{<:Any,<:Any,<:Any,T}}
    sub_results = map(build_basis, reverse(Tuple(T.parameters)))
    result = Vector{C}(undef, prod(length, sub_results))
    Threads.@threads for i in eachindex(result)
        @inbounds result[i] = C(_collect_addrs(sub_results, i))
    end
    return result
end

@inline _collect_addrs(::Tuple{}, _) = ()
@inline function _collect_addrs((v, vs...), i)
    rest, curr = fldmod1(i, length(v))
    return (_collect_addrs(vs, rest)..., v[curr])
end

###
### Base cases
###
@inline function _basis_basecase_N1!(result::Vector{T}, prefix, index, ::Val{M}) where {M,T}
    rest = ntuple(Returns(0), Val(M))
    for k in 1:M
        @inbounds result[index + k - 1] = T((setindex(rest, 1, k)..., prefix...))
    end
end
@inline function _bose_basis_basecase_M2!(result::Vector{T}, prefix, index, remaining_n) where {T}
    for n1 in 0:remaining_n
        @inbounds result[index + n1] = T((remaining_n - n1, n1, prefix...))
    end
end
@inline function _bose_basis_basecase_M3!(result::Vector{T}, prefix, index, remaining_n) where {T}
    k = 0
    for n1 in 0:remaining_n, n2 in 0:(remaining_n - n1)
        @inbounds result[index + k] = T((remaining_n - n1 - n2, n2, n1, prefix...))
        k += 1
    end
end
