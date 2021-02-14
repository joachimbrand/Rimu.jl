#################################################
"""
    BoseFS{N,M,A} <: BosonicFockStateAddress <: BitStringAddressType
    BoseFS(bs::A) where A <: BitAdd
    BoseFS(bs::A, b)

Address type that represents a Fock state of `N` spinless bosons in `M` orbitals
by wrapping a bitstring of type `A`. Orbitals are stored in reverse
order, i.e. the first orbital in a `BoseFS` is stored rightmost in the
bitstring `bs`. If the number of significant bits `b` is not encoded in `A` it
must be passed as an argument (e.g. for `BSAdd64` and `BSAdd128`).
"""
struct BoseFS{N,M,A} <: BosonicFockStateAddress
  bs::A
end

BoseFS{N,M}(bs::A) where {N,M,A} = BoseFS{N,M,A}(bs)

function BoseFS(bs::A, b::Integer) where A <: BitStringAddressType
  n = count_ones(bs)
  m = b - n + 1
  bfs = BoseFS{n,m,A}(bs)
  check_consistency(bfs)
  return bfs
end

function BoseFS(bs::BitAdd{I,B}) where {B,I}
  n = count_ones(bs)
  m = B - n + 1
  return BoseFS{n,m,BitAdd{I,B}}(bs)
end

"""
    BoseFS(onr::T) where T<:Union{AbstractVector,Tuple}
    BoseFS{BST}(onr::T)
Create `BoseFS` address from an occupation number representation, specifying
the occupation number of each orbital.
If a type `BST` is given it will define the underlying
bit string type. Otherwise, the bit string type is chosen to fit the `onr`.
"""
function BoseFS(onr::T) where T<:Union{AbstractVector,Tuple}
  m = length(onr)
  n = Int(sum(onr))
  b = n + m - 1
  if b ≤ 64
    A = BSAdd64
  elseif b ≤ 128
    A = BSAdd128
  else
    A = BitAdd
  end
  BoseFS{A}(onr,Val(n),Val(m),Val(b))
end

function BoseFS{A}(onr::T) where {A, T<:Union{AbstractVector,Tuple}}
  m = length(onr)
  n = Int(sum(onr))
  b = n + m - 1
  BoseFS{A}(onr,Val(n),Val(m),Val(b))
end

# This constructor is performant!!
@inline function BoseFS{N,M,A}(onr::T) where {N, M, A, T<:Union{AbstractVector,Tuple}}
  return BoseFS{A}(onr, Val(N), Val(M), Val(N+M-1))
end

# This constructor is performant!!
@inline function BoseFS{BSAdd64}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck begin
    B > 64 && throw(BoundsError(BSAdd64(0),B))
    N + M - 1 == B || @error "Inconsistency in constructor BoseFS"
  end
  bs = zero(UInt64) # empty bitstring
  for i in length(onr):-1:1
    on = onr[i]
    bs <<= on+1
    bs |= ~zero(UInt64)>>(64-on)
  end
  return BoseFS{N,M,BSAdd64}(BSAdd64(bs))
end

@inline function BoseFS{BSAdd128}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck begin
    B > 128 && throw(BoundsError(BSAdd128(0),B))
    N + M - 1 == B || @error "Inconsistency in constructor BoseFS"
  end
  bs = zero(UInt128) # empty bitstring
  for i in length(onr):-1:1
    on = onr[i]
    bs <<= on+1
    bs |= ~zero(UInt128)>>(128-on)
  end
  return BoseFS{N,M,BSAdd128}(BSAdd128(bs))
end

@inline function BoseFS{BitAdd{I,B}}(onr::T,::Val{N},::Val{M},::Val{B}) where {I,N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck  ((N + M - 1 == B) && (I == (B-1) ÷ 64 + 1)) || @error "Inconsistency in constructor BoseFS"
  bs = BitAdd{B}(0) # empty bitstring
  for i in length(onr):-1:1
    on = onr[i]
    bs <<= on+1
    bs |= BitAdd{B}()>>(B-on)
  end
  return BoseFS{N,M,BitAdd{I,B}}(bs)
end

@inline function BoseFS{BitAdd}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck  N + M - 1 == B || @error "Inconsistency in constructor BoseFS"
  bs = BitAdd{B}(0) # empty bitstring
  for i in length(onr):-1:1
    on = onr[i]
    bs <<= on+1
    bs |= BitAdd{B}()>>(B-on)
  end
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  return BoseFS{N,M,BitAdd{I,B}}(bs)
end

# comparison delegates to bs
Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
# hashing delegates to bs
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.bitstring(b::BoseFS) = bitstring(b.bs)
numChunks(::Type{BoseFS{N,M,A}}) where {N,M,A} = numChunks(A)
numBits(::Type{BoseFS{N,M,A}}) where {N,M,A} = N+M-1 # generally true for bosons
numParticles(::Type{BoseFS{N,M,A}}) where {N,M,A} = N
numModes(::Type{BoseFS{N,M,A}}) where {N,M,A} = M

function check_consistency(b::BoseFS{N,M,A}) where {N,M,A}
  numBits(b) ≤ numBits(A) || error("Inconsistency in $b: N+M-1 = $(N+M-1), numBits(A) = $(numBits(A)).")
  check_consistency(b.bs)
end
function check_consistency(b::BoseFS{N,M,A}) where {N,M,A<:Union{BSAdd64,BSAdd128}}
  numBits(b) ≤ numBits(A) || error("Inconsistency in $b: N+M-1 = $(N+M-1), numBits(A) = $(numBits(A)).")
  leading_zeros(b.bs.add) ≥ numBits(A) - numBits(b) ||  error("Ghost bits detected in $b.")
end




#################################
"""
    BoseFS2C{NA,NB,M,AA,AB} <: BosonicFockStateAddress <: BitStringAddressType

Address type that constructed with two [`BoseFS{N,M,A}`](@ref). It represents a
Fock state with two components, e.g. two different species of bosons with particle
number `NA` from species A and particle number `NB` from species B. The number of
orbitals `M` is expacted to be the same for both components.
"""
struct BoseFS2C{NA,NB,M,AA,AB} <: BitStringAddressType
  bsa::BoseFS{NA,M,AA}
  bsb::BoseFS{NB,M,AB}
end

BoseFS2C(onr_a::Tuple, onr_b::Tuple) = BoseFS2C(BoseFS(onr_a),BoseFS(onr_b))

function Base.show(io::IO, b::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
  print(io, "BoseFS2C(")
  Base.show(io,b.bsa)
  print(io, ",")
  Base.show(io,b.bsb)
  print(io, ")")
end

# performant and allocation free (if benchmarked on its own):
"""
    onr(bs)
Compute and return the occupation number representation of the bit string
address `bs` as an `SVector{M,Int32}`, where `M` is the number of orbitals.
"""
function onr(bba::BoseFS{N,M,A}) where {N,M,A}
  r = zeros(MVector{M,Int32})
  address = bba.bs
  for orbitalnumber in 1:M
    bosonnumber = Int32(trailing_ones(address))
    r[orbitalnumber] = bosonnumber
    address >>>= bosonnumber + 1
    iszero(address) && break
  end
  return SVector(r)
end

@inline function m_onr(bba::BoseFS{N,M,A}) where {N,M,A}
  r = zeros(MVector{M,Int32})
  address = bba.bs
  for orbitalnumber in 1:M
    bosonnumber = Int32(trailing_ones(address))
    @inbounds r[orbitalnumber] = bosonnumber
    address >>>= bosonnumber + 1
    iszero(address) && break
  end
  return r
end
# for some reason this is slower than the above onr() when benchmarked
s_onr(arg) = m_onr(arg) |> SVector



# # works but is not faster
# @generated function onr2(bba::BoseFS{N,M,A}) where {N,M,A}
#   quote
#     address = bba.bs
#     t = @ntuple $M k->(
#       bosonnumber = trailing_ones(address);
#       address >>>= bosonnumber + 1;
#       bosonnumber
#     )
#     return SVector(t)
#   end
# end

"""
  OccupationNumberIterator(address)
An iterator over the occupation numbers in `address`.
"""
struct OccupationNumberIterator{BS}
    bs::BS
    m::Int
end

OccupationNumberIterator(ad::BoseFS{N,M}) where {N,M} = OccupationNumberIterator(ad.bs, M)

Base.length(oni::OccupationNumberIterator) = oni.m
Base.eltype(oni::OccupationNumberIterator) = Int32

function Base.iterate(oni::OccupationNumberIterator, bsstate = (oni.bs, oni.m))
    bs, m = bsstate
    iszero(m) && return nothing
    bosonnumber = Int32(trailing_ones(bs))
    bs >>>= bosonnumber + 1
    return (bosonnumber, (bs, m-1))
end

# fast and works without allocations
function i_onr(bba::BoseFS{N,M,A}) where {N,M,A}
    SVector{M,Int32}(on for on in OccupationNumberIterator(bba))
end

"""
    nearUniformONR(N, M) -> onr::SVector{M,Int}
Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each orbital filled with at least
`N ÷ M` particles and at most with `N ÷ M + 1` particles.
"""
function nearUniformONR(n::Number, m::Number)
  return nearUniformONR(Val(n),Val(m))
end
function nearUniformONR(::Val{N}, ::Val{M}) where {N, M}
  fillingfactor, extras = divrem(N, M)
  # startonr = fill(fillingfactor,M)
  startonr = fillingfactor * @MVector ones(Int,M)
  startonr[1:extras] += ones(Int, extras)
  return SVector{M}(startonr)
end

"""
    nearUniform(BoseFS{N,M})
    nearUniform(BoseFS{N,M,A}) -> bfs::BoseFS{N,M,A}
Create bosonic Fock state with near uniform occupation number of `M` modes with
a total of `N` particles. Specifying the bit address type `A` is optional.

# Examples
```jldoctest
julia> nearUniform(BoseFS{7,5,BitAdd})
BoseFS{BitAdd}((2,2,1,1,1))

julia> nearUniform(BoseFS{7,5})
BoseFS{BSAdd64}((2,2,1,1,1))
```
"""
function nearUniform(::Type{BoseFS{N,M,A}}) where {N,M,A}
  return BoseFS{A}(nearUniformONR(Val(N),Val(M)),Val(N),Val(M),Val(N+M-1))
end
function nearUniform(::Type{BoseFS{N,M}}) where {N,M}
  return BoseFS(nearUniformONR(Val(N),Val(M)))
end

# function Base.show(io::IO, b::BoseFS{N,M,A}) where {N,M,A}
#   print(io, "BoseFS{$N,$M}|")
#   r = onr(b)
#   for (i,bn) in enumerate(r)
#     isodd(i) ? print(io, bn) : print(io, "\x1b[4m",bn,"\x1b[0m")
#     # using ANSI escape sequence for underline,
#     # see http://jafrog.com/2013/11/23/colors-in-terminal.html
#     i ≥ M && break
#   end
#   print(io, "⟩")
# end
function Base.show(io::IO, b::BoseFS{N,M,A}) where {N,M,A}
  print(io, "BoseFS")
  if A <: BSAdd64
    print(io, "{BSAdd64}")
  elseif A <: BSAdd128
    print(io, "{BSAdd128}")
  elseif A <: BitAdd
    print(io, "{BitAdd}")
  else
    print(io, "{$A}")
  end
  print(io, "((")
  for (i, on) in enumerate(onr(b))
    print(io, on)
    i ≥ M && break
    print(io, ",")
  end
  print(io, "))")
end
