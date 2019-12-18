module MatrixDiag

using Arpack
using SparseArrays
using LinearAlgebra
using Parameters
using Test
using Rimu

#using Walkers
#using Hamiltonians

export matrixDiagHam
export powerMethod, matHmul!, newStateVec, StateVec

function matrixDiagHam(ham; fullDiagonalise = false)
  dim = dimensionLO(ham)
  println("linear dimension = ",dim)
  println("building Hamiltonian as sparse matrix")
  @time kh, ih, bvs = sphamiltonian(ham)
  # n = numParticles(ham); m = numModes(ham) # get number of particles and modes from hamiltonian ham
  # @time kh2, ih2, bvs2 = sphamiltonian(n,m)
  # @test bvs == bvs2
  # @test kh == kh2
  # @test ih == ih2
  fh = kh + ih
  nz=nnz(fh) # number of nonzeros
  println("nonzero elements: ",nz)
  println("filling ratio: ",nz/dim^2)
  println("Hamiltonian is hermitian: ",fh==fh')

  if fullDiagonalise
    fullmatrix = Matrix(fh)
    #eigenvalue4=eig(fullmatrix)
    #println("eigenvalue4= ",eigenvalue4)
    println("diagonalise Hamiltonian full")
    @time evm = eigmin(fullmatrix)
    println("smallest eigenvalue: ",evm)
    return fullmatrix
  end

  println("now use Lanczos")
  @time ret=eigs(fh,which=:SR)
  println("smallest Lanczos eigenvalue: ",minimum(ret[1]))
  return ret, bvs, fh
end # matrixDiagHam

function sphamiltonian(ham)#::BosonicHamiltonian)
  # generate kinetic and interaction energy matrix and return as sparse matrices
  nparticles = ham.n
  mmodes = ham.m
  # if bit_String_Length(ham) > 63
  #   error("dimension error: bitstrings larger than 63 bits not supported")
  # end
  #dim = binomial(nparticles + mmodes - 1, nparticles)
  dim = dimensionLO(ham)
  table = zeros(Int,dim)
  basisvecs = buildbasis!(table, nparticles, mmodes)
  #println(map(bits,basisvecs))
  # now build the KE matrix column by column
  # and return as sparse matrix
  rows = Vector{Int}(undef,2mmodes) # definition and preallocation
  vals = Vector{Float64}(undef,2mmodes) # definition and preallocation
  diags = Vector{Float64}(undef,dim) # definition and preallocation
  kinham = spzeros(dim,dim)
  for i = 1 : dim   # columns of the Hamiltonian matrix
    bs = BSAdd64(basisvecs[i]) # configuration now with address type
    #diags[i] = diagH(bs)    # compute diagonal matrix element
    diags[i] = diagME(ham, bs) # compute diagonal matrix element
    #hopsfromhere = SpHops(bs) # iterator over linked addresses
    hopsfromhere = Hops(ham, bs) # iterator over linked addresses
    nls = length(hopsfromhere) # how many possible hops
    rows = Vector{Int}(undef,nls) # definition and preallocation
    vals = Vector{Float64}(undef,nls) # definition and preallocation
    for chosen = 1 : nls
      nadd, melem = hopsfromhere[chosen]
      #rows[chosen] = something(findfirst(isequal(nadd), basisvecs), 0)
      rows[chosen] = findfirst(isequal(nadd.add), basisvecs) # returns nothing if not found
      # index of new address in basisvecs
      vals[chosen] = melem # value of hopping matrix element
    end
    #println("rows ",rows," nls ",nls)
    kinham += sparse(rows, i*ones(Int,nls), vals, dim, dim)
  end
  #intham = sparse(Diagonal(diags))
  intham = spdiagm(0=>diags) # makes sparse matrix with diags on prinicipal diagonal
  dropzeros!(intham) # purge matrix elements with value zero
  dropzeros!(kinham)
  return kinham, intham, basisvecs
  # it is useful to check that symmetric matrices are returned
  if kinham != transpose(kinham)
    warning("Hamiltonian matrix is not symmetric")
  end
end


function buildbasis!(table::Array{T,1}, nparticles, mmodes) where T
  # this runs surprisingly quickly
  address = T(2)^(nparticles)-T(1)
  table[1] = address
  #while nadd = nextaddress(address, mmodes) > 0
  for i = 2 : binomial(nparticles + mmodes -1, nparticles)
    address = nextaddress(address, nparticles, mmodes)
    table[i] = address
  end
  return table
end

function nextaddress(address::T, nparticles, mmodes) where T
  onr = occupationnumberrepresentation(address, mmodes)
  #for p = mmodes - 1 : -1 : 1
  p = mmodes - 1
  while onr[p] == 0
    p -= 1
    if p == 0
      return zero(T)
    end
  end
  onr[p] -= 1
  onr[p+1] = nparticles - sum(onr[1:p])
  if p < mmodes - 1
    onr[mmodes] = 0
  end
  return bitaddr(onr, T)
end


"""
    occupationnumberrepresentation(address, m)

Compute and return the occupation number representation as an array of `Int`
corresponding to the given address.
"""
function occupationnumberrepresentation(address::Integer,mm::Integer)
  # compute and return the occupation number representation corresponding to
  # the given address
  # note: it is much faster to pass mm as argument than to access it as global
  # This is the fastest version with 11 seconds for 30,000,000 calls
  # onr = zeros(UInt16,mm) # this limits us to 2^16-1 orbitals
  onr = zeros(Int, mm)
  orbitalnumber = 0
  while address > 0
    orbitalnumber += 1
    bosonnumber = trailing_ones(address)
    # surpsingly it is faster to not check whether this is nonzero and do the
    # following operations anyway
    address >>>= bosonnumber
    # bosonnumber has now the number of bosons in orbtial orbitalnumber
    onr[orbitalnumber] = bosonnumber
    address >>>= 1 # shift right and get ready to look at next orbital
  end # while address
  return onr
end #

occupationnumberrepresentation(address::BSAdd64,mm::Integer) =
    occupationnumberrepresentation(address.add,mm)

occupationnumberrepresentation(address::BSAdd128,mm::Integer) =
    occupationnumberrepresentation(address.add,mm)

end # module MatrixDiag
