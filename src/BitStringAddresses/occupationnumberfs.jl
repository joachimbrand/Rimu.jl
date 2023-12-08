struct OccupationNumberFS{M,T} <: SingleComponentFockAddress{missing,M}
    onr::SVector{M,T}

    function OccupationNumberFS(onr::SVector{M,T}) where {M,T}
        T <: Unsigned || throw(ArgumentError("T must be an unsigned integer type, got $T"))
        new{M,T}(onr)
    end
end

function OccupationNumberFS{M,T}(args...) where {M,T}
    return OccupationNumberFS(SVector{M,T}(args...))
end

function OccupationNumberFS(args...)
    sv = SVector(args...)
    all(isinteger, sv) || throw(ArgumentError("all arguments must be integers"))
    all(x -> x ≥ 0, sv) || throw(ArgumentError("all arguments must be non-negative"))
    all(x -> x < 256, sv) || throw(ArgumentError("arguments don't fit in a byte, specify type"))
    return OccupationNumberFS(SVector{length(sv),UInt8}(args...))
end

function OccupationNumberFS{M}(args...) where M
    sv = SVector{M}(args...)
    all(isinteger, sv) || throw(ArgumentError("all arguments must be integers"))
    all(x -> x ≥ 0, sv) || throw(ArgumentError("all arguments must be non-negative"))
    all(x -> x < 256, sv) || throw(ArgumentError("arguments don't fit in a byte, specify type"))
    return OccupationNumberFS(SVector{M,UInt8}(args...))
end

function print_address(io::IO, ofs::OccupationNumberFS{M,T}; compact=false) where {M,T}
    if compact
        BITS = sizeof(T) * 8
        print(io, "|", join(ofs.onr, ' '), "⟩{", BITS, "}")
    else
        print(io, "OccupationNumberFS{", M, ", ", T, "}", Int.(tuple(ofs.onr...)))
    end
end

onr(ofs::OccupationNumberFS) = ofs.onr
Base.length(::OccupationNumberFS{M}) where M = M
Base.getindex(ofs::OccupationNumberFS, i::Int) = ofs.onr[i]
num_occupied_modes(ofs::OccupationNumberFS) = mapreduce(!iszero, +, onr(ofs))
num_modes(::Type{OccupationNumberFS{M}}) where M = M
num_particles(ofs::OccupationNumberFS) = sum(onr(ofs))

function Base.iterate(ofs::OccupationNumberFS, state=1)
    if state > num_modes(ofs)
        return nothing
    else
        return ofs[state], state+1
    end
end
