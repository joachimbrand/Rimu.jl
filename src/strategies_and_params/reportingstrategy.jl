"""
    ReportingStrategy
Abstract type for strategies for reporting data in a DataFrame with [`report!()`](@ref). It
also affects the calculation and reporting of projected quantities in the DataFrame.

# Implemented strategies:

* [`EveryTimeStep`](@ref)
* [`EveryKthStep`](@ref)
* [`ReportDFAndInfo`](@ref)
* [`ReportToFile`](@ref)

Every strategy accepts the keyword arguments `projector` and `hproj` according to which a
projection of the instantaneous coefficient vector `projector⋅v` and `hproj⋅v` are reported
to the DataFrame in the fields `df.vproj` and `df.hproj`, respectively. Possible values for
`projector` are

* `nothing` - no projections are computed (default)
* `dv::AbstractDVec` - compute projection onto coefficient vector `dv` (set up with [`copy`](@ref) to conserve memory)
* [`UniformProjector()`](@ref) - projection onto vector of all ones (i.e. sum of elements)
* [`NormProjector()`](@ref) - compute 1-norm (instead of projection)
* [`Norm1ProjectorPPop()`](@ref) - compute 1-norm per population
* [`Norm2Projector()`](@ref) - compute 2-norm

In order to help set up the calculation of the projected energy, where `df.hproj` should
report `dot(projector, ham, v)`, the keyword `hproj` accepts the following values (for
`ReportingStrategy`s passed to `lomc!()`):

* `:auto` - choose method depending on `projector` and `ham` (default)
* `:lazy` - compute `dot(projector, ham, v)` every time (slow)
* `:eager` -  precompute `hproj` as `ham'*v` (fast, requires `adjoint(ham)`)
* `:not` - don't compute second projector (equivalent to `nothing`)

# Interface

A `ReportingStrategy` must define the following:

* [`report!`](@ref)
* [`report_after_step`](@ref) (optional)
* [`finalize_report!`](@ref) (optional)

# Examples

```julia
r_strat = EveryTimeStep(projector = copy(svec))
```
Record the projected energy components `df.vproj = svec⋅v` and
`df.hproj = dot(svec,ham,v)` with respect to
the starting vector (performs fast eager calculation if
`Hamiltonians.LOStructure(ham) ≠ Hamiltonians.AdjointUnknown()`),
and report every time step.

```julia
r_strat = EveryKthStep(k=10, projector = UniformProjector(), hproj = :lazy)
```
Record the projection of the instananeous coefficient vector `v` onto
the uniform vector of all 1s into `df.vproj` and of `ham⋅v` into `df.hproj`,
and report every `k`th time step.
"""
abstract type ReportingStrategy{P1,P2} end

"""
    Rimu.refine_r_strat(r_strat::ReportingStrategy, ham)

Refine the reporting strategy by replacing `Symbol`s in the keyword argument
`hproj` by the appropriate value. See [`ReportingStrategy`](@ref)
"""
refine_r_strat(r_strat::ReportingStrategy, ham) = r_strat # default

function refine_r_strat(r_strat::ReportingStrategy{P1,P2}, ham) where
                                                {P1 <: Nothing, P2 <: Symbol}
    # return ReportingStrategy(r_strat, hproj = nothing) # ignore `hproj`
    return @set r_strat.hproj = nothing # ignore `hproj`
    # using @set macro from the Setfield.jl package
end

function refine_r_strat(r_strat::ReportingStrategy{P1,P2}, ham) where
                                                {P1, P2 <: Symbol}
    if r_strat.hproj == :lazy
        @info "`hproj = :lazy` may slow down the code"
        return @set r_strat.hproj = missing
    elseif r_strat.hproj == :not
        return @set r_strat.hproj = nothing
    elseif r_strat.hproj == :eager
        return @set r_strat.hproj = copy(ham'*r_strat.projector)
    elseif r_strat.hproj == :auto
        if P1  <: AbstractProjector # for projectors don't compute `df.hproj`
            return @set r_strat.hproj = nothing
        elseif Hamiltonians.has_adjoint(ham) # eager is possible
            hpv = ham'*r_strat.projector # pre-calculate left vector with adjoint Hamiltonian
            # use smaller container to save memory
            return @set r_strat.hproj = copy(hpv)
        else # lazy is default
            return @set r_strat.hproj = missing
        end
    end
    @error "Value $(r_strat.hproj) for keyword `hproj` is not recognized. See documentation of [`ReportingStrategy`](@doc)."
end

"""
     report!(::ReportingStrategy, step, report::Report, keys, values, id="")
     report!(::ReportingStrategy, step, report::Report, nt, id="")

Report `keys` and `values` to `report`, which will be converted to a `DataFrame` before
[`lomc!`](@ref) exits. Alternatively, a `nt::NamedTuple` can be passed in place of `keys`
and `values`. If `id` is specified, it is appended to all `keys`. This is used to
differentiate between values reported by different replicas.

To overload this function for a new `ReportingStrategy`, overload
`report!(::ReportingStrategy, step, args...)` and apply the report by calling
`report!(args...)`.
"""
function report!(::ReportingStrategy, _, args...)
    report!(args...)
    return nothing
end

"""
    report_after_step(::ReportingStrategy, step, report, state)

This function is called at the very end of a step. It can let the `ReportingStrategy`
print some information to output.
"""
function report_after_step(::ReportingStrategy, args...)
    return nothing
end

"""
    finalize_report!(::ReportingStrategy, report)

Finalize the report. This function is called after all steps in [`lomc!`](@ref) have finished.
"""
function finalize_report!(::ReportingStrategy, report)
    DataFrame(report)
end

"""
    EveryTimeStep(;projector = nothing, hproj = :auto)
Report every time step. Include projection onto `projector`. See
[`ReportingStrategy`](@ref) for details.
"""
@with_kw struct EveryTimeStep{P1,P2} <: ReportingStrategy{P1,P2}
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end

"""
    EveryKthStep(;k = 10, projector = nothing, hproj = :auto)
Report every `k`th step. Include projection onto `projector`. See
[`ReportingStrategy`](@ref) for details.
"""
@with_kw struct EveryKthStep{P1,P2} <: ReportingStrategy{P1,P2}
    k::Int = 10
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end
function report!(s::EveryKthStep, step, args...)
    step % s.k == 0 && report!(args...)
    return nothing
end

"""
    ReportDFAndInfo(; k=10, i=100, io=stdout, writeinfo=true, projector = nothing, hproj = :auto)
Report every `k`th step in DataFrame and write info message to `io` every `i`th
step (unless `writeinfo == false`). The flag `writeinfo` is useful for
controlling info messages in MPI codes. Include projection onto `projector`.
See [`ReportingStrategy`](@ref) for details.
"""
@with_kw struct ReportDFAndInfo{P1,P2} <: ReportingStrategy{P1,P2}
    k::Int = 10 # how often to write to DataFrame
    i::Int = 100 # how often to write info message
    io::IO = stdout # IO stream for info messages
    writeinfo::Bool = true # write info only if true - useful for MPI codes
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end
function report!(s::ReportDFAndInfo, step, args...)
    step % s.k == 0 && report!(args...)
    return nothing
end
function report_after_step(s::ReportDFAndInfo, step, args...)
    if s.writeinfo && step % s.i == 0
        println(s.io, "Step ", step)
        flush(s.io)
    end
end

"""
    ReportToFile(; kwargs...) <: ReportingStrategy

Reporting strategy that writes the report directly to a file. Useful when dealing with long
jobs or large numbers of replicas, when the report can incur a significant memory cost.

# Keyword arguments

* `filename`: the file to report to. If the file already exists, a new file is created.
* `chunk_size = 1000`: the size of each chunk that is written to the file.
* `save_if = true`: if this value is true, save the report, otherwise ignore it. Use
  `save_if=is_mpi_root()` when running MPI jobs.
* `return_df`: if this value is true, read the file and return the data frame at the end of
  computation. Otherwise, an empty `DataFrame` is returned.
* `io=stdout`: The `IO` to print messages to. Set to `devnull` if you don't want to see
  messages printed out.
* `projector = nothing`: include projection onto `projector`
* `hproj = :auto`: secondary projector
See `ReportingStrategy` for details regarding the use of projectors.
"""
@with_kw struct ReportToFile{P1,P2} <: ReportingStrategy{P1,P2}
    filename::String
    chunk_size::Int = 1000
    save_if::Bool = true
    return_df::Bool = false
    io::IO = stdout
    projector::P1 = nothing
    hproj::P2 = :auto
end
function refine_r_strat(s::ReportToFile{P1,P2}, ham::H) where {P1,P2,H}
    if s.save_if
        # If filename exists, add -1 to the end of it. If that exists as well,
        # increment the number after the dash
        new_filename = s.filename
        if isfile(new_filename)
            base, ext = splitext(new_filename)
            new_filename = string(base, "-", 1, ext)
        end
        while isfile(new_filename)
            base, ext = splitext(new_filename)
            m = match(r"(.*)-([0-9]+)$", base)
            if !isnothing(m)
                new_filename = string(m[1], "-", parse(Int, m[2]) + 1, ext)
            end
        end
        if s.filename ≠ new_filename
            println(s.io, "File `$(s.filename)` exists. Using `$(new_filename)`.")
            s = @set s.filename = new_filename
        else
            println(s.io, "Saving report to `$(s.filename)`.")
        end
    end
    # Do the standard refine_r_strat to take care of projectors.
    return invoke(refine_r_strat, Tuple{ReportingStrategy{P1,P2}, H}, s, ham)
end
function report!(s::ReportToFile, _, args...)
    if s.save_if
        report!(args...)
    end
    return nothing
end
function report_after_step(s::ReportToFile, step, report, state)
    if s.save_if && step % s.chunk_size == 0
        # Report some stats:
        print(s.io, "[ ", lpad(step, 11), " | ")
        shift = lpad(round(state.replicas[1].params.shift, digits=4), 10)
        norm = lpad(round(state.replicas[1].pnorm, digits=4), 10)
        println(s.io, "shift: ", shift, " | norm: ", norm)

        if isfile(s.filename)
            Arrow.append(s.filename, report.data)
        else
            Arrow.write(s.filename, report.data; file=false)
        end
        empty!(report)
    end
end
function finalize_report!(s::ReportToFile, report)
    if s.save_if
        println(s.io, "Finalizing")
        if isfile(s.filename)
            Arrow.append(s.filename, report.data)
        else
            Arrow.write(s.filename, report.data; file=false)
        end
        if s.return_df
            return DataFrame(Arrow.Table(s.filename))
        end
    end
    return DataFrame()
end

###
### Proj observables
###
"""
    compute_proj_observables(v, ham, r::ReportingStrategy)
Compute the projection of `r.projector⋅v` and `r.hproj⋅v` or
`r.projector⋅ham*v` according to
the [`ReportingStrategy`](@ref) `r`.
"""
function compute_proj_observables(v, ham, ::ReportingStrategy{Nothing,Nothing})
    return (;)
end

# catch an error
function compute_proj_observables(v, ham, ::ReportingStrategy{<:Any,Symbol})
    error("`Symbol` is not a valid type for `hproj`. Use `refine_r_strat`!")
end

#  single projector, e.g. for norm calculation
function compute_proj_observables(v, ham, r::ReportingStrategy{<:Any,Nothing})
    return (; vproj=r.projector⋅v)
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# (slow) generic version with single projector, e.g. for computing projected energy
function compute_proj_observables(v, ham, r::ReportingStrategy{<:Any,Missing})
    return (; vproj=r.projector⋅v, hproj=dot(r.projector, ham, v))
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# fast version with 2 projectors, e.g. for computing projected energy
function compute_proj_observables(v, ham, r::ReportingStrategy)
    return (; vproj=r.projector⋅v, hproj=r.hproj⋅v)
end
# The dot products work across MPI when `v::MPIData`; MPI sync
