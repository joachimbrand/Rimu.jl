"""
    setup_working_memory(dvec, threading, targetwalkers, initiator=0)
This function creates a default working memory based on the DVec type, threading, MPI...
"""
function setup_working_memory(dvec, threading, targetwalkers, initiator=0)
    if threading == :auto
        threading = max(real(targetwalkers), imag(targetwalkers)) ≥ 500
    end
    if initiator > 0
        threading && @warn "Threading does not yet jive with initiators. Disabling threading."
        V = valtype(dvec)
        return InitiatorMemory(DVecMemory(similar(dvec, InitiatorValue{V})), initiator)
    elseif threading
        v = localpart(dvec)
        cws = capacity(v) ÷ Threads.nthreads() + 1
        return Tuple(similar(v, cws) for i=1:Threads.nthreads())
    else
        return DVecMemory(similar(dvec))
    end
end
