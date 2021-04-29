"""
This function creates a default working memory based on the DVec type, threading, MPI...
"""
function setup_working_memory(dvec, threading, targetwalkers)
    if threading == :auto
        threading = max(real(targetwalkers), imag(targetwalkers)) ≥ 500
    end
    if threading
        v = localpart(dvec)
        cws = capacity(v) ÷ Threads.nthreads() + 1
        return Tuple(similar(v, cws) for i=1:Threads.nthreads())
    else
        return DVecMemory(similar(dvec))
    end
end
