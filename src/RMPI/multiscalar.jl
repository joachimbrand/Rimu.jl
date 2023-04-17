# Make MPI reduction of a `MultiScalar` work on non-Intel processors.
# The `MultiScalar` is converted into a vector before sending through MPI.Allreduce.
# Testing shows that this is about the same speed or even a bit faster on Intel processors
# than reducing the MultiScalar directly via a custom reduction operator.
# Defining the method in RMPI is strictly type piracy as MultiScalar belongs to Rimu and
# not to RMPI. Might clean this up later.
function MPI.Allreduce(ms::Rimu.MultiScalar{T}, op, comm::MPI.Comm) where {T<:Tuple}
    result_vector = MPI.Allreduce([ms...], op, comm)
    return Rimu.MultiScalar(T(result_vector))
end
