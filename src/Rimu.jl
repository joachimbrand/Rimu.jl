module Rimu

using Arrow: Arrow
using DataFrames: DataFrames, DataFrame, metadata
using DataStructures: DataStructures
using LinearAlgebra: LinearAlgebra, dot, isdiag
using OrderedCollections: OrderedCollections, LittleDict, freeze
using Parameters: Parameters, @pack!, @unpack, @with_kw
using ProgressLogging: ProgressLogging, @logprogress, @withprogress
using Reexport: Reexport, @reexport
using Setfield: Setfield, @set
using StaticArrays: StaticArrays, SVector, SMatrix
using StatsBase: StatsBase
using TerminalLoggers: TerminalLogger
using Logging: ConsoleLogger
using OrderedCollections: freeze
using Random: Random, RandomDevice, seed!
using NamedTupleTools: NamedTupleTools, namedtuple, delete
import Tables
import ConsoleProgressMonitor
import TOML
import MPI

@reexport using LinearAlgebra
@reexport using VectorInterface
@reexport using CommonSolve: CommonSolve, init, step!, solve, solve!
@reexport using DataFrames

"""
    Rimu.PACKAGE_VERSION
Constant that contains the current `VersionNumber` of `Rimu`.
"""
const PACKAGE_VERSION = VersionNumber(TOML.parsefile(pkgdir(Rimu, "Project.toml"))["version"])

@doc """
    Rimu
**Random integrators for many-body quantum systems**

Welcome to `Rimu` version $PACKAGE_VERSION.
Read the documentation [online](https://joachimbrand.github.io/Rimu.jl/).
"""
Rimu

include("helpers.jl") # non MPI-dependent helper functions
include("mpi_helpers.jl")

include("Interfaces/Interfaces.jl")
@reexport using .Interfaces
using .Interfaces: dot_from_right
include("BitStringAddresses/BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("Hamiltonians/Hamiltonians.jl")
@reexport using .Hamiltonians
include("StochasticStyles/StochasticStyles.jl")
@reexport using .StochasticStyles
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
using .DictVectors: FrozenDVec
include("ExactDiagonalization/ExactDiagonalization.jl")
@reexport using .ExactDiagonalization
include("RimuIO/RimuIO.jl")
@reexport using .RimuIO
include("StatsTools/StatsTools.jl")
@reexport using .StatsTools

export mpi_rank, is_mpi_root, @mpi_root, mpi_barrier
export mpi_comm, mpi_root, mpi_size, mpi_seed!, mpi_allprintln
export lomc!
export default_starting_vector
export FciqmcRunStrategy, RunTillLastStep
export ShiftStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export DontUpdate, DoubleLogUpdate, DoubleLogUpdateAfterTargetWalkers
export ReportingStrategy, ReportDFAndInfo, ReportToFile
export ReplicaStrategy, NoStats, AllOverlaps
export PostStepStrategy, Projector, ProjectedEnergy, SignCoherence, WalkerLoneliness, Timer,
    SingleParticleDensity, single_particle_density
export TimeStepStrategy, ConstantTimeStep, OvershootControl
export localpart, walkernumber
export smart_logger, default_logger
export ProjectorMonteCarloProblem, SimulationPlan, state_vectors
export FCIQMC, num_replicas, num_spectral_states, GramSchmidt

function __init__()
    # Turn on smart logging once at runtime. Turn off with `default_logger()`.
    smart_logger()

    # Initialise the MPI library once at runtime.
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
end

include("strategies_and_params/fciqmcrunstrategy.jl")
include("strategies_and_params/poststepstrategy.jl")
include("strategies_and_params/replicastrategy.jl")
include("strategies_and_params/reportingstrategy.jl")
include("strategies_and_params/shiftstrategy.jl")
include("strategies_and_params/timestepstrategy.jl")
include("strategies_and_params/spectralstrategy.jl")

include("projector_monte_carlo_problem.jl")

include("qmc_states.jl")
include("fciqmc.jl")
include("pmc_simulation.jl")

include("lomc.jl")                  # top level

end # module
