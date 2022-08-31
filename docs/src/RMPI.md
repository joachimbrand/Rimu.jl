# Module `RMPI`

```@docs
Rimu.RMPI
```

## MPIData

```@docs
Rimu.RMPI.MPIData
```

### Setup functions
The following distribute strategies are available. The functions are unexported.

```@docs
Rimu.RMPI.mpi_point_to_point
Rimu.RMPI.mpi_one_sided
Rimu.RMPI.mpi_all_to_all
Rimu.RMPI.mpi_no_exchange
```

### Strategies
```@docs
Rimu.RMPI.MPIPointToPoint
Rimu.RMPI.MPIOneSided
Rimu.RMPI.MPIAllToAll
Rimu.RMPI.MPINoWalkerExchange
```

## MPI convenience functions

```@autodocs
Modules = [Rimu.RMPI]
Pages = ["helpers.jl"]
Private = true
Public = true
```

## Index
```@index
Pages   = ["RMPI.md"]
```