# Module `BitStringAddresses`

This module contains the implementations of [`BitString`](@ref) and various Fock addresses.
The addresses serve as a basis for a Hamiltonian.

While there are not restrictions on the type of address a Hamiltonian uses, Rimu provides
implementations for Bosonic, Fermionic, and mixed [Fock
States](https://en.wikipedia.org/wiki/Fock_state).

When implementing a new address type, care must be taken to make them space-efficient and
stack-allocated - avoid using (heap-allocated) arrays to represent your addresses at all costs!

## Fock addresses

Rimu provides a variety of address implementations that should make it
straightforward to implement efficient Hamiltonians. Examples are:

- [`BoseFS`](@ref) Single-component bosonic Fock state with fixed particle and mode number.
- [`FermiFS`](@ref) Single-component fermionic Fock state with fixed particle and mode number.
- [`CompositeFS`](@ref) Multi-component Fock state composed of the above types.
- [`OccupationNumberFS`](@ref) Single-component bosonic Fock state with a fixed number of modes. The number of particles is not part of the type and can be changed by operators.

### Fock address API

```@autodocs
Modules = [BitStringAddresses]
Pages = ["fockaddress.jl","bosefs.jl","fermifs.jl","multicomponent.jl","onrfs.jl"]
Private = false
```

## Internal representations

The atomic addresses, [`BoseFS`](@ref) and [`FermiFS`](@ref), are implemented as either
bitstrings or sorted lists of particles. Using these approaches over an occupation number
representation makes the addresses much more space-efficient.

### Internal APIs

```@autodocs
Modules = [BitStringAddresses]
Pages = ["bitstring.jl", "sortedparticlelist.jl"]
Private = false
```

## Index
```@index
Pages   = ["addresses.md"]
```
