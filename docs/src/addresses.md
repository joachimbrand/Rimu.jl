# Module `BitStringAddresses`

This module contains the implementations of [`BitString`](@ref) and various Fock addresses.
The addresses serve as a basis for a Hamiltonian.

While there are not restrictions on the type of address a Hamiltonian uses, Rimu provides
implementations for Bosonic, Fermionic, and mixed [Fock
States](https://en.wikipedia.org/wiki/Fock_state).

When implementing a new address type, care must be taken to make them space-efficient and
stack-allocated - avoid using arrays to represent your addresses at all costs!

## Fock addresses

Rimu provides a variety of address implementations (see below) that should make it
straightforward to implement efficient Hamiltonians.

### Fock address API

```@autodocs
Modules = [BitStringAddresses]
Pages = ["fockaddress.jl","bosefs.jl","fermifs.jl","multicomponent.jl"]
Private = false
```

## BitStrings

The atomic addresses, [`BoseFS`](@ref) and [`FermiFS`](@ref), are implemented as bitstrings.
Using this approach over an occupation number representation makes the addresses much more
space-efficient. The API for [`BitString`](@ref)s is as follows.

### BitString API

```@autodocs
Modules = [BitStringAddresses]
Pages = ["bitstring.jl"]
Private = false
```

## Index
```@index
Pages   = ["addresses.md"]
```
