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

Currently, only single-particle operators are implemented directly (via
[`move_particle`](@ref)). To implement multi-particle operators, convert the address to the
[`onr`](@ref) representation and back. See [the
implementation](../../src/Hamiltonians/HubbardMom1D.jl) of [`HubbardMom1D`](@ref) for an
example.

### Fock address API

```@docs
AbstractFockAddress
SingleComponentFockAddress
BoseFS
FermiFS
BoseFS2C
CompositeFS
num_particles
num_modes
num_components
onr
near_uniform
occupied_modes
is_occupied
num_occupied_modes
find_occupied_mode
find_mode
move_particle
```

## BitStrings

The atomic addresses, [`BoseFS`](@ref) and [`FermiFS`](@ref), are implemented as bitstrings.
Using this approach over an occupation number representation makes the addresses much more
space-efficient. The API for [`BitString`](@ref)s is as follows.

### BitString API

```@docs
BitString
num_bits
num_chunks
chunk_type
chunk_bits
top_chunk_bits
```
