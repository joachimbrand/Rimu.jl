# Change log

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
* Change log ([#409])
* Pass custom external potential to `HubbardRealSpace` ([#399, #400])
* Allow restricting the number type in `FroehlichPolaron`; make it compatible with GPU. ([#405])
* New Hamiltonian `HubbardMomSpace` is added. ([#353])

### Deprecated
* Keyword `mass` in `FroehlichPolaron` renamed to `two_m` ([#405])

### Fixed
* Dot products with `UniformProjector` could error for empty vectors or operators without offdiagonals ([#403])

### Other changes
* Documentation update ([#398, #407])
* `excitation` accepts floating point type as argument to define the type for the returned value and internal calculations. This permits compiling the function on a GPU with reduced precision arithmetic. ([#405])
* compat changed to `ElemCo` v"0.16"

## v0.18.0 - 2026-08-13

### Added
* `SignCorrelator` is a new observable for calculating mutual sign coherence ([#397])
* New mechanism for scaling and shifting Hamiltonians by scalar values based on `ScaledOrShiftedHamiltonian` with public entry points given by `*`, `+`, `VectorInterface.scale`, and `VectorInterface.add`.  ([#396])

### Removed
* `ScaledHamiltonian` is no longer available. ([#396])

### Fixed
* Docs: fix some typos ([#393])
* Fix failing benchmark runs for fork PRs: skip PR comments for fork PRs, modernize output syntax ([#394])
* Fix mystery allocation in allocation testing ([#395])

## Older versions
For changes prior to [v0.18.0](#v0180---2026-08-13) see the [list of releases](https://github.com/RimuQMC/Rimu.jl/releases) on GitHub.

## Instructions

Add all important changes while preparing a PR to the Unreleased section. When preparing a release, move the entries into a new section for the release.

Types of changes (based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/))

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities
* `Other changes` for changes to internals or documentation (not API breaking).
