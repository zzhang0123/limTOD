# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-07-24

### Added

- 🚀 **`limtod_jax` package**: pure-JAX, differentiable port of the sky→TOD
  chain (pointing → ZYZ Euler angles → Wigner rotation of beam alms →
  beam-weighted harmonic dot), implementing the e-RHINO
  `docs/limtod-port-contract.md`. Ships `zyz_of_pointing`, `rotate_alm`,
  `beam_weighted_sum`, `generate_tod_sky` (+ exact adjoint),
  `generate_projection_rows`, and JAX-side HEALPix map↔alm wrappers.
  Matches `limTOD.simulator.generate_TOD_sky(..., truncate_frac_thres=0.0)`
  to ~1e-12 relative in float64 (oracle-tested at extreme pointing corners);
  jit/vmap/grad-safe. Install with `pip install -e ".[jax]"` (Python ≥ 3.11).
- 📦 **Dependency extras**: `[mpi]` (mpi4py), `[gdsm]` (pygdsm), `[jax]`
  (jax + s2fft), and `[full]` (all of the above).

### Changed

- **Fresh-install default changed**: `mpi4py` and `pygdsm` are no longer
  required dependencies — a plain `pip install -e .` now installs a lighter,
  wheel-only stack (numpy, healpy, astropy, scipy, tqdm, mpmath). Rationale:
  mpi4py needs a system MPI toolchain and pygdsm downloads sky-model data,
  both an unnecessary burden for downstream users (e.g. e-RHINO) who only
  consume the JAX port. **Existing environments are unaffected** (already-
  installed packages stay); for a fresh full setup use
  `pip install -e ".[full]"`, for MPI runs `.[mpi]`, for GDSM skies `.[gdsm]`.
- `limTOD.mpiutil` falls back to serial mode (`rank=0, size=1, world=None`)
  when mpi4py is absent — restoring the graceful-degradation behavior of the
  upstream caput `mpiutil` it is adapted from. With mpi4py installed,
  behavior is unchanged.
- `GDSM_sky_model` imports pygdsm lazily and raises a clear `ImportError`
  pointing at `pip install "limTOD[gdsm]"` when it is missing. All other
  functionality works without pygdsm.

### Fixed

- Aligned `pyproject.toml`/`limTOD.__version__` (stuck at 1.1.0) with the
  changelog version history.

## [1.2.0] - 2025-10-06

### Added

- Add `CHANGELOG.md`

### Changed

- **BREAKING**: Renamed `TODsim` class to `TODSim` for better Python naming conventions
- Renamed old `limTODsim` class references to `TODSim` throughout codebase
- Updated all import statements and class instantiations to use `TODSim`
- Updated `__init__.py` and `__all__` exports to reflect new class name

### Improved

- 📝 **Documentation**:
  - **Structure**: Moved "Latest Updates" section from README.md to dedicated CHANGELOG.md file
  - 📋 **Table of Contents**: Updated README.md Table of Contents to accurately reflect document structure
  - **Examples**: Removed in flavour of example notebooks to simplify maintainace
- 🔧 **Code Organization**:
  - Move example notebooks to `examples/`
  - Improved consistency in class naming across all files including:
    - Source code (`simulator.py`)
    - Package exports (`__init__.py`)
    - Documentation (`README.md`)
    - Example notebooks (`examples.ipynb`, `mm_example.ipynb`)
    - Change Log (`CHANGELOG.md`)

### Fixed

- Corrected all references to use consistent `TODSim` class name
- Fixed import statements in example notebooks and documentation

## [1.1.0] - 2025-10-05

### Added

- 🎯 **Full Stokes Support**: Added complete polarization handling (I, Q, U, V) for both TOD simulation and map-making
- 🗺️ **Map-Making Pipeline**: Implemented `HPW_mapmaking` class combining high-pass filtering and Wiener filtering for sky reconstruction from TOD
- 🎲 **Gaussian Random Field Generator**: Added generator for correlated sky realizations from frequency-frequency angular power spectra C_ℓ(ν,ν'), enabling realistic simulation of line intensity mapping signals with spectral correlations (credit: Katrine Alice Glasscock, Philip Bull)
- 📓 **Example Notebooks**: Added comprehensive Jupyter notebook demonstrating the full map-making workflow ([examples/mm_example.ipynb](examples/mm_example.ipynb))

### Changed

- **BREAKING**: `beam_func` and `sky_func` now require keyword-only arguments, two of which must be `freq` and `nside`:

  ```python
  # Old: beam_func(freq, nside) and sky_func(freq, nside)
  # New: beam_func(freq=xx, nside=xx) and sky_func(freq=xx, nside=xx)
  ```

- **BREAKING**: Function outputs must be HEALPix maps with specific shapes:
  - 1D array of length npix for unpolarized (**I**) beam/sky
  - 2D array of shape (3, npix) for polarized (**I, Q, U**) beam/sky
  - 2D array of shape (4, npix) for polarized (**I, Q, U, V**) beam/sky

### Fixed

- 🐛 **Bug Fix**: Corrected a critical sign error in coordinate rotation transformations

## [1.0.0] - 2025-09-01

### Initial Release

- Initial release of limTOD: Time-Ordered Data simulation for single-dish radio telescopes

### Key Features

- TOD simulation with realistic noise models (1/f noise, white noise, gain variations)
- Support for asymmetric beam patterns
- Direct beam convolution using HEALPix spherical harmonics rotation and sum to calculate Tsky
- Flexible beam and sky model functions
- Global Sky Model (GDSM) integration
- MPI parallelization support
- Example scanning patterns and beam models
- Example notebooks for getting started
- Documentation and examples
