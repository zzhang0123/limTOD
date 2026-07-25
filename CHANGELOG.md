# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.2] - 2026-07-25

### Documentation

- 🧭 **Beam convention restated in fully coordinate-anchored language**
  (`docs/theory.md`): the tangent basis is now *defined* by coordinate
  derivatives — `ê_el ≡ ∂b̂/∂e` (increasing elevation) and
  `ê_az ≡ (1/cos e) ∂b̂/∂A` (increasing azimuth) — with explicit
  East/North/Up component formulas, so the statement is closed under
  the horizontal system alone. Observer-dependent words ("up/right")
  are removed from every convention statement (docs, docstrings,
  tests): they depend on the facing direction — a φ = 90° feature maps
  toward East when pointing at azimuth 0° but toward **West** at
  azimuth 180° — which is spelled out as a worked anchor. The
  identity/unrotated case is additionally phrased as the
  North-Pole-zenith reading (beam of an antenna at the terrestrial
  North Pole pointing at the NCP, φ = 0 along the LST = 0 meridian
  toward decreasing declination, φ = 90° along RA = 90°).
- README now opens with the documentation link
  (https://limtod.readthedocs.io).
- **Parked configuration documented and pinned**: with only the site
  steps applied (`A = 0°, e = 90°`), the beam axes land on
  (φ = 0, φ = 90°, boresight) → (south point, east point, zenith) —
  the standard right-handed alt-az Cartesian triad, i.e. the general
  tangent formulas at `A = 0°, e = 90°`. Two new parked-case tests
  (el = 90° was an untested corner; 11 orientation tests total).

## [1.5.1] - 2026-07-25

### Documentation

- 🧭 **Beam coordinate convention stated definitively**
  (`docs/theory.md`): a beam's orientation is only meaningful relative
  to the local horizontal system, so the convention is defined there —
  boresight at the beam map's north pole; the map point (θ, φ) is
  carried to the tangent direction
  `cos(φ+ψ)·ê_up + sin(φ+ψ)·ê_right` at the pointing (ê_up = increasing
  elevation, ê_right = increasing azimuth, ψ = selfrot). φ = 0 → up,
  φ = 90° → right. Special cases spelled out: the identity/equatorial
  reading (lat 0, LST 0, az 0, el 0 — the map IS the equatorial map,
  with the chart-South/antenna-up trap flagged), zenith pointing
  (frame carried continuously: ê_up → azimuth A+180°), and the
  insensitivity of symmetric beams. The `beam_func` contract, the key
  simulator docstrings, and the patch-beam (l, m) wording (axes are
  (right, up) at the pointing, not compass (East, North)) now all
  reference it.
- 🗑️ **conventions.pdf removed**: its beam-frame figure omitted the
  local horizontal system — without which the convention is not
  uniquely defined — and its x̂/ŷ captions disagreed with the
  implementation by 90°. The prose convention above supersedes it.
- 📡 **UVBeam adapter docs rewritten in axis language**
  (`docs/uvbeam.md`): pyuvdata's frame is a *fixed* antenna-local
  (x = East, y = North, z = boresight) system with az running East →
  North, and pyuvdata does not define how it rotates with pointing;
  limTOD's frame is pointing-anchored (up/right). The locked
  identification is **North → up, East → right**
  (`az_uvbeam = 90° − φ`, orientation-reversing at the chart level —
  the source of the historical hand-derivation failures).

### Tests

- `tests/test_beam_orientation.py`: displaced-blob probes through the
  full pointing chain pin φ = 0 → up and φ = 90° → right at two
  independent pointings, the identity/equatorial special case, the
  selfrot sense, and reject the mirrored convention — the orientation
  is invisible to every symmetric-beam test, so these are the only
  guards.

## [1.5.0] - 2026-07-25

### Added

- 🗺️ **`GLS_mapmaking`** (`limTOD.gls_mapmaking`) — a generalised-least-
  squares map-maker ported from hydra-tod's iterative GLS
  (Zhang et al. 2026, RASTI rzag024, §3.2), serial, no extra
  dependencies. Weights the TOD with the inverse of the full
  1/f + white time-time covariance (`flicker_noise_cov`, exactly the
  matrix `sim_noise` draws limTOD's noise from) under the multiplicative
  model of `generate_TOD`, iterating the reweighted solve; a
  `noise_model="additive"` single-solve mode covers externally
  calibrated data (it warns when falling back to the fractional-noise
  parametric covariance). Same constructor as `HPW_mapmaking` (shared
  operator construction via the new `_MapmakingBase`), same return
  conventions, priors, and `return_full_cov`. 29 tests including an
  independent-IRLS oracle, a forced-non-convergence uncertainty pin,
  and an end-to-end where the GLS beats uniform weighting 4x under
  intra-chunk red noise. The faithful single-TOD solver is exported as
  `iterative_gls`.
- 📚 **Sphinx documentation** (`docs/`, MyST-Markdown) with autodoc API
  reference, deployed via ReadTheDocs (`.readthedocs.yaml`,
  https://limtod.readthedocs.io); the `[docs]` extra now installs the
  Sphinx toolchain.

### Changed

- ✏️ **Renamed `limTOD.simeer` → `limTOD.patchbeam`**
  (`SimeerTODSim` → `PatchBeamTODSim`): the disc-restricted (l, m)
  tangent-plane path is not MeerKAT-specific, so the subpackage now
  carries the generic name the docs already used. The `simeer` name
  shipped only in 1.4.0 (same day); `limTOD.simeer` remains available
  as a deprecated alias (top-level API, old class name, and submodule
  paths all keep working with a `DeprecationWarning`) and will be
  removed in 2.0.
- `HPW_mapmaking`'s operator construction and prior assembly moved into
  the shared `_MapmakingBase` base class (behavior unchanged — the
  oracle suite pins the outputs); the hand-written
  `docs/api-reference.md` is superseded by the generated API pages.

## [1.4.0] - 2026-07-25

### Added

- 📡 **`limTOD.simeer` subpackage** — the MeerKLASS-optimal sky-TOD path
  (merged from the standalone Simeer package, same author/license): the
  beam stays on its native (l, m) direction-cosine grid and each pointing
  integrates only a HEALPix disc of sky pixels, avoiding the harmonic
  rotation that narrow, finely-gridded beams make expensive.
  `SimeerTODSim` subclasses `TODSim` (identical noise model and
  `generate_TOD` API, only the sky step differs); `MeerKLASSBeam` loads
  the holographic NPZ format. 44 tests migrated, including the
  cross-validation against the classic HEALPix path. Purely additive —
  classic limTOD usage is unchanged, and the standalone Simeer package
  remains available. Parallelism (`n_jobs != 1`) uses the new optional
  `[parallel]` extra (joblib); the serial default needs nothing.
- 🛰️ **`limTOD.uvbeam` module** (`[uvbeam]` extra, pyuvdata): use
  measured/simulated `UVBeam` objects in either simulation path —
  `uvbeam_beam_func` satisfies `TODSim`'s `beam_func` contract
  (Stokes I or full pseudo-Stokes IQUV via pyuvdata's own conversions,
  chromatic frequency interpolation), and `uvbeam_to_patch_beam` samples
  a UVBeam onto the simeer (l, m) grid. The azimuth convention
  (``az_uvbeam = π/2 − φ_healpix``) was locked NUMERICALLY by a
  three-way test (HEALPix path vs simeer disc path on a strongly
  displaced beam: winner 0.5%, all other candidate mappings 66–90% off) —
  the hand-derived mapping had a handedness error only the numerical
  lock caught. Out-of-coverage pixels zero-fill; error paths tested.

### Changed

- The companion digital-twin project formerly known as e-RHINO is now
  [replicant-telescope](https://github.com/zzhang0123/replicant-telescope)
  (Python package `replicant`); references updated. Historical changelog
  entries keep the old name.

## [1.3.0] - 2026-07-25

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
- 🛡️ **MPI launcher guard**: importing limTOD under `mpirun`/`srun`
  *without* mpi4py now raises a clear `RuntimeError` instead of silently
  running N duplicated serial copies (each process believing it is rank 0).
  Escape hatch for intentionally independent serial copies:
  `LIMTOD_FORCE_SERIAL=1`.
- 🚢 **First PyPI release**: `pip install limTOD`.

### Code quality (post-review polish)

- ✅ **New test suites**: full-Stokes physical invariants (spin structure,
  1D/3-row/4-row consistency, linearity, normalization) pin the polarized
  chain the Stokes-I oracle never covered; a flicker-noise boundary sweep
  over (alpha, fc, tau) corners; and an independent end-to-end
  `HPW_mapmaking` oracle that rebuilds the regularized normal equations in
  the test (unfiltered, priors, per-TOD noise variance, high-pass
  consistency, gain/injection calibration).
- 🧾 **Type annotations** across the numpy package (`mypy
  --disallow-untyped-defs` clean); the previously dead ArrayLike aliases
  are now used.
- 🪵 **Logging**: all library `print()` calls migrated to module loggers
  (`logging.getLogger`); informational output is now opt-in via standard
  logging configuration instead of unconditional stdout, and MPI ranks no
  longer duplicate messages.
- 🧩 **`HPW_mapmaking.__call__` decomposed** into `_filter_and_stack`,
  `_build_priors`, and `_normalize_noise_variance` (behavior pinned
  bit-exact by the new oracle tests before refactoring).
- 📓 Notebooks that require external, non-shipped data
  (`demonstration.ipynb`, `RotatingBeam/baseSim*.ipynb`) now carry a
  warning cell up top instead of failing with a bare `FileNotFoundError`.

### Documentation

- 📚 **Restructured**: README is now a concise landing page (install,
  quick starts, citation); detailed guides moved to `docs/`
  (tod-simulation, mapmaking, theory & conventions, api-reference,
  limtod-jax). Fixed duplicated/garbled parameter listings, broken
  fences, and case-mismatched notebook links from the old monolithic
  README.

### Packaging

- Version is single-sourced from `pyproject.toml`: both
  `limTOD.__version__` and `limtod_jax.__version__` read the installed
  distribution metadata (the hardcoded copies had drifted once).
- Modernized metadata (SPDX license expression + `license-files`,
  Python 3.12/3.13 classifiers, Documentation/Changelog URLs) and a lean
  sdist (packages + docs + CHANGELOG; notebooks/data stay on GitHub).

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

Pre-release full-code review (independent numpy-side and JAX-side passes)
caught and fixed, with regression tests for each:

- 🐛 **`horizontal_mask` orientation**: the horizontal-frame mask was
  rotated with `elevation=0` instead of the zenith pointing
  (`elevation=90`), tipping it 90° onto the horizon — every masked
  simulation placed the mask at the wrong sky location. Fixed; a
  regression test pins the mask cap to (RA = LST, Dec = latitude).
- 🐛 **Silent NaN from zero-sum beams**: `normalize_beam`/`normalize=True`
  divided by the beam's pixel sum unguarded; a fully-truncated or
  fully-masked pointing silently poisoned the TOD with NaN. Now raises a
  clear `ValueError` naming the pointing conditions; negative beam sums
  scale all Stokes rows consistently (previously Q/U/V were left
  unscaled).
- 🐛 **Flicker-noise error handling**: `aux_int` no longer swallows
  mpmath errors into a printed message plus a fabricated `inf`
  (it raises with context); `flicker_corr`/`sim_noise` explicitly reject
  the singular `alpha=1` exponent instead of failing type-dependently.
- 🐛 **`wiener_filter_map` short TODs**: the default rolling-variance
  estimator crashed with an opaque matmul error for TODs shorter than its
  100-sample window; the window now caps at the TOD length and mismatched
  `noise_variance` lengths raise a clear error.
- 🐛 **`HPW_mapmaking` prior routing**: mixed 1D/2D
  `Tsys_other_prior_inv_cov_group` elements were all routed by the FIRST
  element's dimensionality, silently discarding off-diagonal covariance
  entries; each element now routes by its own shape.
- 🐛 **`generate_gaussian_field(seed=None)`** no longer reseeds the global
  NumPy RNG from OS entropy (which clobbered callers' reproducibility
  seeding); it only seeds when a seed is given.
- 🐛 **Bugs surfaced by the typing pass** (all regression-tested):
  `HPW_mapmaking(return_full_cov=True)` crashed unpacking
  `wiener_filter_map`'s 3-tuple — the posterior covariance is now returned
  as the final element; `wiener_filter_map` no longer hits a `NameError`
  when the covariance is uncomputable (clear `LinAlgError` instead);
  `Tsys_others_operator_group` accepts the documented bare-2D-array form
  (and validates its length against the number of TODs);
  `sim_noise` accepts a plain Python list `time_list` as documented.
- Pointing-array length mismatches now raise upfront (zip previously
  truncated silently); MPI-unsafe unconditional prints are rank-0-gated;
  `HPW_mapmaking` argument validation raises `ValueError` instead of
  `assert` (which `python -O` strips); example scripts' obfuscated
  serializer imports (written to dodge a security hook) were replaced with
  plain, warning-commented imports.
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
- 📓 **Example Notebooks**: Added comprehensive Jupyter notebook demonstrating the full map-making workflow ([examples/mm_example.ipynb](https://github.com/zzhang0123/limTOD/blob/main/examples/mm_example.ipynb))

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
