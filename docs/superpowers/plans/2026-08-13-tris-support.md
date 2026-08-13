# TRIS compatibility implementation plan

> **For agents:** Follow this plan task-by-task with tests first. Work only in
> `/private/tmp/limtod-tris-support`; do not modify or merge `main`.

**Goal:** Add an offline TRIS reader/adapter and rank-gated low-dimensional
inference path whose assumptions are visible in code and compact documentation.

**Architecture:** One dependency-light module, `limTOD/tris.py`, owns typed
archive models, three strict readers, a HEALPix Gaussian main-lobe adapter,
zenith pointing translation, and small dense linear-algebra helpers. Existing
mapmakers and simulator defaults remain unchanged. `tests/test_tris.py` pins the
interface and scientific failure modes.

**Stack:** Python 3.8+, NumPy, healpy, pytest, Sphinx/MyST.

---

### Task 1: Archive models and offline readers

**Files:** create `limTOD/tris.py`, create `tests/test_tris.py`.

1. Write failing tests for both RA token formats, CR/LF input, schema errors,
   nominal/effective frequency separation, 600/820 statistical uncertainty,
   asymmetric 820 zero level, repeated 2.5-GHz common uncertainty, and dB cuts.
2. Run `python -m pytest -q tests/test_tris.py` and record the expected import
   failure.
3. Implement frozen dataclasses and readers with ASCII text input, finite-value
   validation, preserved `ra_text`, degrees in `[0, 360)`, and no network code.
4. Parse dB values without changing them and expose explicit linear-power
   properties using `10**(dB/10)`.
5. Run the unit tests, then validate the ignored official files: 120, 120, 6,
   and 55 rows respectively.

### Task 2: Approximate beam and zenith geometry

**Files:** modify `limTOD/tris.py`, modify `tests/test_tris.py`.

1. Add failing tests for RING output shape, finite/non-negative values,
   peak/sum/none normalization, achromatic callable behavior, 18/23-deg
   principal-axis half power, site vs nominal declination, and `selfrot=-7`.
2. Implement `approximate_tris_gaussian_beam_map` with intrinsic E axis at
   `phi=0/180` and H axis at `phi=90/270`; label it an approximation in every
   public docstring.
3. Implement `tris_beam_func` with the existing keyword-only
   `beam_func(*, freq, nside)` protocol. Validate but do not use frequency,
   because the public archive states one common beam.
4. Implement `TRISZenithGeometry` and `tris_zenith_geometry`, preserving input
   RA samples and making latitude and E-plane offset explicit.
5. Run focused tests plus `tests/test_beam_orientation.py`.

### Task 3: Rank-gated Fourier/template inference

**Files:** modify `limTOD/tris.py`, modify `tests/test_tris.py`.

1. Add failing tests for Fourier column order, exact coefficient recovery,
   explicit uncertainty flooring, symmetric common-mode covariance, and
   rank-deficient/under-determined designs.
2. Implement `build_tris_fourier_design` with columns
   `[1, cos(alpha), sin(alpha), ..., cos(m alpha), sin(m alpha)]` when the
   constant is requested.
3. Implement covariance whitening by Cholesky, SVD rank/condition diagnostics,
   a fail-before-solve gate, and GLS coefficient/covariance/prediction outputs.
4. Require a caller-supplied scalar for common-mode covariance; do not silently
   symmetrize `AsymmetricUncertainty`.
5. Run focused tests and a forward-model sanity test through
   `limTOD.simulator.generate_TOD_sky` for a constant sky.

### Task 4: Compact convention and API documentation

**Files:** create `docs/tris.md`, create `docs/api/tris.md`, modify
`docs/index.md`, modify `docs/api/index.md`, modify `README.md`, modify
`CHANGELOG.md`.

1. Write `docs/tris.md` as the compact report: problem, archive inventory,
   confirmed/assumed/unsupported matrix, data and beam conventions, translation
   to limTOD, uncertainty model, identifiability limit, offline example, and
   unsupported claims.
2. Document every public API via autosummary/autofunction directives in
   `docs/api/tris.md`; link both pages into navigation.
3. Add one short README capability/link and one changelog entry.
4. Build Sphinx with warnings as errors and fix only regressions caused here.

### Task 5: Whole-branch verification

1. Run `python -m pytest -q tests/test_tris.py tests/test_beam_orientation.py`.
2. Run `python -m pytest -q` with serial MPI and a writable matplotlib cache.
3. Run Sphinx HTML build with warnings as errors.
4. Inspect the branch diff for accidental runtime downloads, root exports,
   changes to existing mapmaker defaults, or unsupported 2D/full-Stokes claims.
5. Obtain Python-specific and whole-change code reviews; address all high and
   medium findings and re-run affected verification.

