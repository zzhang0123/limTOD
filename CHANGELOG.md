# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 📻 **`limTOD.tris`** — an offline bridge to the public TRIS archive, laid out
  as a package (`archive`, `sky`, `beam`, `geometry`, `noise`, `inference`,
  `mapmaking`). Strict readers for the four LAMBDA text products, TRIS→limTOD
  convention conversion, and rank-gated low-dimensional profile fits.
- 🛰️ **Cut-based TRIS beam** — `tris_cut_beam_map` / `tris_cut_beam_func` build
  the beam from the archive's own E- and H-plane cuts (0–176°, nulls to −60 dB)
  instead of a main-lobe Gaussian. Forward-modelling a 600-MHz sky with the
  Gaussian instead moves the predicted ring by ~0.94 K rms, against a 0.010 K
  statistical error. `tris_horizon_mask` supplies the horizon treatment the
  cut-based beam now makes necessary.
- 🌡️ **Temperature convention** — `cmb_monopole_rj_k`,
  `to_tris_temperature_convention` and `galactic_spectral_index`. Established
  from the data: TRIS temperatures are Rayleigh-Jeans and include the CMB
  monopole (the alternative gives a Galactic index of −2.1, which is not
  synchrotron).
- 🗺️ **Map-making inputs** — `build_tris_mapmaking_inputs` assembles the
  sky→sample operator (verified against `generate_TOD_sky` to 1e-15 K), the
  noise model, beam-coverage diagnostics and an optional zero-level nuisance
  parameter, ready for `wiener_filter_map`. `monopole_degeneracy` and
  `implied_monopole_prior_sigma_k` expose the near-exact degeneracy between a
  free zero level and the sky monopole.
- 📊 `TRISLinearFit` now reports `chi_square`, `degrees_of_freedom` and
  `reduced_chi_square`.

### Changed

- 🔗 **The companion digital-twin project is now
  [rheplicant](https://github.com/RHINO-Experiment/rheplicant)** (Python
  package `rheplicant`), moved to the RHINO Experiment organisation; the docs
  and the `limtod_jax` module docstrings that still said
  `replicant-telescope` are updated, as is `NativeLimTODProjector`, renamed
  upstream to `GeneralPointingProjector`. Historical changelog entries keep
  the old names.
- 🧭 **"Who uses this" now lives in exactly one place.** limTOD is upstream —
  it depends on nothing that consumes it — so naming a specific downstream
  pipeline across five pages read as a coupling that does not exist. The name
  is now confined to a new `Downstream` section on the docs front page (and
  the README's one-line twin); the body pages and the `limtod_jax` module
  docstrings describe the capability instead and link there when a reader
  actually needs the pointer.

### Docs

- 📈 **`docs/driftscan.md` now shows the m-mode path rather than only
  describing it.** Ported the drift-scan narrative from the rheplicant
  documentation and rewrote it against limTOD's own API: four generated
  figures (a sidereal-day waterfall, the m-mode spectrum, the generic-vs-m-mode
  agreement at float64 roundoff, and wall-clock scaling against band-limit),
  side-by-side `generate_tod_sky` / `DriftScanMmode` cards, and the measured
  headline — 69 s against 58 ms at `lmax` 191, agreeing to 1.2e-15.
- 🌡️ **Two sections that were only in docstrings before.** *Is the output a
  temperature?* measures the `normalize=False` trap (a hand-normalized beam
  map still carries +0.19 % at `nside` 16, +4.2 % at `nside` 8) and explains
  why `op.mmodes()` now refuses a normalized operator. *Where does the beam stop?* documents
  `horizon_truncated_beam` / `horizon_beam_fraction` / `horizon_partition_weights`
  — the f_sky split, the T_collected weighting, and the horizon-ring
  convention that is worth ~8.6 K of a 200 K effect.
- 🔒 **The two TRIS pages are no longer published.** `limTOD.tris` is still
  unreleased, so `docs/tris.md` and `docs/api/tris.md` are excluded from the
  documentation build by default and restored with a `-t tris` tag:
  `python -m sphinx -b html docs docs/_build -t tris`. Opt-in rather than
  opt-out on purpose — keying it off an environment variable Read the Docs
  happens to set would publish the pages the moment that variable changed.
  Note that the changelog entries below still describe the module, and the
  changelog *is* published.
- 📐 **Documentation content column widened to 1100px** (`sphinx_rtd_theme`
  caps it at 800). The pages carry two-panel figures, wide comparison tables
  and side-by-side cards that were all being squeezed; the drift-scan
  waterfall in particular now renders near its native size. Running prose is
  capped separately at 45rem (~90 characters, the measure the 800px column
  used to give) so only the things that wanted the room take it: figures,
  tables, cards, code blocks, autodoc signatures and parameter tables. The cap
  covers admonition boxes and docstring prose too — a warning and a docstring
  are running prose, and were the worst of the 125-character lines.
- 🧹 **Read the Docs builds clean.** The four warnings it had been carrying —
  invisible because `.readthedocs.yaml` sets `fail_on_warning: false` — are
  gone. `limTOD.visual` could not import there (matplotlib is in the `dev` and
  `notebooks` extras but was missing from `docs`, and RTD installs only
  `.[docs,jax]`), so its API section rendered empty; it is added to `docs`.
  That alone would not have filled the section, because neither
  `view_patch_map` nor `gnomview_patch` had a docstring for `:members:` to
  pick up — both now do. The other two were `conventions` label collisions:
  `docs/cstbeam.md` restates the `limTOD.cstbeam` docstring and then autodocs
  it onto the same page, and `limTOD.uvbeam` used the same section title, so
  those docstring sections are now `CST conventions` and `UVBeam conventions`.
- 🩹 **`limtod_jax` no longer lists horizon masking as out of scope** — it has
  been in the module since 1.9 (`horizon_truncated_beam`,
  `horizon_masked_beam_alm`, `horizon_beam_fraction`), so the page now points
  at the section documenting it instead of disclaiming it.
- 🖼️ `docs/make_engine_figures.py` generates those figures from live code
  (`--smoke`, `--replot`); `sphinx-design` is now a docs dependency, with
  `docs/_static/custom.css` sizing card code against the RTD theme's 800px
  content column.

### Fixed

- 📐 **`DriftScanMmode.mmodes()` no longer disagrees with `__call__`.** It
  forwarded neither `normalize` nor `ones_alm`, so a `normalize=True`
  operator returned the m-modes of the *un-normalized* TOD — wrong by the
  beam's solid angle (×33 for a 20° FWHM beam at `nside` 16 / `lmax` 47,
  ×165 for 45°), silently. It now raises `ValueError` naming both
  replacements. Dividing the projection by the scalar $d_0$ is deliberately
  not offered: division by the $t$-dependent denominator is not diagonal in
  $m$, so the approximation's per-mode error reaches 2e-4 in the high-$m$
  tail, and it would leave `mmodes()` and `adjoint()` normalized by
  different quantities.
- 🔢 **TRIS common-mode whitening is now exact for any `sigma_c`.** The dense
  `diag(σ²) + σ_c²11ᵀ` Cholesky kept succeeding after the matrix had lost
  positive definiteness in float64: at `σ_c = 1e5` coefficients were wrong by
  ~190σ with no exception and a healthy-looking condition number.
  `TRISNoiseModel` applies the rank-1 term analytically instead.
- 📄 Readers' dependence on universal-newline mode is documented and
  regression-tested; two of the four public products use a bare CR as their
  record separator.
- ⚖️ Beam-factory arguments are validated at construction, validation errors name
  the offending column, array-carrying models no longer raise from `__eq__` or
  `hash()`, and `fit_tris_linear_model` can check the design against the ring's
  own RA sampling.

## [1.10.0] - 2026-07-31

### Added

- 📡 **`limTOD.cstbeam` — CST Studio far-field exports as beam maps.** The
  sibling of `limTOD.uvbeam` for the other format a horn arrives in: CST
  Microwave Studio far-field ASCII, one file per frequency. Three entry points
  — `read_cst_farfield` (one file onto its own `(theta, phi)` grid),
  `cst_beam_maps` (a directory onto HEALPix maps, frequency-interpolated) and
  `cst_beam_func` (the `beam_func(freq=..., nside=...)` callable `TODSim`
  takes). Needs only `healpy` and `scipy`, both base dependencies, so there is
  no extra to install.

  Moved from `rheplicant.radio.beams`, which now delegates to it: how a
  measured beam becomes a beam map is limTOD's subject, exactly as the horizon
  partition of 1.9 is. Downstream keeps only the placement.

  `cst_beam_func` validates the directory and the options at construction
  rather than at the first channel, and **caches each file's HEALPix
  resampling across calls** — a sweep of 200 channels over a 61-file directory
  parses each file once instead of hundreds of times.

  Conventions are documented and tested rather than assumed. CST `Theta` maps
  onto the HEALPix colatitude (boresight at the pole); `Abs(Dir.)` is power in
  dBi and comes back as linear power, **unnormalized**, so the consumer's own
  quadrature is what the band limit cancels against. The azimuth cannot be
  locked numerically the way `uvbeam`'s is — which physical direction the CST
  `+x` axis points is a fact about the as-built horn and is simply not in the
  file — so `phi0_deg` and `phi_sense` expose the offset and the handedness,
  their defaults are an assumption to check rather than a result, and the tests
  lock that the knobs act correctly: `phi_sense` is a reflection about
  `phi = 0` rather than a relabelling, `phi0_deg` a rotation that conserves the
  integral. For a horn with 30-60 % azimuthal structure the handedness is not a
  detail, and it leaves every integral, peak and symmetric diagnostic unchanged
  when it is wrong.

  Frequency is in MHz throughout, as elsewhere in limTOD (rheplicant's Hz is
  converted at its own adapter). Extrapolation beyond the simulated band is
  refused.

- 🎭 **Full Stokes in `limtod_jax` — the harmonic chain, generic and drift-scan
  alike.** Pass the static `npol` (1, 3 or 4) and give the alms a leading
  Stokes axis `(..., npol, n_alm)` of packed `T`/`E`/`B`/`V` rows — the healpy
  analysis of limTOD's `I`/`Q`/`U`/`V` maps, `V` analysed separately as numpy
  limTOD does. The rows are **contracted** into each TOD sample
  (`V_t = Σ_row ⟨R_t b_row, s̃_row⟩`), which is numpy limTOD's
  `np.sum(beam_map * sky_map)`; the TOD stays `(n_time,)`.

  Covers `generate_tod_sky`, `generate_tod_sky_adjoint`, `beam_weighted_sum`,
  `beam_alm_at_reference`, `mmodes_from_sky`, `driftscan_tod`,
  `driftscan_tod_adjoint` and `DriftScanMmode` (which infers `npol` from a 2-D
  `beam_alm` in `from_pointing`). New module `limtod_jax.stokes` states the
  contract in one place.

  **It costs the formalism nothing structural**, and that is the point. Three
  facts, each locked numerically in `tests/limtod_jax/test_polarisation.py`
  rather than trusted on paper: (1) spin-weighted alms rotate with the *same*
  Wigner-D as scalar ones, so healpy's 3-row `rotate_alm` is **bit-identical**
  to the scalar rotation applied per row and there is no spin-2 rotation
  anywhere in the package; (2) spin-2 harmonics are orthonormal, so the pixel
  dot splits row-wise and the quadrature-alm exactness contract survives — the
  E-B cross terms are purely imaginary under the real-field symmetry and drop;
  (3) a rotation about z gives `δ_{m'm}·e^{−imα}` *independently of spin*, so E
  and B pick up the same drift phase as T. The m-mode expression therefore
  gains exactly one sum, `Ṽ_m = Σ_row Σ_l conj(B_row,lm(ref))·S̃_row,lm`, and
  the phase law, the FFT fast path and the block-diagonal-in-m structure are
  untouched.

  Verified as the acceptance criterion: **drift-scan m-mode TOD == generic JAX
  TOD == numpy `limTOD.simulator.generate_TOD_sky`** for npol ∈ {1, 3, 4}, with
  and without `normalize`, on direct and FFT synthesis, at ~1e-15 (nside 8 and
  nside 16) against the numpy oracle — including a zenith-gimbal pointing, lat = 0/±90,
  LST wrapping past 360°, Δ = 0, and E-only / B-only beams. The numpy leg runs
  through healpy's own polarised transforms end to end, making it the **first
  independent oracle numpy limTOD's full-Stokes chain has had** (it was
  previously pinned only by physical invariants — see the header of
  `tests/test_stokes_and_boundaries.py`).

  `normalize` divides every row by the rotated **Stokes-I** beam's pixel sum,
  matching `pointing_beam_in_eq_sys` — and necessarily so, since a beam with
  zero net Q would make a per-row normalizer singular.

- 🧵 **`rotate_flm_2d` accepts one leading axis**, sharing the Risbo recursion
  across the rows. This is not `jax.vmap` over rows and must not be: every
  Stokes row shares one rotation, so vmapping would repeat the only
  O(lmax³) step `npol` times. A 3-row beam now costs ~1× that step, not 3×.
  Hoisting the full Wigner-d plane into `dl_array` achieves the same but costs
  O(lmax³) *memory* (541 MB at lmax=256) where the recursion carries only
  O(lmax²) — that trade stays the caller's; this one is free. The 2-D path
  keeps its original contraction string, so **every unpolarised result is
  unchanged bit for bit**; a 1-row stack matches only to roundoff (~1 ulp,
  different association order).

- 🪓 **The horizon utilities take `npol`** — `horizon_truncated_beam`,
  `horizon_beam_fraction` and `horizon_masked_beam_alm`. The taper is a real
  per-pixel scalar, so it multiplies I/Q/U/V by broadcasting with no spin
  algebra; `f_sky` is a **solid-angle** split and therefore comes from the
  Stokes-I row alone (Q/U carry no total power to divide — a beam with zero net
  Q would make a per-row fraction singular).

- 🌀 **Spin-2 HEALPix transforms in `limtod_jax.hpx`**: `eb_to_qu`,
  `qu_to_eb_quad`, and `npol` on `alm2map` / `map2alm_quad` / `map2alm_iter`.
  These are what make the polarised horizon mask possible — it is the one place
  the chain leaves harmonic space. Convention locked numerically against healpy
  (`Q ± iU = −Σ (E ± iB)·(±2)Y`, quadrature factor `npix/4π`; every sign
  alternative loses by ≥ 1.3 relative), matching to ~1e-14.

  ⚠️ **s2fft's on-the-fly (Price–McEwen) recursion is WRONG at spin ≠ 0 on the
  HEALPix grid, and fails silently.** It renormalises with `log|d^l_mn|` and
  `1/|d^l_mn|` and accumulates with `nansum`; wherever a Wigner-d is *exactly*
  zero at a ring colatitude that gives `log 0 = -inf`, `0*inf = nan`, and the
  whole `l` term is dropped. HEALPix rings sit at rational `cos(theta)`, so the
  exact zeros really occur — at `cos(theta) = ±2/l`, killing l = 3, 6, 8, 16,
  32, 48… depending on nside. MW/GL/DH never trigger it, and neither does
  spin 0 (so nothing previously in this package was affected). Measured cost of
  using it regardless: 4–6 % on the synthesis, tens of percent on the affected
  multipoles alone, and a silent 1e-3…1e-2 break of the pixel-dot == alm-dot
  exactness contract. `recursion="auto"` routes back to the same code.

  `hpx` therefore uses the **precompute** transforms with `recursion="risbo"`.
  The price is a dense `(n_theta, L, 2L-1)` complex128 kernel,
  O(nside·lmax²) — 0.5 MB at nside 8 / lmax 23, 16 MB at nside 32 / lmax 63,
  but ~1 GB at nside 128 / lmax 255. Cached per `(L, spin, nside, forward)`;
  masking is a one-off beam preparation, but this does cap the resolution at
  which a polarised beam can be masked inside JAX. The spin-2 path also needs
  `nside >= 2` and `lmax + 1 >= 2*nside` — s2fft asserts both without a
  message, `limtod_jax` raises a stated `ValueError`.
  `test_onthefly_backend_really_is_broken` pins the defect: if it ever starts
  failing, s2fft has fixed the recursion and the kernel can be dropped.

### Changed

- `limtod_jax.projection.generate_projection_rows` now **rejects** a 2-D
  `beam_alm` explicitly. It previously rejected it via the `rotate_flm_2d`
  shape guard; now that a Stokes stack is a legal rotation input, the guard has
  to be stated where the restriction actually lives. A polarised projection
  *matrix* is the one thing still missing — a row is a pixel-space beam, so it
  would need the spin-2 synthesis on every pointing rather than once.

- **The Stokes axis is checked against `npol` itself, not merely for
  self-consistency.** A leading axis is now validated at every entry point
  that has one, because each of these previously returned a finite, plausible,
  wrong number (found in review, each now pinned by a test in
  `TestSilentFailureSurface`):
  - a 5-row *frequency* stack was contracted into one sample by
    `beam_weighted_sum` for any `npol` — the axis was only compared against
    the other argument, which agreed with it perfectly;
  - `npol=4` on a 3-row array fabricated a V row, because JAX **clamps**
    out-of-bounds indices, so `alm[3]` silently returned `alm[2]`
    (`alm2map`, `map2alm_quad`);
  - a `ones_alm` with a Stokes axis broadcast instead of failing, silently
    giving the TOD an extra axis — it is always one unpolarised row, since the
    denominator is the Stokes-I pixel sum;
  - 2-D alms with `npol=None` flowed through `generate_tod_sky` as an
    undocumented frequency batch once `rotate_flm_2d` learned to accept a
    leading axis. Restored to a `ValueError`: batch with `jax.vmap`.

### Fixed

- **The spin-2 precompute kernel is cached as NUMPY, not as a jax array.**
  s2fft's `spin_spherical_kernel_jax` is built from `jnp` ops, so the first
  call inside a `jit` trace returns a *tracer* — which `lru_cache` then stored
  process-globally, making every later call outside that trace fail with
  `UnexpectedTracerError` for the rest of the session. Order-dependent, and so
  invisible to a suite that happens to warm the cache eagerly: jitting the
  documented polarised-mask entry point first was enough to poison the
  process. The kernel is a pure function of static ints, so numpy is the
  honest representation, and `jnp.asarray` at the use site stages it as the
  compile-time constant it always was.

## [1.9.0] - 2026-07-30

### Added

- 🧭 **The horizon as a PARTITION, not just a mask.**
  `horizon_partition_weights(nside)` — 1 above the horizon, 0 below, **0.5 on
  it**. This is a different object from `horizon_weights`, and the difference
  is not cosmetic. `horizon_weights` is a *mask*: what to multiply a beam by
  before re-analysis, and its hard cut is a strict `el > 0`, so the ring of
  pixels centred exactly ON the horizon (4·nside of them — their elevation is
  exactly zero, not nearly) gets weight 0. As a mask that is a thin edge
  detail. As a partition it is systematic: a pixel centred on the horizon is
  half sky and half ground.

  Measured on the quantity that needs a partition — the above-horizon beam
  fraction `f_sky` that splits an antenna temperature into its sky and ground
  shares — against a projector run on a sky map with the ground painted in, at
  a latitude where the horizon is fixed in celestial coordinates. On a ~200 K
  effect at nside 16: the ring counted as nothing costs **−8.6 K**, as all sky
  **+8.7 K**, and half **+0.005 K**. The two one-sided errors are symmetric and
  halve with nside — the signature of a miscounted ring, not of anything
  harmonic. `horizon_weights` is unchanged; masking semantics were never wrong.

- ✂️ **`horizon_truncated_beam(beam_map, *, nside, el_deg=90, apod_deg=0)`** —
  cut a beam MAP at the horizon, and get the surviving fraction back with it.
  For a drift scan the pointing is fixed, so the horizon is fixed and the
  truncated beam is a **constant**: one elementwise multiply, done once, before
  the single analysis the caller was going to run anyway.

  Beside `horizon_masked_beam_alm`, which masks the ALMS and therefore pays a
  Wigner rotation, a synthesis, an iterative re-analysis and a rotation back on
  **every call** — 14.6 ms against 1.79 ms unmasked at nside 16 / lmax 47,
  **8.2×**, of which the re-analysis alone is 65%. The map path costs **1.04×**.
  The two agree to 2.8e-5; the residual is the alm→map→alm round trip the
  masking path takes *before* it masks.

  `el_deg = 90` is exact and anything else is refused, for a reason rather than
  a limitation: the mask is a pure function of elevation and this chart puts the
  ZENITH at the pole, while a beam-local map puts the BORESIGHT there. At a
  zenith pointing those poles coincide, so the charts can differ only by a
  rotation ABOUT that shared pole — which a pure-elevation function cannot see.
  Azimuth and self-rotation are therefore irrelevant and no rotation is needed.
  Away from zenith the poles part and `horizon_masked_beam_alm` is the tool.

- 📐 **`horizon_beam_fraction(beam_alm, az, el, selfrot, *, nside, lmax)`** —
  the same fraction for any fixed pointing, from beam-local alms, using the
  same (az, el, selfrot) sub-chain as `horizon_masked_beam_alm` so the two
  describe one beam. Computed in pixel space on purpose: the band-limited
  masked beam is a Gibbs approximation to a discontinuous target and its own
  solid-angle integral is off by ~0.7% at nside 16 / lmax 47, because
  `map2alm` of a sharply cut map does not preserve the mean. Using that as
  `f_sky` leaves −17 K of the 200 K bias.

  Prompted by RHEPLICANT, which needs `f_sky` to split an antenna temperature
  and had been computing it on its own side — beam-weighted-sky physics belongs
  here.

### Tests

- `tests/limtod_jax/test_horizon_partition.py` (25 tests): the painted-ground
  closure at latitude 90 (where the horizon coincides with the celestial
  equator and stops moving with LST, so the reference is computable rather than
  arguable), the symmetric ±8.6 K cost of the one-sided ring conventions, the
  two fraction routes against each other, and the zenith azimuth-invariance
  that makes the map path exact.

## [1.8.0] - 2026-07-26

### Fixed

- 🛡️ **`dl_array` is now shape-checked.** JAX *clamps* out-of-bounds integer
  indices rather than raising, so a Wigner-d plane built at a different
  band-limit than the call was accepted silently and rotated by the wrong
  sub-block — finite, no warning, ~100% error, under `jit` too. Harmless
  until now because `dl_array` was an internal parameter of
  `rotate_flm_2d`; this release is what exposes it publicly (via
  `dl_plane_for_pointing` and the new `rotate_alm` / `beam_alm_at_reference`
  arguments), so the guard lands with it. Shapes are static, so the check is
  free and jit/vmap/grad-safe. The superset case (a plane built at a *larger*
  lmax) is included: it looks reasonable, and it silently zeroed the low-ℓ
  coefficients. `docs/limtod-jax.md` now points at `dl_plane_for_pointing`
  (which takes `lmax`, like every other public entry point) and spells out
  that the lower-level `generate_rotate_dls` takes `L = lmax + 1`.

### Performance

- ⚡ **`tod_from_mmodes` and `_zeta` pick between a phase matmul and the
  sequential accumulation on a static size threshold** (`_PHASE_MATRIX_MAX`,
  10⁷ phase-matrix entries). The `lax.scan` / `lax.map` forms never
  materialize the `(lmax+1, n_time)` phase matrix, which is right at large
  `n_time` — 133 MB at `n_time=86400, lmax=191` — but below the threshold
  they were paying an enormous premium to avoid an allocation that does not
  matter. Measured at lmax=191 / n_time=512, one forward synthesis:

  | n_freq | matmul | scan | speed-up |
  |---|---|---|---|
  | 1 | 0.30 ms | 0.49 ms | 2× |
  | 8 | 0.40 ms | 3.88 ms | 10× |
  | 32 | 0.52 ms | 15.16 ms | **29×** |

  The gap grows with `n_freq` because the matmul becomes a real GEMM while
  the scan stays a chain of 192 tiny sequential kernels. The extra memory is
  **1.57 MB regardless of `n_freq`** — `dphi` is unbatched under `vmap`, so
  the phase matrix is shared across frequencies, not duplicated. Forward and
  adjoint switch on the SAME condition, so the dot-product identity holds
  whichever branch each side takes (tested).

- ⚡ **`rotate_alm` and `beam_alm_at_reference` accept a precomputed
  `dl_array`**, plus a new `dl_plane_for_pointing(lat, az, el, selfrot, lmax)`
  to build it. The Wigner-d plane depends on the polar angle alone, and LST
  enters the zyz composition in the first-applied slot — so it shifts `psi`
  and never the plane. A drift scan can therefore build one plane and reuse
  it at every LST, **bit-for-bit identical** (tested across the LST fixture),
  skipping the Risbo recursion: measured at lmax=127, **44.0 ms → 2.8 ms
  (15.6×)**. This is the only way to amortize the rotation when the BEAM is
  the fitted parameter, where the reference-frame trick cannot be used
  because gradients must reach the beam-local alms.

### Fixed

- 🐛 **The Wigner-d plane no longer forces float64 in an x64 session.**
  `wigner.py` computed its dtype as `jnp.result_type(beta.dtype,
  jnp.zeros(0).dtype)`; the second term is the session default, so under x64
  a deliberately float32 `beta` was silently promoted, doubling the largest
  array in the rotation (215 MB at lmax=191). The floor is now float32, so
  the caller's choice wins: float32 angles give a float32 plane (67 → 33 MB
  at lmax=127) agreeing to 5.5e-6, which is float32 roundoff — the Risbo
  recursion is float32-stable. float64 angles are unaffected, so the default
  path does not change.

### Documentation

- 🧲 **Polarization: the basis follows from the beam convention, the
  handedness does not** (`docs/theory.md`, new subsection). Q/U live in
  the same tangent basis as everything else, so there is no separate frame
  to declare, and the transport is automatic — `map2alm` decomposes
  (I,Q,U) into (T,E,B), E/B rotate as scalar alms without mixing, and
  synthesis returns Q/U in the correctly rotated local basis (the position
  angle co-rotates; the spin-0 mistake is excluded by two orders of
  magnitude).

  But a convention change is harmless only if it COMMUTES with that
  transport, and (Q,U) transport is a rotation in the (Q,U) plane.
  Measured, applied to beam and sky together: a **rotation** of the
  reference axis is harmless (1.7e-16) and V's sign is harmless (exactly
  0, V is spin-0) — while **U → −U handedness (IAU vs CMB) is not**
  (4.3e-2), because a reflection anticommutes with the rotation
  (F R F = R⁻¹). So beam and sky must share a handedness; the error is
  O(polarized fraction) and invisible to every Stokes-I check. Stated as
  a caller contract, in the same class as the UVBeam azimuth adapter.

### Documentation

- 🧭 **The North-Pole-zenith reading of the identity now states its mount
  azimuth** (`docs/theory.md`): it is `A = 0°`, **not** `A = 180°`, and
  more generally `A = Θ_LST + ψ` (at `lat = 90°, e = 90°` the chain
  collapses to `Rz(ψ − A + Θ_LST)`). The docs described this
  configuration — "an antenna at the terrestrial North Pole pointing at
  its zenith" — without ever giving the azimuth, which is exactly the
  gap that makes the question ambiguous: at `e = 90°` the boresight is
  the zenith whatever `A` is, so azimuth only *rolls* the beam about the
  boresight. The tempting wrong answer 180° comes from conflating the
  mount azimuth with the compass direction the `φ = 0` axis lands on
  (the south point, at `A = 0°`). Multiplying a beam map by a sky map
  with no rotation is the `A = 0°` configuration; `A = 180°` would
  rotate the beam by 180° in `φ`.
- 🧭 **Promoted to the most prominent slot on every entry path**, since it
  doubles as the fastest way to *determine* the convention: a titled
  callout at the very top of the beam-convention section in
  `docs/theory.md` (with a runnable three-line check), one on the docs
  landing page, one in the README (the PyPI landing page), and the
  `docs/driftscan.md` box where readers first hold a beam map and a sky
  map at once.
- The check is deliberately **comparative, not an equality test**: of
  azimuths 0/90/180/270 only 0 reproduces the plain product (measured
  2.1e-3 vs 1.3e-1 / 2.7e-1 / 1.4e-1). The winner is not zero because
  `generate_TOD_sky` re-analyses the beam internally, and HEALPix
  analysis/synthesis is not exactly idempotent — a fact the docs now
  state, so the residual is not mistaken for a convention mismatch.
  Two traps are called out because both silently defeat the check: a beam
  symmetric under `φ → φ + 180` (including any beam that is a function of
  `θ` alone) cannot see a 180° roll at all, and an un-band-limited beam
  inflates the residual to several percent.

### Tests

- Polarization-convention pins (`tests/test_stokes_and_boundaries.py`):
  the (Q,U) reference axis is free (rotation invariance at two angles),
  V's sign is free, and — asserted **positively** — a Q or U reflection
  is NOT free. That last one is the load-bearing test: if a refactor ever
  made it invariant, (Q,U) would have stopped being transported at all.
  Each invariance test also checks the row actually contributes, so it
  cannot pass vacuously.
- Two North-Pole-zenith pins in `tests/test_beam_orientation.py`: the
  identity holds at `A = 0°` and demonstrably fails at 90/180/270°, and
  the general rule `A = Θ_LST + ψ` holds across four (LST, selfrot)
  combinations. Map-level comparisons, with the default `1e-10`
  truncation disabled — it is a nonlinear cleanup applied to the rotated
  map only, so it breaks a map-vs-map identity check that `argmax`-based
  pins never notice.

## [1.7.0] - 2026-07-26

### Added

- ⚡ **FFT fast path for uniformly-sampled drift scans**
  (`limtod_jax.driftscan`): when the LST grid is uniform over a full
  sidereal turn, the m-mode time synthesis is an inverse real FFT and the
  analysis a forward one — O(n_time·log n_time) *independent of lmax*,
  measured 19–51× faster than the direct phase sum end-to-end (and
  identical to roundoff). New `tod_from_mmodes_uniform` / `mmodes_from_tod_uniform`
  (the latter carries measured TOD into m-space in one FFT), plus static
  `uniform=` / `uniform_sampling=` opt-ins on `driftscan_tod`,
  `driftscan_tod_adjoint` and `DriftScanMmode`. The adjoint uses the exact
  FFT counterpart of the synthesis, so it remains an exact transpose
  (dot-tested in both modes).

  The dispatch is deliberately **static and never auto-detected** — a
  choice made on the *values* of `dphi` would be impossible under `jit`.
  The sampling-theorem condition `2·lmax < n_time` is always enforced
  (a shape statement), and uniformity itself is enforced in two layers:
  a clear `ValueError` from the now-public `check_uniform_grid` while the
  values are concrete, and a pure-JAX NaN guard when they are traced.
  The second layer is not optional — an eager-only check is bypassed by
  ANY `jit` wrapping (arithmetic inside a trace yields a tracer even for a
  compile-time-constant grid), and a uniform *half*-turn grid then returned
  a silently 74%-wrong TOD. The guard costs 30–45% of the raw FFT call,
  leaving a 10–100× net win, and poisons only the offending row under
  `vmap`. The uniformity tolerance is **dtype-scaled** so that an
  exactly-uniform float32 grid (~1e-7 rad of `deg2rad` representation
  error) is not misread as irregular, and `phase0` enters the output-dtype
  promotion so the FFT path is never less precise than the sum it
  reproduces.

### Fixed

- **Uniformity tolerance: the float64 branch was 1.1e4x too loose.** The
  `1e-9` absolute floor swamped the dtype-scaled term (`64·eps·2π` =
  8.9e-14), so an f64 grid with ~1e-9 rad of jitter was accepted and gave
  up to ~2e-6 relative error in a path documented as float64-roundoff
  exact. The floor is gone; measured headroom over the worst legitimate
  representation error is ~100x (f64) and ~40x (f32). A non-floating
  `dphi` dtype now raises instead of silently yielding a zero tolerance.
  The docstring's claim that the f32 bound "stays orders below the
  transform error" is corrected: it reaches ~1e-3 (lmax = 256) to ~1e-2
  (lmax = 1024), the same order — x64 is required for the roundoff
  contract. NOTE for downstream adapters: do not upcast a narrower grid
  before checking, or this bound rejects it (an f32 degree grid upcast to
  f64 deviates ~3e-7, 3e6x the f64 bound).

### Tests

- The tolerance is now pinned in the repo that owns it, on both sides of
  the bound and in **both dtypes**: limTOD's suite is x64-only, so every
  call previously resolved to the old flat floor and a mutant returning a
  constant survived the whole suite (only the downstream float32 repo could
  see it).
- The uniform path's LST gradient is pinned: the contract makes `dphi`'s
  Jacobian a single column at index 0, which must equal the direct path's
  row sum (the derivative w.r.t. a global LST shift). This documents the
  one-parameter semantics and kills a `stop_gradient`/static-phase0
  refactor, which is forward-identical and was otherwise invisible.

### Documentation

- `docs/driftscan.md` gains a measured cost breakdown per stage: the
  rotation is the only O(lmax³) step and happens once; the direct
  synthesis can exceed it at low lmax over a full day; and once the
  rotation is amortized the pixel↔harmonic transform becomes the
  bottleneck (175 ms at lmax = 256, more than the rotation) — which
  parameterizing the sky in harmonic space removes entirely. Also notes
  the block-diagonality in m and the linear growth of compile time with
  lmax (Python-unrolled ℓ loop).

## [1.6.0] - 2026-07-26

### Added

- 🌀 **Drift-scan m-mode path** (`limtod_jax.driftscan`, pure JAX +
  equinox): the drift-scan special case of the sky→TOD simulator in
  harmonic space, following the standard m-mode formalism in the
  conventions of the reference note (MT interpretation, fringe ≡ 1).
  One Wigner rotation for the whole
  scan (`beam_alm_at_reference`) plus per-m phases replaces the generic
  per-sample rotation — O(lmax³ + n_time·lmax) vs O(n_time·lmax³) —
  and agrees with `generate_tod_sky` (and numpy limTOD) to float64
  roundoff, `normalize` branch included. Public API: `DriftScanMmode`
  (an `equinox.Module` operator with `__call__`/`mmodes`/`adjoint`),
  `mmodes_from_sky`, `tod_from_mmodes`, `mmodes_from_tod`,
  `driftscan_tod`, `driftscan_tod_adjoint` (exact transpose,
  O(n_time·lmax)). Phase-sign conventions locked numerically, never on
  paper. `equinox>=0.13` joins the `[jax]`/`[full]` extras.
- 🌄 **Horizon mask for drift-scan beams** (`horizon_masked_beam_alm`,
  `horizon_weights`): the physical below-ground cut applied in the
  horizontal frame (where the horizon is the pure-colatitude circle
  θ = 90°), with optional cosine apodization; the masked beam returns
  in the beam-local frame so it drops into either TOD path.
  `limtod_jax.hpx.map2alm_iter` added (healpy-equivalent iterative
  analysis inside JAX). Off by default — numpy limTOD does not mask.
- 📊 **Ringing study** (`docs/driftscan_ringing_study.py`, results in
  `docs/driftscan.md`): narrow beams don't need the mask (~1e-6 TOD
  effect); wide low-elevation beams do (~30%), where a hard cut leaves
  a ~0.5% TOD error that plateaus with lmax (analysis aliasing) and a
  2–5° apodization recovers 1–2 orders of magnitude. Pinned as a
  regression test.

### Documentation

- New user-guide page `docs/driftscan.md` (formalism, usage, mask
  guidance, generic-vs-m-mode decision table); `limtod_jax.driftscan`
  section on the API page.

## [1.5.3] - 2026-07-25

### Documentation

- 📐 **Beam convention restructured for readability** (`docs/theory.md`):
  the section now opens with the convention as a *mount motion* — park
  at the zenith with the chart meridians on the compass
  (φ = 0, 90°, 180°, 270° → S, E, N, W), then rotate by A in the
  azimuth-increasing sense, tilt down by 90° − e, spin by ψ — followed
  by a new three-panel figure (`docs/_static/beam-convention.svg`,
  generated by `docs/make_beam_convention_figure.py`) that includes the
  local horizontal system in every panel (the fix for what the retired
  conventions.pdf figure lacked). The coordinate-derivative formulas
  remain the normative definition; a "practical reading" line states
  which pixel responds to which source offset.

### Tests

- Azimuth compass pins: boresight landings for az ∈ {0°, 90°, 180°,
  270°} → {N, E, S, W} horizon points, verifying azimuth is measured
  from North increasing towards East (the southern/western half was
  previously untested).

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
