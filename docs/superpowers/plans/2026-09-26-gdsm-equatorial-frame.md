# GDSM sky frame: handoff (2026-09-26)

> **For agents and for the next session:** what was fixed, what was *not*,
> and how the fix was checked. The unchecked items below are open work; the
> numbers under "Expected changes" are estimates, not results.

**Problem.** `GDSM_sky_model` returned pygdsm's GSM16 map unrotated. pygdsm
documents that map as "galactic coordinates, ring format"; `TODSim` (whose
default `sky_func` it is), `generate_TOD_sky`, `limTOD.patchbeam`,
`limTOD.tris` and `limtod_jax` read every sky map as equatorial
(RA = phi, Dec = 90 - theta). So the Galactic centre was simulated at RA 0,
Dec 0, and a zenith drift at latitude +53.2 deg (Jodrell Bank) swept the
Galactic-latitude b = +53 ring, which never meets the plane, instead of the
Dec = +53 ring, which crosses it near RA 21.7 h and 4.0 h.

**Evidence.** With pygdsm replaced by a stub returning a Galactic RING map
(the GSM data hosts were unreachable from the session that did this):

- a zenith beam at LST 266.4, latitude -28.9 lands at (phi, 90 - theta) =
  (266.5, -28.6): the simulator frame is RA/Dec;
- a blob at (l, b) = (0, 0) came out of the old `GDSM_sky_model` at
  "RA 0, Dec 0.6", 93 deg from the Galactic centre; a TODSim zenith scan at
  latitude -28.9 never saw it (peak 1.1 K against 469 K after rotation).

## Done (commit "fix: GDSM_sky_model returns an equatorial map")

- [x] `limTOD/sky_model.py`: `GDSM_sky_model(*, freq, nside, coord="C")`
  rotates Galactic -> equatorial after `ud_grade`
  (`hp.Rotator(coord=["G", "C"]).rotate_map_alms(..., use_pixel_weights=False)`:
  milliseconds at the target nside, monopole exact; rotating the native
  nside-1024 map costs ~15 s per call). `coord="G"` returns the old map; any
  other value raises `ValueError`. Docstring states frame and ordering.
- [x] `tests/test_sky_model_frame.py` (new, 5 tests, pygdsm stubbed, expected
  position from astropy not healpy): GC lands at RA 266.4, Dec -28.9;
  `coord="G"` is unrotated; monopole kept; a zenith beam sees the GC at its
  transit; unknown `coord` rejected. Verified to FAIL on the old code
  (peak 93 deg off; transit sample 1.03 K).
- [x] `limTOD/patchbeam/projection.py`: docstring no longer calls GDSM
  equatorial.
- [x] `docs/tod-simulation.md`: new "Sky frame" paragraph; built-ins list
  says GDSM is equatorial.
- [x] `docs/make_engine_figures.py`: `gsm_sky()` applies the same rotation;
  module docstring corrected (the north Galactic pole never passes overhead
  at +53.2 deg: it stays 26 deg from the zenith).
- [x] `CHANGELOG.md`: Unreleased -> Fixed entry, flagging that every GDSM TOD
  changes and that the notebook and engine figures predate the fix.
- [x] Checked no caller double-rotates: the only other `Rotator` in the repo,
  `examples/TRIS/tris_limtod_walkthrough.ipynb` cell 22, rotates raw
  `pygdsm` output, not `GDSM_sky_model`.

Tests run on the committed tree (Python 3.11, healpy 1.20, no jax, no
mpi4py): all of `tests/` except `tests/limtod_jax`: 492 passed, 5 skipped.

## Not done

- [ ] **Re-run `examples/mmode_drift_scan.ipynb`** with the real GSM16 (about
  12 min). All its sky-derived outputs are from the Galactic-frame map. Then
  rewrite cell 8's numbers and fix two statements that are wrong for the
  real sky too: the Galactic centre *does* rise above the JBO horizon (to
  ~7.8 deg), and the plane crosses Dec +53 near RA 21.7 h and 4.0 h
  (northernmost +62.9 deg at RA 0.86 h), not "RA 20-1 h".
- [ ] **Regenerate the engine figures**: `python docs/make_engine_figures.py`
  (needs `limTOD[jax]` and pygdsm), then check the docstring's "cold
  stretch near LST 11h" against GSM16 (it comes from a Haslam stand-in).
  This script was edited but neither run nor syntax-checked here; no test
  imports it.
- [ ] **`TODSim` `sky_func` docstring** in `limTOD/simulator.py`: add
  "equatorial (RA = phi, Dec = 90 - theta), RING". Left to the owner, who has
  uncommitted edits in that file.
- [ ] **Labels and claims found by the sweep but not changed:**
  `limTOD/visual.py:134,139` (`gnomview_patch` hard-codes `coord='C'`, fine
  now that GDSM is equatorial, but its old outputs in `mm_example.ipynb`
  cells 12-13 showed Galactic content under RA/Dec);
  `examples/DSA/README.md` ("RA 168-193, Dec 48-58" was really l, b);
  `docs/tris.md:226-229` (template file `galactic_600mhz_ring_nside16.npy`
  of unstated frame fed to `tris_prior_from_template`, whose docstring also
  omits the frame); `docs/mmode.md:26` calls `res.d0` "the monopole mode",
  which the notebook contradicts.
- [ ] **Re-run, low priority** (self-consistent before, since GDSM was both
  input and truth, but their outputs change): `TODsim_examples.ipynb`,
  `demonstration.ipynb`, `mm_example.ipynb`, the DSA notebooks and scripts,
  `scripts/simulate_tod.py`, `scripts/make_map.py`, `RotatingBeam/*`
  (`Nside_4`/`Nside_8` use `ANT_LAT = 90`, which stared at the north
  *Galactic* pole before the fix and at the celestial pole after it).
- [ ] **Not in scope, noted:** the simulator uses apparent sidereal time
  (equinox of date) while healpy's "C" is J2000: about 0.37 deg in 2026.
  Negligible for the 30 deg notebook beam, a third of the 1.1 deg example
  beam.
- [ ] `tests/limtod_jax` not run (no jax in that session).

## Expected changes in `mmode_drift_scan.ipynb` (70 MHz, estimates)

From a Haslam 408 MHz stand-in (cora's `skydata.npz`, Galactic) scaled to
70 MHz, run through the notebook's exact scan (JBO, zenith, FWHM
30 x 70 / f deg, NSIDE 32, 359 samples): d_0 / monopole is 0.650 as-is and
0.939 rotated; a smoothing-plus-ring-average check agrees to 0.002, and a
uniform-index stand-in gives the same picture. The stand-in has more plane
contrast than GSM16 (0.650 vs the notebook's 0.712), so its deficit was
scaled down accordingly.

| quantity | now | after re-run (estimate) |
|---|---|---|
| monopole | 2757.6 K | unchanged |
| d_0 | 1964.0 K | ~2560-2650 K |
| residual d_0 - monopole | -793.7 K (-40.4%) | ~ -100 to -200 K (-4 to -8%) |
| d_0 over the band | 1165-3881 K | roughly 1600-5200 K |
| TOD peak | LST ~21 deg (North Polar Spur on b = +53) | RA ~21 h (Cygnus) |
| sigma per sample / sigma(d_0), one night | 133 mK / 7.03 mK / 3.58 ppm | ~175 mK / ~9.3 mK / ~3.5 ppm |
| `cond` | 1.000 | unchanged |
| fit-residual table, mode ladder, cell 13 sweep | | re-run; not predictable from the stand-in |

A local version of the notebook quoting d_0 = 2237.6 K (not on the
`mmode-solver` branch) should land at ~2500-2700 K: rotated, d_0 / monopole
stays within 0.90-0.98 for beam FWHM 15-60 deg.
