# TRIS support: data, conventions, and prior-driven map-making

`limTOD.tris` is an offline bridge to the public TRIS archive. The archive ships
**final beam-convolved profiles and points, not raw TOD**, so this package's job
is to translate TRIS conventions into limTOD's, supply a beam that is faithful to
the archive's own measurement, and hand back the operator/noise/prior objects a
regularized map-maker needs.

Everything in the convention tables below was re-derived from the archive files
and from `limTOD.simulator` itself. Where a statement came from the data rather
than from a header, the evidence is given.

## Archive, provenance, and scope

Source: the [LAMBDA TRIS product page](https://lambda.gsfc.nasa.gov/product/tris/tris_prod_table.html).
Every ring header names the reference: M. Zannoni et al., *ApJ* 688:12-23 (2008)
([arXiv:0806.1415](https://arxiv.org/abs/0806.1415)). Files were retrieved on
2026-08-13 into ignored `downloads/TRIS/`. Readers take explicit local paths and
never fetch anything.

| Product | Rows | Schema | SHA-256 |
|---|---:|---|---|
| `TRIS_absolute_600.txt` | 120 | RA label, K, per-row statistical K | `f0c9d99d6276e689b83af5cb4383d0b20cab2e901c3e3e91a6c1138433b84928` |
| `TRIS_absolute_820.txt` | 120 | RA label, K, per-row statistical K | `64d1af05d917015bb923c82b7786349d2b10c23278309b967043c419039499ae` |
| `TRIS_absolute_2500MHz.txt` | 6 | RA label, K, common zero-level K | `9cc153ceabf87ed78a4f9522bbb1785449c989118cc35d059a0734da319ed42d` |
| `TRIS_Beam_Profile.txt` | 55 | angle deg, H-cut dB, E-cut dB | `082fc49d5a8cba5b158bcdc301c1c3b2f1e5b668fd84088ef19c5231a907f190` |

Two of these use a **bare CR** as the record separator inside their data block
(`TRIS_Beam_Profile.txt` throughout; `TRIS_absolute_2500MHz.txt` packs its first
two records onto one physical line). The readers open in universal-newline mode
for exactly that reason, and it is regression-tested.

## Confirmed by the archive headers

| Statement | Evidence |
|---|---|
| Declination +42 deg label, two 120-point rings, six 2.5-GHz points, 55-row beam table | file headers and row counts |
| The third 2.5-GHz column is a **zero level**, not a statistical error | `# Column 3 = Zero Level uncertainty in K` |
| 820-MHz zero level is asymmetric, +0.430/-0.300 K | `# Systematic Zero Level Uncertainty = +0.430K/-0.300K` |
| 600-MHz zero level is symmetric, 0.066 K | ring header |
| The E plane is tilted **7 deg east of the meridian** | `# NOTE2 = the E-plane is tilted 7 degrees Eastwards ...` |
| The beam is the **same at all three frequencies** | `# ... the feed horns are scaled versions of a 8 GHz prototype` |
| RA labels are given every 3 deg | `# NOTE1 = The data points are given every 3 degrees` -- a *sample spacing*, not a resolution |

## Derived here from the data (not stated in any header)

**The beam's measured half-power widths are 19.155 deg (E) and 23.366 deg (H).**
Interpolating `TRIS_Beam_Profile.txt` at the exact half-power level
(-3.0103 dB) gives these; the ring headers' "beam at 3dB is 18 degrees wide
(E-plane)" is a rounded restatement, 6 per cent narrow. Use
`TRISPrincipalPlaneCuts.half_power_full_width_deg()`, which reads them off the
table.

**TRIS temperatures are Rayleigh-Jeans (antenna) temperatures and include the
CMB monopole.** The archive says only "Sky Brightness Temperature (K)". The two
rings share an RA grid, so the Galactic spectral index between them is a clean
per-sample test:

| Treatment | Galactic index across 600.5 / 817.8 MHz |
|---|---|
| CMB monopole left in | min -2.39, median -2.10, max -1.81 |
| CMB monopole removed in RJ | min **-3.12**, median **-2.91**, max **-2.63** |

Only the second is synchrotron, and it flattens toward the plane the right way
(-2.98 at the coldest RA, -2.76 at the hottest). The RJ monopole is 2.7111 K at
600.5 MHz and 2.7059 K at 817.8 MHz against a thermodynamic 2.72548 K -- a
0.014-0.020 K difference, **larger than the 0.010 K median statistical error**.
Use `cmb_monopole_rj_k` and `to_tris_temperature_convention`, not 2.725.

**The published RA labels are not on an exact grid.** Real spacings are 2.75,
3.00 and 3.25 deg, because the labels are rounded to the minute. The readers
preserve them; do not regrid.

**Each ring contains exactly one row with a zero statistical uncertainty.** A
positive `uncertainty_floor_k` is therefore mandatory for real data, not a
theoretical nicety.

## Beam and pointing translation

| Physical quantity | limTOD representation |
|---|---|
| Frequency | MHz. Nominal/effective/bandwidth: 600/600.5/0.3, 820/817.8/0.3, 2500/2427.8/3 |
| Sky and beam maps | Scalar HEALPix, RING ordering, equatorial |
| Boresight | Beam-map north pole |
| Intrinsic axes | E at `phi = 0/180`; H at `phi = 90/270` |
| Scan samples | Each RA label becomes `LST_deg`. Exact, not approximate: at a zenith park the boresight hour angle is zero |
| Park | Zenith: `azimuth_deg=0`, `elevation_deg=90` (azimuth is degenerate there) |
| Latitude | 42 deg 26 arcmin N by default; pass `latitude_deg=42.0` for the rounded label |
| Roll | `selfrot = -7 deg` |

The pointing chain was verified numerically rather than asserted. At the real
site latitude and several LSTs, the boresight lands at `RA = LST`,
`dec = latitude`; with `selfrot = 0` the E plane lands at position angle
180/0 deg (exactly the meridian) and the H plane at 90/270 deg; with
`selfrot = -7` both rotate by **+7 deg** in position angle, i.e. the E plane
sits 7 deg east of the meridian as the archive states. Because E is the narrow
plane, **the TRIS beam is narrow in declination and wide in right ascension.**

`tris_zenith_geometry` owns the orientation. Do not apply a second beam rotation.

### Which beam to use

`tris_cut_beam_map` builds the beam from the archive's own E/H cuts by
interpolating in `theta` and blending in `phi` with `cos^2/sin^2` weights.
The default `blend="db"` is the standard horn construction and is the *exact*
generalization of an elliptical Gaussian: given Gaussian cuts it reproduces
`approximate_tris_gaussian_beam_map` identically. Both blends are exact on the
principal planes; their spread in between is a fair estimate of the
interpolation error, which no public TRIS product can remove.

`approximate_tris_gaussian_beam_map` remains available but **should not be used
for quantitative work**. Forward-modelling a realistic 600-MHz sky through
limTOD with the Gaussian instead of the cut-based beam moves the predicted ring
by **0.94 K rms / 2.3 K peak**, against a 0.010 K statistical error and a
0.066 K zero-level systematic. Refitting the Gaussian to the measured widths
only improves it to 0.78 K rms -- the shape, not the width, is the problem:

| | within 10 deg | within 20 deg | within 30 deg | below horizon |
|---|---|---|---|---|
| Gaussian 18/23 | 0.479 | 0.929 | 0.9963 | 2e-20 |
| archive cuts | 0.369 | 0.814 | 0.9647 | 1.2e-04 |

The last column is why `apply_horizon_mask` defaults to true: the real beam has
genuine below-horizon response worth about **0.035 K against a 300 K ground**,
comparable to the published zero-level systematic, while the Gaussian silently
sets it to zero. `tris_horizon_mask` supplies the mask limTOD expects.

## Map-making

A single fixed-declination ring constrains, per temporal Fourier mode, only
`V_m = sum_l conj(B_lm) S_lm`, and the ~23-deg beam suppresses everything above
`m ~ 8`. About **15 numbers are measured, not 120**. Reconstruction is therefore
explicitly prior-driven: the likelihood supplies the directions the ring
constrains and the prior supplies the rest. That is a legitimate MAP estimate,
and it is what `limTOD.wiener_filter_map` computes -- but it is not a measured
full-sky map, and the objects here report enough to see the difference.

Measured behaviour on a simulated ring (nside 16, 1632 pixels in a +/-45 deg
band, 120 samples):

| Template error | per-pixel rms | beam-convolved rms |
|---|---|---|
| smooth 12% + gradient | 1.646 -> 1.328 K | 1.512 -> **0.005** K |
| white per-pixel 12% | 1.775 -> 1.788 K | 0.212 -> **0.005** K |

The ring pins what it measures to the noise level in both cases. Per pixel it
helps against smooth template error and cannot help at all against white
small-scale error -- which is the identifiability statement, made quantitative.

### The zero level is degenerate with the sky monopole

`monopole_degeneracy` returns `1 - 4e-12` for a real ring. The reason is
structural: the zero-level column is all ones, and the sky monopole's response
is `A @ 1` = `beam_coverage`, which is constant for a normalized beam on a
zenith drift scan. **A free zero level and a free sky monopole are not
separately measurable from one ring**; whatever splits them comes from the
priors. Use `implied_monopole_prior_sigma_k` to check before trusting a fitted
offset:

| implied monopole prior | `zero_level_sigma_k` | fitted offset (truth +0.0300) |
|---|---|---|
| 0.126 K | 0.066 K | +0.011 +/- 0.053 K -- prior-dominated, unreliable |
| 0.013 K | 0.066 K | **+0.0296 +/- 0.0086 K** -- recovered |

### Choosing `nside`

The beam is 19-23 deg wide. `nside=16` (3.7 deg pixels) already oversamples it
six times; `nside=32` is a comfortable ceiling. Anything finer is pure prior.
A +/-45 deg declination band retains 99.95 per cent of the beam
(+/-30 deg retains 99.1 per cent); `beam_coverage` reports what was dropped,
and the missing fraction biases the model low by that amount.

### Offline example

No downloads. Substitute your own sky template for the placeholder.

```python
from pathlib import Path

import numpy as np
import healpy as hp
from limTOD.tris import (
    build_tris_mapmaking_inputs,
    read_tris_beam_cuts,
    read_tris_ring,
    to_tris_temperature_convention,
    tris_prior_from_template,
)

archive = Path("downloads/TRIS")
ring = read_tris_ring(archive / "TRIS_absolute_600.txt")
cuts = read_tris_beam_cuts(archive / "TRIS_Beam_Profile.txt")
nside = 16

inputs = build_tris_mapmaking_inputs(
    ring,
    nside=nside,
    cuts=cuts,                    # cut-based beam, not the Gaussian
    dec_half_width_deg=45.0,
    uncertainty_floor_k=0.004,    # each ring has one zero-error row
    zero_level_sigma_k=0.066,     # 600 MHz only; 820 is asymmetric
)
print(inputs.beam_coverage.min(), inputs.monopole_degeneracy)

# A Galactic-only template (Haslam/GSM extrapolation) must be put into the
# TRIS convention before it can be compared with the archive.
galactic = np.load("galactic_600mhz_ring_nside16.npy")     # K, RJ, no CMB
template = to_tris_temperature_convention(galactic, ring.effective_frequency_mhz)

guess, sigma = tris_prior_from_template(
    template, inputs.pixel_indices, relative_sigma=0.10, absolute_sigma_k=0.3
)
print("implied monopole prior:", inputs.implied_monopole_prior_sigma_k(sigma))

solution = inputs.solve(prior_map=guess, prior_sigma_k=sigma)
sky = solution.healpix_map()                 # full-sky RING, UNSEEN outside
print(solution.reduced_chi_square)           # read this before anything else
print(solution.zero_level_k, solution.zero_level_uncertainty_k)
```

**Read `reduced_chi_square` first.** Against real data with a rough template it
comes out in the tens, meaning the template — not the noise — dominates the
residual, and the fitted `zero_level_k` is then absorbing template mismatch
rather than measuring the archive's zero point. A trustworthy zero level needs
both a reduced chi-square near 1 *and* an implied monopole prior well below
`zero_level_sigma_k`.

To validate the chain before trusting it on real data, replace the samples with
a simulation of a known sky:

```python
truth = template[inputs.pixel_indices]
simulated = inputs.with_data(inputs.predict(truth, zero_level_k=0.03))
check = simulated.solve(prior_map=guess, prior_sigma_k=sigma)
```

### Low-dimensional profile fits

`fit_tris_linear_model` remains the right tool when you want profile
coefficients rather than a map. It whitens with `TRISNoiseModel` (exact for any
`common_mode_sigma_k`, where a dense factorization silently fails above about
`1e4`), gates the whitened design's SVD rank, and now reports `chi_square`,
`degrees_of_freedom` and `reduced_chi_square`. Watch that last one: an
`m_max=3` Fourier model on the real 600-MHz ring gives a reduced chi-square of
about **1.1e4**, i.e. formal errors too small by a factor of ~100. The `p < n`
gate is a backstop, not a statement that everything below it is well posed --
`m_max=59` passes it and merely restates the data.

## What this is not

Not a measured 2D beam, not a 3-deg-resolution map, not a 2.5-GHz ring, and not
a unique full-sky reconstruction. Nothing here infers unpublished polarization,
a ground/horizon *emission* model, the coordinate epoch, or a sample covariance.
The epoch is unstated by the archive; precession between B1950 and J2000 would
shift RA by ~0.6 deg at this declination, negligible against a 23-deg beam but
still an assumption you own. The 820-MHz asymmetric zero level is never
symmetrized for you.

See the [TRIS API reference](api/tris.md) for signatures.
