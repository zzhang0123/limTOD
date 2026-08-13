# TRIS support: data, conventions, and safe inference

`limTOD.tris` is an offline bridge for the public TRIS archive. Its public products are **final beam-convolved profiles or points**, not raw TOD. The free two-dimensional map-makers in limTOD cannot identify a sky map from one fixed-declination ring: the supported scientific result is a native profile, a reduced Fourier representation, or amplitudes of externally supplied templates.

## Archive, provenance, and scope

The source is the [LAMBDA TRIS product page](https://lambda.gsfc.nasa.gov/product/tris/tris_prod_table.html); the [LAMBDA publication list](https://lambda.gsfc.nasa.gov/papers/index.html) and [TRIS I](https://arxiv.org/abs/0806.1415) are the associated references. The files below were retrieved on 2026-08-13 into ignored `downloads/TRIS/`. Readers take explicit local paths and never fetch data.

| Product | Rows | Schema | SHA-256 |
|---|---:|---|---|
| `TRIS_absolute_600.txt` | 120 | RA label, K, per-row statistical K | `f0c9d99d6276e689b83af5cb4383d0b20cab2e901c3e3e91a6c1138433b84928` |
| `TRIS_absolute_820.txt` | 120 | RA label, K, per-row statistical K | `64d1af05d917015bb923c82b7786349d2b10c23278309b967043c419039499ae` |
| `TRIS_absolute_2500MHz.txt` | 6 | RA label, K, common zero-level K | `9cc153ceabf87ed78a4f9522bbb1785449c989118cc35d059a0734da319ed42d` |
| `TRIS_Beam_Profile.txt` | 55 | angle deg, H-cut dB, E-cut dB | `082fc49d5a8cba5b158bcdc301c1c3b2f1e5b668fd84088ef19c5231a907f190` |

The nominal/effective frequencies and bandwidths are respectively 600/600.5 MHz/0.3 MHz, 820/817.8 MHz/0.3 MHz, and 2500/2427.8 MHz/3 MHz. The frequency metadata is retained; no band integration is implied.

| Topic | Status and use in this bridge |
|---|---|
| Archive and paper | Confirmed: two 120-point rings, six 2.5-GHz points, two common beam cuts, the site and the quantities above. |
| limTOD conversion | Physical interpretation: archive RA labels are used verbatim as LST samples for a parked zenith scan; results are beam-convolved scalar-temperature predictions. |
| Two-dimensional beam | Explicit Gaussian approximation: 18° E × 23° H FWHM main lobe, not a reconstructed measurement. |
| Not supplied | Raw TOD, sample covariance, bandpass integration, coordinate epoch, K_RJ versus K_CMB convention, a measured 2D/polarized beam, and a unique full-sky map. |

## Data and uncertainty conventions

The archive labels the rings as declination +42°, while TRIS I gives Campo Imperatore as 42°26′ N, 13°33′ E, 2000 m. The reader preserves the irregular raw RA text and coordinates—do not replace them with a uniform grid. Its `3°` sampling statement is a sample spacing, **not angular resolution**.

At 600 and 820 MHz, the third column is a per-row statistical uncertainty; it is distinct from one product-wide zero level: ±0.066 K at 600 MHz and −0.300/+0.430 K at 820 MHz. At 2.5 GHz the repeated 0.284 K value is one common zero-level uncertainty, not six statistical errors. A published zero statistical entry requires the caller to set an explicit positive floor before fitting. Archive headers do not state K_RJ versus K_CMB or the coordinate epoch, so this layer calls them scalar K and does not invent either convention.

## Beam and pointing translation

The public beam product contains only common one-dimensional E- and H-plane cuts. Here H/E mean spatial principal planes; they are not HH/VV, Stokes, or polarization labels. The property conversion `10**(dB/10)` is an explicit **physical interpretation** of the dB values as relative power—the header does not say power rather than voltage. The 18° E × 23° H elliptical Gaussian made by `approximate_tris_gaussian_beam_map` is a named 2D main-lobe approximation. It is scalar-intensity only: no cross-polar response, Jones/Mueller model, full Stokes treatment, backlobe, or horizon/ground model is available.

| Physical quantity | limTOD representation |
|---|---|
| Frequency | MHz, including the nominal and effective values above. |
| Sky and beam maps | Scalar HEALPix maps in RING ordering. |
| Boresight | Beam-map north pole. |
| Intrinsic axes | E at phi = 0°/180°; H at phi = 90°/270°. |
| Scan samples | Each supplied RA label becomes the corresponding `LST_deg`; no resampling. |
| Park | Zenith: `azimuth_deg=0`, `elevation_deg=90`. |
| Latitude | Physical default 42°26′ N; request `latitude_deg=42` explicitly to use the rounded declination label. |
| Roll | `selfrot=-7°`: intrinsic north/E axis is rolled so north maps to NE azimuth 7°, south to SW azimuth 187°. |

`tris_zenith_geometry` owns the pointing orientation; do not apply a second beam rotation. Beam-map `normalization="peak"` preserves a unit peak, `"sum"` divides by the discrete HEALPix sum, and `"none"` leaves the Gaussian scale unchanged. For brightness-temperature convolution, keep a physical horizon/ground treatment explicit and normally call `generate_TOD_sky` with `normalize_beam=True`; this makes the beam-weighted result independent of an arbitrary scalar beam scale.

## What can be inferred

For a fixed-declination ring, each temporal Fourier mode constrains only

$$
V_m = \sum_l \operatorname{conj}(B_{lm}) S_{lm}.
$$

Thus many sky harmonic coefficients project to the same datum. Regularization can choose one map but cannot create the missing information. Before `fit_tris_linear_model` solves a caller-supplied reduced model, it requires both `p < n` (parameters versus samples) and full numerical rank of the Cholesky-whitened design from an SVD. It uses the per-row statistical variance and may add the caller's chosen symmetric common covariance `sigma_common**2 * 11^T`. That is appropriate for an explicitly modelled symmetric 600-MHz common zero level; the asymmetric 820-MHz value is never auto-symmetrized. Choose and record an approximation yourself, or use an asymmetric likelihood outside this API.

## Offline workflow

The following performs no download. It uses a local scalar HEALPix RING sky map and deliberately fits only a small Fourier model. The floor policy here is explicit and reproducible: use the smallest positive published statistical error only to replace zero entries; a scientific analysis should justify a different floor if needed.

```python
from pathlib import Path

import numpy as np
import healpy as hp
from limTOD import generate_TOD_sky
from limTOD.tris import (
    approximate_tris_gaussian_beam_map,
    build_tris_fourier_design,
    fit_tris_linear_model,
    read_tris_beam_cuts,
    read_tris_ring,
    tris_zenith_geometry,
)

archive = Path("downloads/TRIS")
ring = read_tris_ring(archive / "TRIS_absolute_600.txt")
cuts = read_tris_beam_cuts(archive / "TRIS_Beam_Profile.txt")
relative_e_power = cuts.e_plane_relative_power  # inspect the stated dB→power interpretation

sky_map = np.load("local_scalar_sky_ring.npy")  # local HEALPix RING scalar K map
nside = hp.get_nside(sky_map)
beam = approximate_tris_gaussian_beam_map(nside=nside, normalization="peak")
geometry = tris_zenith_geometry(ring.ra_deg)     # physical 42°26′ N default, roll -7°
model_profile_k = generate_TOD_sky(
    beam, sky_map, geometry.lst_deg, geometry.latitude_deg,
    geometry.azimuth_deg, geometry.elevation_deg, geometry.selfrot_deg,
    normalize_beam=True,
)

design = build_tris_fourier_design(ring.ra_deg, m_max=3)
positive_stat = ring.statistical_uncertainty_k[ring.statistical_uncertainty_k > 0]
floor_k = float(np.min(positive_stat))  # declared zero-entry floor heuristic
fit = fit_tris_linear_model(
    ring, design, uncertainty_floor_k=floor_k,
    common_mode_sigma_k=float(ring.zero_level_uncertainty_k),  # 600 MHz only
)
```

The forward profile can be compared with the native data or used to construct external-template amplitudes. `model_profile_k` is not a new measured map.

## No-go claims

This support must not be described as a measured 2D beam, a 3°-resolution map, a 2.5-GHz ring, or a unique full-sky reconstruction. It does not infer unpublished polarization, a ground/horizon response, coordinate epoch, temperature convention, or an unprovided covariance. See the [TRIS API reference](api/tris.md) for signatures; this page is the canonical scientific and convention report.
