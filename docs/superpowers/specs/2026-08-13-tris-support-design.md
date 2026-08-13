# TRIS support design

## Goal and scientific boundary

Add a small, explicit TRIS compatibility layer to limTOD. It must read the
four public LAMBDA text products, translate their beam/pointing conventions
into limTOD conventions, and support only data-identifiable low-dimensional
inference. It must not present a regularized full-sky HEALPix solution as a
TRIS measurement.

The public archive contains two complete zenith drift rings (nominal 600 and
820 MHz), six 2.5-GHz samples, and two principal-plane beam cuts. It does not
contain raw TOD, a two-dimensional polarized beam, a bandpass, or a sample
covariance. A fixed-declination ring constrains one combination of sky
spherical-harmonic coefficients per temporal mode, so a general 2D map is
rank-deficient.

## Source ledger

Retrieved 2026-08-13 from the
[LAMBDA TRIS product page](https://lambda.gsfc.nasa.gov/product/tris/tris_prod_table.html).
The working copies live in ignored `downloads/TRIS/`; runtime code performs no
network access.

| Product | Public rows | SHA-256 |
|---|---:|---|
| `TRIS_absolute_600.txt` | 120 | `f0c9d99d6276e689b83af5cb4383d0b20cab2e901c3e3e91a6c1138433b84928` |
| `TRIS_absolute_820.txt` | 120 | `64d1af05d917015bb923c82b7786349d2b10c23278309b967043c419039499ae` |
| `TRIS_absolute_2500MHz.txt` | 6 | `9cc153ceabf87ed78a4f9522bbb1785449c989118cc35d059a0734da319ed42d` |
| `TRIS_Beam_Profile.txt` | 55 | `082fc49d5a8cba5b158bcdc301c1c3b2f1e5b668fd84088ef19c5231a907f190` |

## Convention decisions

| Topic | TRIS/public meaning | limTOD representation | Status |
|---|---|---|---|
| Ring coordinate | RA labels on nominal declination +42 deg | `LST_deg=RA`, parked zenith | documented approximation |
| Site latitude | 42 deg 26 arcmin N | explicit geometry parameter, default 42 + 26/60 deg | measured |
| Beam cuts | peak-relative E/H principal-plane dB | linear relative power `10**(dB/10)` | physical interpretation; archive header does not state power vs voltage |
| Beam image | only two 1D cuts are public | elliptical Gaussian, 18 deg E by 23 deg H FWHM | explicit approximation |
| Beam frame | E-plane/polarization axis 7 deg east of meridian | intrinsic E axis at beam `phi=0`; `selfrot=-7 deg` | convention conversion |
| Frequencies | files labelled 600/820/2500 MHz | retain nominal plus effective 600.5/817.8/2427.8 MHz | measured/reported |
| Bandwidths | 0.3/0.3/3 MHz | metadata only; no band integration | measured/reported |
| Temperature | absolute sky brightness in K | scalar K samples | archive does not specify K_RJ vs K_CMB |
| Per-row uncertainty | statistical at 600/820; no such column at 2.5 GHz | separate statistical array or `None` | confirmed |
| Zero level | one common uncertainty per product; 820 asymmetric | `AsymmetricUncertainty`, never copied into diagonal noise | confirmed |
| Polarization | single linear response; no public cross-pol/Jones/Mueller | scalar-I approximation only | unsupported beyond scalar response |
| HEALPix | not a TRIS product convention | RING output for limTOD beam callable | limTOD convention |

The implementation preserves raw RA labels, accepts legacy CR/LF line
endings, and never silently replaces the published coordinates by an exact
uniform grid. Zero statistical uncertainties require an explicit positive
floor before fitting.

## Public API

All new names live in `limTOD.tris`; the package root is unchanged.

- Data: `AsymmetricUncertainty`, `TRISRing`, `TRISPointSet`,
  `TRISPrincipalPlaneCuts`, `parse_tris_ra`, `read_tris_ring`,
  `read_tris_point_set`, `read_tris_beam_cuts`.
- Beam/geometry: `approximate_tris_gaussian_beam_map`, `tris_beam_func`,
  `TRISZenithGeometry`, `tris_zenith_geometry`.
- Identifiable models: `build_tris_fourier_design`, `TRISRankDiagnostic`,
  `TRISLinearFit`, `fit_tris_linear_model`.

`fit_tris_linear_model` whitens a caller-supplied Fourier/template design,
checks its numerical rank with SVD before solving, and fails on deficient
models. It uses per-row statistical errors and can add a caller-selected
*symmetric* common-mode covariance. Because the 820-MHz zero level is
asymmetric, the API never converts it automatically; the caller must choose
and document an approximation or use an external asymmetric likelihood.

## Acceptance criteria

- Strict, offline readers preserve units, labels, uncertainty roles, and
  official row counts when run on the downloaded products.
- Beam axes, FWHM, normalization, frequency independence, and roll sign are
  tested against the documented conventions.
- Constant-sky forward response is invariant under beam scale when normalized.
- Duplicate/over-complete model columns and zero statistical errors without an
  explicit floor fail loudly; a reduced full-rank Fourier model recovers known
  coefficients.
- `docs/tris.md` is the compact user-facing convention report and includes a
  runnable offline example and explicit unsupported claims.
