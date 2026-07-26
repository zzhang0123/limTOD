# Drift scans in harmonic space — the m-mode path

`limtod_jax.driftscan` is the drift-scan special case of the sky→TOD
simulator: fixed azimuth/elevation/self-rotation at a fixed site, only the
LST advancing. Where the generic path
({func}`limtod_jax.core.generate_tod_sky`) performs one Wigner rotation of
the beam **per time sample** — $O(n_\mathrm{time}\,\ell_\mathrm{max}^3)$ —
the drift-scan geometry collapses the whole scan onto **one** rotation
plus per-$m$ phases. It is pure JAX + [equinox](https://docs.kidger.site/equinox/)
(jit/vmap/grad-safe, an `equinox.Module` operator), designed to drop
straight into
[replicant-telescope](https://github.com/zzhang0123/replicant-telescope)
pipelines.

## Formalism

The implementation follows the m-mode formalism of Z. Zhang, *M-mode RIME
explicit in beam, fringe and sky modes* (Jan 2024,
[MmodeNote.pdf](https://zh-zhang.com/myNotes/MmodeNote.pdf)), in the "MT
interpretation" with the fringe term $\equiv 1$: for a single-dish
autocorrelation the modulated beam **is** the primary beam.

Earth rotation is a rotation about the celestial pole, so the
celestial-frame beam coefficients at any LST differ from those at a
reference LST only by a phase:

$$
B_{\ell m}(\mathrm{lst}) \;=\; B_{\ell m}(\mathrm{lst}_\mathrm{ref})\,
e^{-i m \Delta},
\qquad \Delta = \mathrm{lst} - \mathrm{lst}_\mathrm{ref}\ \text{[rad]} .
$$

The TOD is then a Fourier series in $\Delta$ (eqns 13–15 of the note):

$$
V(\Delta) = \tilde V_0 + 2\,\mathrm{Re}\sum_{m\ge 1} \tilde V_m\,
e^{+i m \Delta},
\qquad
\tilde V_m = \sum_{\ell} \overline{B_{\ell m}(\mathrm{lst}_\mathrm{ref})}\,
\tilde S_{\ell m} .
$$

The $\tilde V_m$ are the **m-modes** — the Fourier coefficients of the
sidereal-day-periodic TOD — and the sky→m-mode projection is a single
per-$(\ell,m)$ product with the reference-frame beam. Both phase signs
above are **locked numerically** against the generic path (never trusted
on paper — see `tests/limtod_jax/test_driftscan.py`); the two paths agree
to ~$10^{-15}$ relative in float64, including the `normalize` branch and
the exact adjoint.

Cost: one $O(\ell_\mathrm{max}^3)$ Wigner rotation total, then an
$O(n_\mathrm{time}\,\ell_\mathrm{max})$ phase synthesis with
$O(n_\mathrm{time}+\ell_\mathrm{max})$ memory.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)   # before any jax array

import numpy as np, healpy as hp, jax.numpy as jnp
import limtod_jax as ltj

nside, lmax = 128, 256
beam_alm = jnp.asarray(hp.map2alm(beam_map, lmax=lmax))          # beam-local
sky_alm  = ltj.map2alm_quad(jnp.asarray(sky_map), nside=nside, lmax=lmax)

op = ltj.DriftScanMmode.from_pointing(
    beam_alm,
    lst_deg=np.linspace(0.0, 360.0, 3600, endpoint=False),
    lat_deg=-30.7, az_deg=41.5, el_deg=52.5,
    lmax=lmax,
)
tod    = op(sky_alm)          # (n_time,) — == generate_tod_sky to roundoff
vm     = op.mmodes(sky_alm)   # (lmax+1,) complex m-modes Ṽ_m
sky_at = op.adjoint(tod)      # exact transpose (map-making normal equations)
```

`DriftScanMmode` is an `equinox.Module` (a frozen pytree): jit it with
`eqx.filter_jit`, batch skies with `jax.vmap(op)`, embed it as a field of
larger models, and differentiate through construction (the reference
rotation is traced, so gradients w.r.t. pointing angles and beam alms
work). The functional layer
({func}`~limtod_jax.driftscan.mmodes_from_sky`,
{func}`~limtod_jax.driftscan.tod_from_mmodes`,
{func}`~limtod_jax.driftscan.driftscan_tod`,
{func}`~limtod_jax.driftscan.driftscan_tod_adjoint`,
{func}`~limtod_jax.driftscan.mmodes_from_tod`) is available when you
don't want the operator object.

Conventions are unchanged from the rest of `limtod_jax`: degrees at the
public pointing boundary, packed healpy alms (m ≥ 0, real fields),
quadrature sky alms for the exactness contract, `lmax`/`nside`/`normalize`
static. Enable x64.

## The horizon mask

Physically, the beam of a ground-based drift scan is the free-space beam
**cut at the horizon** — the antenna cannot see through the ground. The
m-mode formalism handles this naturally: the masked beam is still static
in the antenna frame, so it just replaces the beam alms.
{func}`~limtod_jax.driftscan.horizon_masked_beam_alm` builds it inside
JAX: rotate the beam-local alms into the horizontal frame (where the
horizon is the pure-colatitude circle $\theta = 90°$), multiply by an
elevation taper, re-analyze (healpy-equivalent iterative `map2alm`,
{func}`limtod_jax.hpx.map2alm_iter`), rotate back. The result lives in
the **beam-local frame**, so it drops into either the drift-scan path or
the generic path unchanged:

```python
masked = ltj.horizon_masked_beam_alm(
    beam_alm, az_deg=41.5, el_deg=10.0,
    nside=nside, lmax=lmax, apod_deg=5.0,
)
op = ltj.DriftScanMmode.from_pointing(masked, ...)
# or in one step:
op = ltj.DriftScanMmode.from_pointing(
    beam_alm, ..., nside=nside, horizon_mask=True, apod_deg=5.0,
)
```

Masking is **off by default** so the module reproduces numpy limTOD
(which does not mask) exactly.

### How bad is the ringing? (and when to care)

A sharp cut is not band-limited, so representing the masked beam at
finite $\ell_\mathrm{max}$ produces Gibbs ringing. The study script
[`docs/driftscan_ringing_study.py`](https://github.com/zzhang0123/limTOD/blob/main/docs/driftscan_ringing_study.py)
quantifies the effect end-to-end (reference: pixel-space TOD with the
sharply masked beam at $N_\mathrm{side}=256$; sky band-limited at
$\ell=150$ so the sky side is exact). Headline numbers (relative RMS TOD
error from the harmonic representation of the masked beam):

| scenario | mask matters? | hard cut | apod 2° | apod 5° |
|---|---|---|---|---|
| narrow beam (2° FWHM), any el ≥ 10° | no (~10⁻⁶) | — | — | — |
| wide beam (25° FWHM), el = 41° | ~10⁻⁴ | 6·10⁻⁶ @ ℓ≤192 | 2·10⁻⁶ | 4·10⁻⁷ |
| wide beam (25° FWHM), el = 10° | **yes (~30%)** | **6·10⁻³ @ ℓ≤192, plateaus ~4·10⁻³ @ ℓ≤384** | 1.8·10⁻³ → 3.6·10⁻⁴ @ ℓ≤384 | 1.5·10⁻⁴ → 3.6·10⁻⁵ @ ℓ≤384 |

Reading of the table:

- **Narrow beams don't care.** A few-degree beam at el ≥ 10° has
  ~$10^{-6}$ of its TOD response below the horizon; skip the mask (or
  apply it — it changes nothing).
- **Wide beams at low elevation must mask** (the cut carries ~30% of the
  TOD), and there a **hard cut leaves a ~0.5% TOD error that does not
  converge away with $\ell_\mathrm{max}$** — the discontinuity aliases
  into the analysis step, so raising the band-limit alone stalls.
- **Apodization is the effective mitigation**: a 2–5° cosine taper
  (`apod_deg`) drops the error by 1–2 orders of magnitude at fixed
  $\ell_\mathrm{max}$, at the price of slightly reshaping the beam within
  the taper band (a *physical* modelling choice — real horizons aren't
  knife edges either).
- **Don't over-apodize**: keep `apod_deg` well below
  (elevation − beam radius), or the taper starts eating the main beam
  (visible in the study as a jump in the "mask matters" column for the
  narrow beam at el = 10°, apod 10°).

A compact regression version of this study is pinned in the test suite
(`test_ringing_apodization_mitigates`).

## When to use which path

| | generic `generate_tod_sky` | `DriftScanMmode` |
|---|---|---|
| pointing | arbitrary per-sample | fixed az/el/selfrot (drift) |
| cost | $O(n_t\,\ell_\mathrm{max}^3)$ | $O(\ell_\mathrm{max}^3 + n_t\,\ell_\mathrm{max})$ |
| agreement | — | equal to roundoff on drift scans |
| m-modes | not exposed | {func}`~limtod_jax.driftscan.mmodes_from_sky` / `op.mmodes` |

Anything that is a genuine drift scan should use this path; tracking or
scanning strategies need the generic one.
