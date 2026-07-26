# limtod_jax — the differentiable JAX port

`limtod_jax` reimplements limTOD's sky→TOD chain
(pointing → ZYZ Euler angles → Wigner rotation of beam alms →
beam-weighted sum) in pure JAX: **jit/vmap/grad-safe and differentiable
with respect to both the sky and the beam**. It was built to the port
contract of the
[replicant-telescope](https://github.com/zzhang0123/replicant-telescope)
digital-twin project (formerly e-RHINO; Python package `replicant`), whose
`NativeLimTODProjector` and CG map-making run on top of it.

```bash
pip install "limTOD[jax]"     # requires Python >= 3.11 (s2fft floor)
```

Design rules: `lmax`/`nside` are static Python ints; angles, alms, and
TODs are traced arrays; no value-dependent Python control flow; ambient
dtype is respected (nothing hardcodes float32/float64); healpy never
enters the JAX path (it remains the test-suite oracle).

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)   # see "Precision" below

import healpy as hp
import jax.numpy as jnp
import numpy as np
import limtod_jax as ltj

nside = 32
lmax = 3 * nside - 1
npix = 12 * nside**2

# Beam alms exactly as numpy limTOD computes them internally:
beam_alm = jnp.asarray(hp.map2alm(beam_map, lmax=lmax))

# Sky enters via QUADRATURE alms (see "Exactness contract"):
sky_alm = ltj.map2alm_quad(jnp.asarray(sky_map), nside=nside, lmax=lmax)

# Pointings -> (n_time, 3) ZYZ angles [radians]:
psi, theta, phi = ltj.zyz_of_pointing(lst_deg, lat_deg, az_deg, el_deg, selfrot_deg)
angles = jnp.stack([psi, theta, phi], axis=-1)

tod = ltj.generate_tod_sky(beam_alm, sky_alm, angles, lmax=lmax)   # (n_time,)
```

Everything composes with JAX transformations:

```python
fast   = jax.jit(ltj.generate_tod_sky, static_argnames=("lmax", "normalize"))
multi  = jax.vmap(lambda b, s: ltj.generate_tod_sky(b, s, angles, lmax=lmax))
d_beam = jax.grad(lambda b: ltj.generate_tod_sky(b, sky_alm, angles, lmax=lmax).sum().real)
```

## Public API

| Function | Purpose |
|---|---|
| `zyz_of_pointing(lst, lat, az, el, selfrot)` | Pointing (degrees) → `(ψ, θ, φ)` radians; batched inputs broadcast |
| `zyzyz2zyz(α, β, γ, δ, χ)` | Generic five-angle → ZYZ collapse |
| `rotate_alm(alm, ψ, θ, φ, *, lmax)` | Wigner rotation of packed alms — reproduces `_rotate_healpix_map` exactly (including limTOD's `hp.rotate_alm(alm, φ, θ, ψ)` argument order, locked numerically) |
| `beam_weighted_sum(beam_alm, sky_alm, *, normalize=False, ones_alm=None)` | One harmonic-space TOD sample |
| `generate_tod_sky(beam_alm, sky_alm, zyz_angles, *, lmax, normalize=False, ones_alm=None)` | Full TOD chain; `lax.map` over pointings, vmappable over a leading frequency axis |
| `generate_tod_sky_adjoint(tod, beam_alm, zyz_angles, *, lmax, ...)` | **Exact transpose** of the forward map (accumulated rotated beams) — what map-making normal equations need |
| `generate_projection_rows(beam_alm, zyz_angles, pixel_indices, *, lmax, nside, ...)` | Native `generate_sky2sys_projection`: pointed-beam rows at selected pixels |
| `alm2map(alm, *, nside, lmax)` / `map2alm_quad(m, *, nside, lmax)` | HEALPix synthesis / quadrature analysis inside JAX (s2fft) |
| `ones_quadrature_alm(*, nside, lmax)` | The exact pixel-sum functional — the `normalize` denominator |
| `packed_to_2d` / `packed_from_2d` / `alm_dot` / `nalm_of_lmax` / `lmax_of_nalm` | healpy packed-alm layout utilities and the weighted real-field inner product |

Packed alms follow healpy's real-field layout (m ≥ 0 only, index
`m(2·lmax+1−m)/2 + l`); m = 0 coefficients must be real, as `hp.map2alm`
produces them.

## Exactness contract

The port matches `limTOD.simulator.generate_TOD_sky(...,
truncate_frac_thres=0.0)` to **~1e-12 relative in float64** — not "close",
but exactly up to roundoff, because of one identity: the rotated beam is
strictly bandlimited, so limTOD's pixel-space sample
`Σ_p B_rot(p)·s(p)` equals the weighted harmonic dot `⟨R b, s̃⟩` **exactly**
when `s̃` holds *quadrature* alms

```
s̃_lm = Σ_p s(p)·Y*_lm(p) = (npix/4π) · hp.map2alm(s, lmax, iter=0)
     = limtod_jax.map2alm_quad(s)
```

Feed `generate_tod_sky` beam alms computed the way numpy limTOD does
(`hp.map2alm(beam_map, lmax=lmax)`, its default `iter=3`) and quadrature
sky alms, and the two implementations agree to float64 roundoff at every
pointing — including zenith and pole gimbal corners (oracle-tested at
nside 8–16 over an extreme-pointing grid).

`normalize=True` reproduces `normalize_beam` semantics exactly the same
way: the denominator `Σ_p B_rot(p)` is `⟨R b, ones_quadrature_alm⟩`.

**Out of scope** (deliberately): `truncate_frac_thres` (a *nonlinear*
cleanup of synthesis ringing — the port is the linear chain; compare
against the oracle with `truncate_frac_thres=0.0`), horizontal masks,
full-Stokes beams, noise generation, and map-making (replicant-telescope's
`SkySpaceFilter` covers that in JAX).

## Precision: enable x64

Two different numerical regimes coexist:

- The **Wigner rotation core** (Risbo recursion via s2fft) is stable even
  in float32 (~1e-7 relative at lmax ≲ 50).
- The **HEALPix map↔alm transforms** (`alm2map`/`map2alm_quad`, s2fft's
  Price–McEwen recursion) are **float64-only in practice**: in a float32
  session they carry O(10%) errors even at lmax ≈ 12.

So for any quantitative work:

```python
jax.config.update("jax_enable_x64", True)
```

## Drift scans: skip the per-sample rotation

For a genuine drift scan (fixed az/el/selfrot, only LST advancing) the
per-sample Wigner rotation is unnecessary: the m-mode path
([drift scans in harmonic space](driftscan.md)) reproduces
`generate_tod_sky` to roundoff with **one** rotation total plus per-m
phases — $O(\ell_\mathrm{max}^3 + n_t\,\ell_\mathrm{max})$ instead of
$O(n_t\,\ell_\mathrm{max}^3)$ — and exposes the m-modes and a horizon
mask along the way.

## Performance notes

- `generate_tod_sky` iterates pointings sequentially (`lax.map`): the
  per-pointing Wigner-d plane is `(L, 2L−1, 2L−1)` (~3.5 MB at L = 48,
  float64), so batching the time axis would multiply that by `n_time`.
  Batch the **frequency** axis instead (`jax.vmap` over leading alm axes).
- `rotate_alm` is vmappable over angle batches when you do want parallel
  rotations and can afford the memory.
- Compile time grows with `lmax` (the Wigner recursion unrolls over ℓ);
  at production lmax consider precomputing the Wigner-d plane where
  pointings share β. For a drift scan use
  {func}`limtod_jax.driftscan.dl_plane_for_pointing` (takes `lmax`, like
  every other public entry point) and pass the result as `dl_array`; one
  plane serves every LST bit-for-bit. The lower-level
  `limtod_jax.wigner.generate_rotate_dls` takes **`L = lmax + 1`** — mind
  the off-by-one, a plane built at the wrong band-limit is rejected by a
  shape check rather than silently rotating by the wrong sub-block.

## Conventions locked against healpy

Euler conventions are never trusted on paper: the mapping from limTOD's
`(ψ, θ, φ)` to the Wigner-D application was locked by numerically testing
8 candidate mappings against `hp.rotate_alm` on random alms (winner at
1.8e-15 relative; all others O(1)), and the test suite permanently pins
both the winner *and* the losers
(`tests/limtod_jax/test_rotation_convention.py`), so a silent convention
change in a future s2fft release cannot slip through.
