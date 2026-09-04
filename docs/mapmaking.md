# Map-making

limTOD ships two map-makers that share the same operator construction
(scan geometry → truncated pixel set → per-TOD system operators) and
differ only in how they treat the noise:

| Class | Noise treatment | Use when |
|---|---|---|
| `HPW_mapmaking` | Optional Butterworth **h**igh-**p**ass filter + diagonally-weighted **W**iener solve | 1/f drifts are removed by filtering; fast, robust default |
| `GLS_mapmaking` | Full 1/f + white time-time covariance, iteratively-reweighted GLS ([hydra-tod](https://github.com/hydra-cosmology/hydra-tod) port) | You know (or assume) the noise parameters and want the statistically optimal weights |

The HPW estimator is effectively ordinary least squares on filtered
data — unbiased for the retained modes but not minimum-variance under
red noise. The GLS estimator whitens the 1/f noise exactly instead of
cutting it out.

For a complete worked example see
[mm_example.ipynb](https://github.com/zzhang0123/limTOD/blob/main/examples/mm_example.ipynb):
simulating several TOD sets at different elevations, building the
map-maker, solving, and visualizing the reconstruction.

## `HPW_mapmaking`: high-pass + Wiener

## Estimator

```
x̂ = (Aᵀ N⁻¹ A + S⁻¹)⁻¹ (Aᵀ N⁻¹ d + S⁻¹ μ)
```

| Symbol | Meaning |
|---|---|
| `A` | System operator: pointed-beam rows over the selected sky pixels (built from `generate_sky2sys_projection`), optionally composed with other Tsys operators |
| `N` | Noise covariance (after high-pass filtering, if enabled) |
| `S`, `μ` | Prior covariance and mean for the sky (and optionally for other components) |
| `d` | Measured TOD |

Pixel selection: only pixels where the stacked |beam| response exceeds
`threshold × peak` enter the solve (`truncate_stacked_beam`), keeping the
dense linear algebra tractable.

## Construction

```python
from limTOD import HPW_mapmaking

mapmaker = HPW_mapmaking(
    beam_map=beam_map,                      # (npix,), (3,npix) or (4,npix)
    LST_deg_list_group=LST_group,           # one LST array per TOD set
    lat_deg=-30.7130,
    azimuth_deg_list_group=az_group,
    elevation_deg_list_group=el_group,
    selfrot_deg_list_group=None,            # default: zero self-rotation
    threshold=0.01,                         # pixel-selection beam threshold
    Tsys_others_operator_group=None,        # optional extra Tsys operators
    nside_hires=None,                       # upgrade beam before processing
    nside_target=None,                      # output map resolution
    beam_truncate_frac_thres=None,          # defaults to `threshold`
)
```

All arguments are keyword-only. Groups are lists with one entry per TOD
set (e.g. per scan/elevation).

## Solving

```python
sky_map, sky_uncertainty = mapmaker(
    TOD_group=TOD_group,                    # list of TOD arrays
    dtime=2.0,                              # sampling interval [s]
    cutoff_freq_group=cutoff_freqs,         # high-pass cutoffs [Hz] (or None)
    gain_group=None,                        # per-TOD gain calibration
    known_injection_group=None,             # known signals to subtract
    Tsky_prior_mean=None,
    Tsky_prior_inv_cov_diag=None,
    Tsys_other_prior_mean_group=None,
    Tsys_other_prior_inv_cov_group=None,
    return_full_cov=False,
    filter_order=4,                         # Butterworth order
    use_high_pass=False,                    # False: solve on unfiltered TOD
)
```

Returns the reconstructed `sky_map` and per-pixel `sky_uncertainty`; when
other-Tsys operators are supplied, their estimates and uncertainties are
returned as additional groups. With `return_full_cov=True` the full
posterior covariance replaces the diagonal uncertainty.

Notes:

- **The Gaussian prior is the only regularisation.** No ridge is added to
  the normal equations. A `+ lambda*I` term is exactly a zero-mean
  Gaussian prior of precision `lambda` on top of the one you declared —
  and, with no matching `lambda*mu` on the right-hand side, one that
  contradicts `Tsky_prior_mean`. Its whole effect falls on the directions
  the scan does not measure, where it answers "zero" and reports
  `1/sqrt(lambda)` as the uncertainty on that answer. Without it, an
  under-determined pixel set raises `LinAlgError` instead: the fix is an
  informative `Tsky_prior_inv_cov_diag`, a coarser `nside_target`, or a
  higher `threshold`.
- With `use_high_pass=False`, `cutoff_freq_group` may be `None` — the
  solve then uses the unfiltered TOD and operator.
- The high-pass filter is applied consistently to both the data and the
  operator (`filtfilt`, represented as a dense matrix), so the estimator
  stays unbiased for the retained modes.
- Cost is dominated by dense `(n_selected_pixels)²` linear algebra —
  control it with `threshold` and `nside_target`.

## `GLS_mapmaking`: full noise covariance (hydra-tod port)

`GLS_mapmaking` takes **the same constructor arguments** as
`HPW_mapmaking` (the geometry/operator build is shared) and replaces the
solve. It implements the iteratively-reweighted GLS of
[Zhang et al. (2026), RASTI, rzag024, §3.2](https://doi.org/10.1093/rasti/rzag024)
for the multiplicative noise model that `TODSim.generate_TOD` simulates:

$$
d = g_\mathrm{bg}\,(1 + n_g)\,(U p + \mu)\,(1 + n_w)
\;\approx\; g_\mathrm{bg}\,(U p + \mu)\,(1 + n),
$$

where $n_g$ is 1/f gain noise and $n_w$ fractional white noise. The
fractional-noise covariance

$$
N = \mathrm{toeplitz}\!\big(\mathrm{flicker\_corr}(\tau; f_0, f_c, \alpha)\big)
    + \sigma_w^2 I
$$

is **exactly** the covariance `limTOD.flicker_model.sim_noise` draws the
simulated noise from, so on limTOD-simulated TOD the GLS weights are
exact. Because the noise is multiplicative, the data covariance
$\Sigma = \mathrm{diag}(Up+\mu)\, N\, \mathrm{diag}(Up+\mu)$ depends on
the signal, and the solve is iterated (IRLS) until the parameters
converge.

```python
from limTOD import GLS_mapmaking

mapmaker = GLS_mapmaking(**same_geometry_kwargs_as_HPW)

sky_map, sky_uncertainty = mapmaker(
    TOD_group=TOD_group,
    dtime=2.0,                        # or time_list_group=[t1, t2, ...]
    gain_noise_params=(1.335e-5, 1.099e-3, 2),   # (f0, fc, alpha), angular
    white_noise_var=2.5e-6,           # generate_TOD defaults
    known_injection_group=None,       # e.g. noise-diode temperature
    noise_model="multiplicative",     # or "additive"
)
```

Notes:

- `known_injection_group` stays **inside** the model, $(Up+\mu)(1+n)$ —
  it is *not* subtracted from the data, which would mis-weight the noise.
  (`HPW_mapmaking` subtracts it, consistent with its additive model.)
- For externally calibrated data with additive noise
  ($d = Up + \mu + \varepsilon$), pass `noise_model="additive"` — a
  single non-iterative GLS solve. With a white covariance this
  reproduces `HPW_mapmaking`'s unfiltered solution exactly.
- Explicit per-TOD inverse covariances can be supplied via
  `noise_inv_cov_group` (overriding the flicker/white parameters).
- Priors and `return_full_cov` work as in `HPW_mapmaking` (including
  the no-ridge rule above); IRLS convergence is controlled by
  `tol`/`min_iter`/`max_iter`.
- Cost adds one dense `(n_time)²` covariance inversion per TOD (once)
  and one reweighted normal-equations solve per iteration (typically
  converges in ≲10 iterations).
- The lower-level pieces are exported too: `flicker_noise_cov` /
  `flicker_noise_inv_cov` (the Toeplitz builder) and `iterative_gls`
  (the faithful single-TOD hydra-tod solver).

### What a wrong noise covariance costs

The GLS assumes $N$ is **known**. It estimates no noise parameters of its
own: `gain_noise_params` defaults to the simulator's own truth
(`DEFAULT_GAIN_NOISE_PARAMS`), which on simulated TOD makes the weights
exact and on real data makes them an *assertion*, not a measurement. It is
worth being precise about what that assertion buys and what it costs.

Write the assumed weight $W$ and $M = (U^\mathsf{T} W U)^{-1}$, so the
(additive-model) estimate is $\hat p = M U^\mathsf{T} W d$.

- **The map stays unbiased.** $\hat p$ is linear in the data and
  $\mathbb{E}[d] = Up$, so $\mathbb{E}[\hat p] = M U^\mathsf{T} W U p = p$
  for *every* invertible $W$ — exactly, not asymptotically. A wrong $N$
  costs **efficiency**, not accuracy: in a Monte Carlo over the drift-scan
  geometry of `tests/test_gls_mapmaking.py`, $\|U(\hat p - p)\|$ went
  0.131 (matched) → 0.182 (1/f ignored, white weights) → 0.196 (knee 10x
  too high) → 0.250 (knee 10x too low).

- **The error bars do not survive.** The map-maker reports
  $\sqrt{\mathrm{diag}\,M}$, but the estimator's actual covariance is the
  sandwich

  $$
  \mathrm{Cov}[\hat p] = M\,U^\mathsf{T} W N W U\,M ,
  $$

  which collapses to $M$ only at $W = N^{-1}$. Median over pixels of
  reported $\sigma$ / true scatter, same Monte Carlo:

  | assumed $N$ | reported $\sigma$ / true scatter | verdict |
  |---|---|---|
  | true $N$ (matched) | 1.00 | honest |
  | white, 1/f ignored | 1.34 | 1.3x underconfident |
  | knee 10x too high | 0.96 | ~honest, but 1.5x noisier |
  | knee 10x too low | **0.43** | **2.3x overconfident** |
  | $\alpha$: 2 → 1.5 | 3.57 | 3.6x underconfident |

  Underestimating the knee is the dangerous direction: the map claims a
  precision it does not have.

Both statements above are exact only for the **additive** model, where
$\hat p$ is linear in the data. On the default multiplicative path the
IRLS weights depend on $U\hat p$, so the estimator is nonlinear and
inherits neither result for free: unbiasedness survives only to the order
the data model itself is derived to — the same $O(n^2)$ at which
`generate_TOD`'s $n_g n_w$ cross term is dropped, ~1e-4 at the default
noise levels. Measured, both paths agree to two decimals on every ratio
above, so the reweighting neither causes the breakage nor cures it. All
of this is pinned by `tests/test_gls_noise_misspecification.py`.

So the reported uncertainties are only as good as the noise model handed
in, and misplaced confidence in $N$ shows up as misplaced confidence in
the map — never as a shifted map. If the 1/f parameters are not known,
they should be **sampled jointly with the sky** rather than fixed at a
default; that is what the sibling
[hydra-tod](https://github.com/hydra-cosmology/hydra-tod) package does,
Gibbs-sampling gains, noise parameters and map together. `GLS_mapmaking`
is the single-shot solve you reach for once you are willing to condition
on a particular $N$.

## JAX alternative

For differentiable map-making inside a JAX pipeline, the same
normal-equations solve runs matrix-free: conjugate gradients on top of the
[limtod_jax](limtod-jax.md) projector and its exact adjoint, with the
operator never formed. limTOD ships the projector and the adjoint; the CG
solver belongs to the pipeline that consumes them
([Downstream](index.md#downstream)).
