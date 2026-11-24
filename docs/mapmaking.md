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
- **The weighting is diagonal.** `noise_variance` is a scalar or one
  variance per sample; a full time-time covariance is rejected with a
  message rather than silently mis-broadcast. Left as `None` it is
  estimated from the residuals with a 100-sample rolling window — which
  is biased low when the operator does not span the projectable signal,
  so prefer an explicit value. For correlated noise use `GLS_mapmaking`
  (below), or call `wiener_filter_map` directly with
  `noise_inv_cov=inv(N)`. That last route works only on **unfiltered**
  data: the high-pass path fits `H d` with operator `H A`, whose noise
  covariance is `H N Hᵀ`, and `H` annihilates DC exactly — there is no
  inverse to pass.
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
  `noise_inv_cov_group` (overriding the flicker/white parameters). Each
  entry is the dense $N^{-1}$ or, for uncorrelated noise, its diagonal: a
  1D array of per-sample inverse variances (zero for a flagged sample),
  which may vary from sample to sample.
- Priors and `return_full_cov` work as in `HPW_mapmaking` (including
  the no-ridge rule above); IRLS convergence is controlled by
  `tol`/`min_iter`/`max_iter`. A system that is singular to working
  precision raises `LinAlgError`.
- Each IRLS iteration costs one reweighted normal-equations solve; the
  IRLS typically converges in ≲10 iterations. With 1/f noise, setup adds
  one dense `(n_time)²` covariance inversion per TOD.
- With uncorrelated noise (`gain_noise_params=None`, or 1D
  `noise_inv_cov_group` entries) $N^{-1}$ stays diagonal throughout, so
  memory and time grow linearly in `n_time` (see the table below).
- The lower-level pieces are exported too: `flicker_noise_cov` /
  `flicker_noise_inv_cov` (the Toeplitz builder) and `iterative_gls`
  (the faithful single-TOD hydra-tod solver).

### White noise, several TODs, sky only

When the 1/f noise is negligible, pass `gain_noise_params=None`. Leaving
out `Tsys_others_operator_group` makes the sky pixels the only parameters.

```python
mapmaker = GLS_mapmaking(beam_map=beam,
                         LST_deg_list_group=[lst_1, lst_2, ...],
                         azimuth_deg_list_group=[az_1, az_2, ...],
                         elevation_deg_list_group=[el_1, el_2, ...],
                         lat_deg=lat, threshold=0.7, nside_target=nside)
data = dict(
    TOD_group=[tod_1, tod_2, ...], dtime=2.0,
    gain_noise_params=None, white_noise_var=2.5e-6,
    known_injection_group=[trec_1, trec_2, ...],   # still in the TOD, not subtracted
)
sky_map, sky_uncertainty = mapmaker(**data)
samples = mapmaker.posterior_samples(1000, seed=0, **data)   # needs hydra-tod
samples.sky   # (1000, n_pix) posterior map draws
```

`posterior_samples` takes the same keywords as the map-maker and draws
maps from the same posterior, through hydra-tod; see
[Posterior map samples](#posterior-map-samples-through-hydra-tod). Drop
`known_injection_group` if the TOD contains no receiver temperature.

Under this (default, multiplicative) model the noise scales with the whole
system temperature, receiver included. A known receiver temperature
therefore stays in the TOD and goes in `known_injection_group`. If it was
subtracted earlier, add it back, or use `noise_model="additive"` for data
calibrated that way. A receiver temperature that is unknown but present in
the data cannot be left out, since the sky would absorb it. Model it as a
column of `Tsys_others_operator_group` and discard its estimate.

The table gives wall time and peak memory for random dense operators, with
`gain_noise_params=None` and the multiplicative model. Peak memory is the
resident memory added during the call, measured in a fresh process. Most
of it is the ordinary least-squares solve that starts the IRLS: the stacked
operators, plus LAPACK's own copy of them.

The noise covariance is block-diagonal across TODs: different TODs share
no noise, so a dense representation stores one $n_i \times n_i$ block of
$N^{-1}$ per TOD, $8\,n_i^2$ bytes. The column "Dense $N^{-1}$ blocks"
gives that storage, per TOD and summed over TODs.

| TODs × samples, parameters | Operators | Dense $N^{-1}$ blocks | Map-maker | 200 posterior draws |
|---|---|---|---|---|
| 2 × 200 000, 20 | 64 MB | 320 GB per TOD, 640 GB in all | 0.2 s, 0.21 GB | 1.2 s, 0.22 GB |
| 64 × 3 000, 200 | 307 MB | 72 MB per TOD, 4.6 GB in all | 0.9 s, 0.67 GB | 2.3 s, 0.69 GB |

The block storage is a lower bound on the dense path. It holds every block
at once and, while building or using one TOD's block, about three more
arrays of that size. Measured on the dense (1/f) path:

| TODs × samples, parameters | Dense $N^{-1}$ blocks | Peak memory, map-maker or 200 draws |
|---|---|---|
| 2 × 4 000, 20 | 256 MB | 0.70 GB (2.7×) |
| 16 × 3 000, 200 | 1.15 GB | 1.63–1.67 GB (1.4×) |

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

### Posterior map samples (through hydra-tod)

`GLS_mapmaking.posterior_samples` draws maps from the posterior that
`__call__` summarises by its estimate $\hat p$ and covariance. limTOD runs
the same GLS solve as `__call__`. It passes the estimate, operators,
$N^{-1}$, known temperatures and prior precision to
`hydra_tod.map_sampler.sample_map_posterior`, which draws the samples;
limTOD draws no random numbers itself. Inside hydra-tod:

- `hydra_tod.linear_sampler.params_space_oper_and_data` builds, per TOD,
  $U_i^\mathsf{T}\Sigma_i^{-1}U_i$ and the noise projection
  $U_i^\mathsf{T}\Sigma_i^{-1/2}$ at $\hat p$, for a dense or diagonal
  $N^{-1}$ and either noise model. With a diagonal $N^{-1}$ each time
  sample is weighted by its own inverse variance, and no
  $n_\mathrm{time}\times n_\mathrm{time}$ array is formed;
- `hydra_tod.linear_sampler.sample_p_v2` draws a zero-mean constrained
  realisation $\delta$, and each draw is $x = \hat p + \delta$, with

$$
(A + S^{-1})\,\delta =
  \sum_i U_i^\mathsf{T}\Sigma_i^{-1/2}\omega_i + S^{-1/2}\omega',
  \qquad \omega_i, \omega' \sim \mathcal{N}(0, I).
$$

Here $A = \sum_i U_i^\mathsf{T}\Sigma_i^{-1}U_i$ and $S^{-1}$ is the prior
precision. $\Sigma_i^{-1/2}$ stands for any factor $F$ with
$FF^\mathsf{T} = \Sigma_i^{-1}$. hydra-tod uses $F = \mathrm{diag}(1/(U_i\hat
p+\mu_i))\,L$ under the multiplicative model and $F = L$ under the additive
model, where $LL^\mathsf{T} = N_i^{-1}$: the Cholesky factor, the
element-wise square root of a diagonal, or $V\Lambda^{1/2}$ from an
eigendecomposition when $N_i^{-1}$ is only semi-definite. Each $x$ follows
$\mathcal{N}(\hat p, (A+S^{-1})^{-1})$, the estimate and covariance that
`__call__(..., return_full_cov=True)` returns. The prior mean enters only
through $\hat p$. Drawing $\delta$ and adding $\hat p$ keeps the draws
centred on $\hat p$ even before the IRLS has converged, and avoids the
rounding of a large $(A + S^{-1})\hat p$ when $A$ is ill-conditioned.

What is fixed and what is drawn:

| Quantity | In the draws |
|---|---|
| Sky pixels | drawn |
| Temperatures in `Tsys_others_operator_group` (e.g. a receiver temperature) | drawn with the sky |
| Temperatures in `known_injection_group` (e.g. a noise diode) | fixed |
| Gains, 1/f and white-noise parameters | fixed |

```python
samples = mapmaker.posterior_samples(
    500, seed=0,
    TOD_group=TOD_group, dtime=2.0,
    gain_noise_params=(1.335e-5, 1.099e-3, 2), white_noise_var=2.5e-6,
)
samples.sky        # (500, n_pix), or (500, npol, n_pix)
samples.others     # per-TOD other-Tsys draws, or None
samples.mean       # the GLS estimate the draws are centred on
samples.precision  # inverse of return_full_cov's posterior_cov
```

`posterior_samples` takes every keyword of the map-maker, so the noise and
data options are the same. `TOD_group` may hold any number of TODs, and all
of them constrain one joint posterior. White noise works through
`gain_noise_params=None` or 1D `noise_inv_cov_group` entries, with no
$n_\mathrm{time}\times n_\mathrm{time}$ arrays. The example above includes
1/f noise; [White noise, several TODs, sky only](#white-noise-several-tods-sky-only)
shows the white-noise call.

#### Multiplicative model

Under the multiplicative model $\Sigma$ depends on $p$. The sampler holds
it at $\hat p$. hydra-tod's Gibbs step does the same at its own GLS
estimate, which omits the prior; limTOD's $\hat p$ includes it. The exact
posterior keeps the $p$-dependence of $\Sigma$ in the quadratic form and in
$\log\det\Sigma(p)$. The difference scales with the fractional noise
amplitude. Over 20 noise realisations of the test geometry in
`tests/test_gls_sampler.py`, at limTOD's default noise:

| Fixed-$\Sigma$ Gaussian vs exact posterior | Largest value |
|---|---|
| Mode shift | 0.004σ |
| Curvature change | 0.9% |

Under the additive model the posterior is Gaussian and the draws are exact.

#### Notes

- Posterior sampling needs mpi4py and a version of hydra-tod that provides
  `hydra_tod.map_sampler`. Install them with
  `pip install mpi4py "git+https://github.com/zzhang0123/hydra-tod"`.
- If $A + S^{-1}$ is singular to working precision, some parameter
  combination is constrained by neither the data nor the prior, and the
  call raises `LinAlgError`; no ridge is added.
- `seed` is passed to hydra-tod as its `rng`. A non-negative integer or a
  `numpy.random.Generator` supplies every random number and leaves NumPy's
  global RNG state untouched. `None` draws from, and advances, the global
  state, as the rest of hydra-tod does.
- Cost per TOD at setup, dense $N^{-1}$: one factorisation of the
  $n_\mathrm{time}\times n_\mathrm{time}$ matrix and one
  $(n_\mathrm{params}, n_\mathrm{time})$ projection. Diagonal $N^{-1}$:
  one $O(n_\mathrm{params}^2 n_\mathrm{time})$ product
  $U^\mathsf{T}\mathrm{diag}(w)U$, and no $n_\mathrm{time}\times
  n_\mathrm{time}$ array. Once per call: an eigenvalue check of
  $A + S^{-1}$. Cost per draw: one
  $(n_\mathrm{params}, n_\mathrm{time})$ matrix-vector product per TOD and
  one dense $O(n_\mathrm{params}^3)$ solve, plus a Cholesky factorisation
  of $S^{-1}$ when the prior is non-diagonal.
- A semi-definite dense `noise_inv_cov_group` entry (flagged samples, or
  a whole TOD, zeroed out of $N^{-1}$) is accepted, as in `__call__`.
  hydra-tod then takes its square root from an eigendecomposition instead
  of a Cholesky factor. A diagonal entry may hold zeros; its square root
  is taken element by element.
- A 2D entry in `Tsys_other_prior_inv_cov_group` makes $S^{-1}$
  non-diagonal. $S^{-1}$ must then be positive definite over every
  parameter, sky included, because `sample_p_v2` draws the prior term with
  $\mathrm{cholesky}(S^{-1})$. Give the sky a prior as well
  (`Tsky_prior_inv_cov_diag`), or pass the other-Tsys priors as 1D
  diagonals. With diagonal priors, any parameter may have zero prior
  precision.
- The draws are conditional on $N$. A wrong $N$ miscalibrates their spread
  by the same factors as the reported $\sigma$ in the table under "What a
  wrong noise covariance costs". Sampling the noise parameters needs
  hydra-tod's full Gibbs sampler.

## JAX alternative

For differentiable map-making inside a JAX pipeline, the same
normal-equations solve runs matrix-free: conjugate gradients on top of the
[limtod_jax](limtod-jax.md) projector and its exact adjoint, with the
operator never formed. limTOD ships the projector and the adjoint; the CG
solver belongs to the pipeline that consumes them
([Downstream](index.md#downstream)).
