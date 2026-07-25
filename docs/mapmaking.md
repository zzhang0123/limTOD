# Map-making with `HPW_mapmaking`

`HPW_mapmaking` reconstructs sky maps from TOD by combining an optional
Butterworth **h**igh-**p**ass filter (to suppress 1/f drifts) with a
**W**iener-filter solve of the map-making normal equations, with optional
priors on the sky and on other system-temperature components.

For a complete worked example see
[mm_example.ipynb](../examples/mm_example.ipynb): simulating several TOD
sets at different elevations, building the map-maker, solving, and
visualizing the reconstruction.

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
    regularization=1e-12,                   # numerical stabilizer
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

- With `use_high_pass=False`, `cutoff_freq_group` may be `None` — the
  solve then uses the unfiltered TOD and operator.
- The high-pass filter is applied consistently to both the data and the
  operator (`filtfilt`, represented as a dense matrix), so the estimator
  stays unbiased for the retained modes.
- Cost is dominated by dense `(n_selected_pixels)²` linear algebra —
  control it with `threshold` and `nside_target`.

## JAX alternative

For differentiable map-making inside a JAX pipeline,
[replicant-telescope](https://github.com/zzhang0123/replicant-telescope)'s
`SkySpaceFilter` implements the same normal-equations solve with
matrix-free conjugate gradients on top of the
[limtod_jax](limtod-jax.md) projector.
