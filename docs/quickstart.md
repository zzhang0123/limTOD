# Quick start

## Simulate multi-frequency TOD

A MeerKAT-like scan (the sky model here needs the `[gdsm]` extra; pass
your own `sky_func` to go without):

```python
from limTOD import TODSim, example_scan

simulator = TODSim(
    ant_latitude_deg=-30.7130, ant_longitude_deg=21.4430, ant_height_m=1054,
    beam_nside=256, sky_nside=256,
)
time_list, azimuth_list = example_scan()
tod, sky_tod, gain_noise = simulator.generate_TOD(
    freq_list=[950, 1000, 1050],          # MHz
    time_list=time_list,
    azimuth_deg_list=azimuth_list,
    elevation_deg=41.5,
)                                          # each (n_freq, n_time)
```

See the [TOD simulation guide](tod-simulation.md) for the noise model,
MPI parallelism, and troubleshooting.

## Make a map back from the TOD

```python
from limTOD import GLS_mapmaking, generate_LSTs_deg

lst_deg = generate_LSTs_deg(-30.7130, 21.4430, 1054, time_list)
mm = GLS_mapmaking(
    beam_map=beam_map,                    # HEALPix beam at this frequency
    LST_deg_list_group=lst_deg,
    lat_deg=-30.7130,
    azimuth_deg_list_group=azimuth_list,
    elevation_deg_list_group=41.5 + 0 * azimuth_list,
    threshold=0.01,
)
sky_map, sky_unc = mm(TOD_group=tod[0], dtime=2.0)   # one freq channel
```

The [map-making guide](mapmaking.md) covers both estimators (high-pass +
Wiener, and the GLS with the full 1/f noise covariance) and when to use
which.

## The same chain, differentiable in JAX

With the `[jax]` extra:

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import limtod_jax as ltj

sky_alm = ltj.map2alm_quad(sky_map, nside=nside, lmax=lmax)
psi, theta, phi = ltj.zyz_of_pointing(lst_deg, lat_deg, az_deg, el_deg, 0.0)
tod = ltj.generate_tod_sky(
    beam_alm, sky_alm, jnp.stack([psi, theta, phi], axis=-1), lmax=lmax,
)
grad = jax.grad(lambda b: ltj.generate_tod_sky(
    b, sky_alm, jnp.stack([psi, theta, phi], axis=-1), lmax=lmax).sum().real
)(beam_alm)                                # d(TOD sum)/d(beam alms)
```

See the [limtod_jax guide](limtod-jax.md) for the exactness contract and
precision requirements.

## Worked notebooks

- [TODsim_examples.ipynb](https://github.com/zzhang0123/limTOD/blob/main/examples/TODsim_examples.ipynb)
  — simulation walkthrough
- [mm_example.ipynb](https://github.com/zzhang0123/limTOD/blob/main/examples/mm_example.ipynb)
  — map-making workflow
