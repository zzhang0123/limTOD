# API reference (numpy package)

Signatures of the public `limTOD` API. Parameter semantics are described in
[tod-simulation.md](tod-simulation.md) and [mapmaking.md](mapmaking.md);
docstrings carry the authoritative per-argument detail. For the JAX API see
[limtod-jax.md](limtod-jax.md).

## Simulation

### `TODSim`

```python
TODSim(ant_latitude_deg=-30.7130, ant_longitude_deg=21.4430, ant_height_m=1054,
       beam_func=example_beam_map, sky_func=GDSM_sky_model,
       beam_nside=256, sky_nside=256)
```

Main simulator class. Methods:

```python
TODSim.simulate_sky_TOD(freq_list, time_list, azimuth_deg_list, elevation_deg,
                        selfrot_deg_list=None,
                        start_time_utc="2019-04-23 20:41:56.397",
                        return_LSTs=False, nside_hires=None,
                        normalize_beam=False, horizontal_mask=None,
                        truncate_frac_thres=1e-10)
# -> sky_TOD (nfreq, ntime) [, LST_deg_list]

TODSim.generate_TOD(freq_list, time_list, azimuth_deg_list,
                    selfrot_deg_list=None, elevation_deg=41.5,
                    start_time_utc="2019-04-23 20:41:56.397",
                    Tsys_others_TOD=None, background_gain_TOD=None,
                    gain_noise_TOD=None,
                    gain_noise_params=[1.335e-5, 1.099e-3, 2],
                    white_noise_var=None, return_LSTs=False,
                    nside_hires=None, normalize_beam=False,
                    horizontal_mask=None, truncate_frac_thres=1e-10)
# -> overall_TOD, sky_TOD, gain_noise_TOD (each (nfreq, ntime)) [, LST_deg_list]
```

### `generate_TOD_sky`

```python
generate_TOD_sky(beam_map, sky_map, LST_deg_list, lat_deg, azimuth_deg_list,
                 elevation_deg_list, selfrot_deg_list, nside_hires=None,
                 normalize_beam=False, horizontal_mask=None,
                 truncate_frac_thres=1e-10)
# -> (ntime,) sky TOD for one beam/sky pair
```

Function-level core of the simulation (no LST computation, no noise): beam
map → alms → per-pointing rotation → beam-weighted sum. This is the
function the JAX port reproduces (with `truncate_frac_thres=0.0`, see
[limtod-jax.md](limtod-jax.md)).

## Pointing geometry

```python
zyzyz2zyz(alpha, beta, gamma, delta, chi, output_degrees=False)
# -> (psi, theta, phi): collapse R_z(chi)R_y(delta)R_z(gamma)R_y(beta)R_z(alpha)
#    to R_z(phi)R_y(theta)R_z(psi); degrees in, radians out (by default)

zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg, selfrot_deg)
# -> (psi, theta, phi) [radians] for the pointing; see theory.md for the
#    alpha..chi mapping (note azimuth enters with a minus sign)

generate_LSTs_deg(ant_latitude_deg, ant_longitude_deg, ant_height_m,
                  time_list, start_time_utc="2019-04-23 20:41:56.397")
# -> (ntime,) apparent Local Sidereal Time [deg] via astropy

pointing_beam_in_eq_sys(beam_alm, LST_deg, lat_deg, azimuth_deg,
                        elevation_deg, selfrot_deg, nside, normalize=True,
                        horizontal_mask=None, truncate_frac_thres=1e-10)
# -> pointed beam map in equatorial coordinates (rotates alms, synthesizes,
#    optionally masks/truncates/normalizes)
```

## Beam and sky models

```python
example_beam_map(*, freq, nside, FWHM_major=1.1, FWHM_minor=1.1)
# -> (npix,) elliptical-Gaussian beam (achromatic toy model)

example_symm_beam_map(*, freq, nside, FWHM=1.1)
# -> (npix,) symmetric Gaussian beam, sum normalized to 1

GDSM_sky_model(*, freq, nside)
# -> (npix,) Global Sky Model map at `freq` [MHz]; requires the [gdsm] extra

generate_gaussian_field(freqs, nside, amp, alpha=1.0, beta=1.0, xi=1.0,
                        f_ell=None, nu_ref=300.0, ell_ref=100.0, fwhm=0.0,
                        seed=None, min_eigval=1e-10)
# -> (nfreq, npix) correlated Gaussian sky realizations from
#    C_ell(nu, nu') = A f_ell ((nu nu')/nu_ref^2)^beta exp(-ln^2(nu/nu')/2xi^2)
```

## Map-making

```python
HPW_mapmaking(*, beam_map, LST_deg_list_group, lat_deg,
              azimuth_deg_list_group, elevation_deg_list_group,
              selfrot_deg_list_group=None, threshold=0.01,
              Tsys_others_operator_group=None, nside_hires=None,
              nside_target=None, beam_truncate_frac_thres=None)
# callable: see mapmaking.md for the solve signature and returns

get_filtfilt_matrix(n_samples, b, a)      # dense filtfilt operator
HP_filter_TOD(...)                        # high-pass filter TOD arrays
wiener_filter_map(...)                    # Wiener solve, low-level
simple_wiener_map(...)                    # convenience wrapper
truncate_stacked_beam(...)                # beam-threshold pixel selection
generate_sky2sys_projection(...)          # (ntime, n_selected_pixels) operator rows
```

## Utilities

```python
example_scan(az_s=-60.3, az_e=-42.3, dt=2.0, n_repeats=5)
# -> (time_list, azimuth_list) simple raster scan

limTOD.mpiutil
# rank / size / rank0 / world, partition_list_mpi, parallel_map_gather,
# barrier — serial fallback (rank=0, size=1, world=None) without mpi4py;
# raises RuntimeError if launched under mpirun without mpi4py
# (escape hatch: LIMTOD_FORCE_SERIAL=1)
```
