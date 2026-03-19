# limTOD: Time-Ordered Data Simulator for single-dish (autocorrelation) line intensity mapping measurements

**limTOD** is a Python package for simulating Time-Ordered Data (TOD) from single-dish/autocorrelation observations.

The `TODSim` class provide the main simulation functionality, for which it calculate the TOD assuming asymetric beams and flicker (1/f) noise mode with HEALPix sky and beam models. Althoug symmetric beams can be used, the calculation can be unnecessarily slow compared as direct convolution is done through beam-weighted sum as opposed to convolving the sky with the healpy smoothing function.

A simple (but sophisticated) map-making class, `HPW_mapmaking` is also provided for converting the simulated TODs into maps.

� **For the latest updates and release notes, see [CHANGELOG.md](CHANGELOG.md)**

## Table of Contents

* [Citation](#citation)
* [Installation](#installation)
* [Quick Start](#quick-start)
  + [Input Parameters](#input-parameters)
  + [Output Parameters](#output-parameters)
* [Examples](#examples)
* [Theoretical Background](#theoretical-background)
* [API Reference](#api-reference)
* [Map-Making with HPW_mapmaking](#map-making-with-hpw_mapmaking)
* [Performance Considerations](#performance-considerations)
* [Troubleshooting](#troubleshooting)
* [License](#license)
* [Maintainers](#maintainers)

## Citation

If you use limTOD in your research, please cite:

```bibtex
@misc{zhang2026jointbayesiancalibrationmapmaking,
      title={Joint Bayesian calibration and map-making for intensity mapping experiments}, 
      author={Zheng Zhang and Philip Bull and Mario G. Santos and Ainulnabilah Nasirudin},
      year={2026},
      eprint={2509.10992},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2509.10992}, 
}
```

## Installation

You will need Python version 3.8 or higher and `pip` to install the package.

For a quick installation, run the following command

```bash
pip install git+https://github.com/zzhang0123/limTOD.git
```

or installed from source, 

```bash
git clone https://github.com/zzhang0123/limTOD.git
cd limTOD
python -m pip install .
```

Additionally, to run the example notebooks, do

```bash
python -m pip install jupyter matplotlib ipykernel
python -m ipykernel install --name "limtod" --user
```

This will install `jupyter` and `matplotlib` for the notebook, as well as installing the currently used Python executable as a selectable kernel named "limtod" for the notebook.

### Virtual Environment

It is recommended that the above installation is done inside a Python virtual environment.

For example, run the following command to create a virtual environment named `limtod` in the `~/venv` directory with the [virtualenv](https://virtualenv.pypa.io/en/latest/index.html) tool, 

```bash
virtualenv ~/venv/limtod
```

or equivalently with the Python3 built-in `venv` tool, 

```bash
python -m venv ~/venv/limtod
```

The virtual environment can then be activated by, 

```bash
source ~/venv/limtod/bin/activate
```

then, one can proceed to install from git or from source as above.

### Required Dependencies

All required dependencies will be automatically installed when installing via `pip` .

```
numpy >= 1.19.0
healpy >= 1.14.0
astropy >= 4.0.0
scipy >= 1.5.0
tqdm >= 4.60.0
mpi4py >= 3.0.0 (for parallel processing)
pygdsm >= 1.2.0 (for Global Sky Model)
mpmath >= 1.2.0 (for flicker noise modeling)
```

### Development Setup

The `pip` command should be run with `-e` flag (or `--editable` ) and `[dev]` varient

```bash
pip install -e ".[dev]"
```

## Simulating TOD with TODSim Class

The `TODSim` class provides the main simulation functionality. It assumes a single-dish telescope with a single-pixel beam (thus multiple instances must be run to simulate from a multi-dish/multi-beam intensity mapping telecope) and return the sky temperature, gain noise and overall TOD. Check the following sections for more details.

### TODSim Quick Start

```python
import numpy as np
from limTOD import TODSim, example_scan

# Initialize the simulator with MeerKAT coordinates
simulator = TODSim(
    ant_latitude_deg=-30.7130,   # MeerKAT latitude
    ant_longitude_deg=21.4430,   # MeerKAT longitude
    ant_height_m=1054,           # MeerKAT altitude
    beam_nside=256,              # HEALPix resolution for beam
    sky_nside=256                # HEALPix resolution for sky
)

# Generate a simple scanning pattern
time_list, azimuth_list = example_scan()

# Simulate TOD for multiple frequencies
freq_list = [950, 1000, 1050]  # MHz
tod_array, sky_tod, gain_noise = simulator.generate_TOD(
    freq_list=freq_list,
    time_list=time_list,
    azimuth_deg_list=azimuth_list,
    elevation_deg=41.5
)

print(f"Generated TOD shape: {tod_array.shape}")  # (3, n_time)
```

* **For detailed coordinate system definitions, see [conventions.pdf](conventions.pdf).**
* **For working examples of TOD simulation, see [examples/TODsim_examples.ipynb](examples/TODsim_examples.ipynb)**
* **For working examples of HighPass+Wiener mapmaking, see map-making workflow ([examples/mm_example.ipynb](examples/mm_example.ipynb))**

### Input Parameters

#### Telescope Configuration

* **ant_latitude_deg** (`float`): Latitude of the antenna/site in degrees.
* **ant_longitude_deg** (`float`): Longitude of the antenna/site in degrees.
* **ant_height_m** (`float`): Height of the antenna/site in meters.
* **beam_func** (`function`): Function that takes _keyword-only_ inputs, two of which must be `freq` (for frequency) and `nside` and returns the HEALPix beam map of shape (npix, ). Optional keywords can be passed to the function for customisation.
* **sky_func** (`function`): Function that takes _keyword-only_ inputs, two of which must be `freq` (for frequency) and `nside` and returns the HEALPix sky map of shape (npix, ). Optional keywords can be passed to the function for customisation.
* **beam_nside** (`int`, optional): The nside parameter for the beam Healpix maps. Should be large enough to resolve beam features.
* **sky_nside** (`int`, optional): The nside parameter for the sky Healpix maps. Decides how the sky map is parametrized.

#### Observation Parameters

* **freq_list** (`list` or `array`): List of frequencies.
* **time_list** (`list` or `array`): List of time offsets in seconds (for each observation time) from start_time_utc.
* **azimuth_deg_list** (`list` or `array`): List of azimuth values in degrees for each observation.
* **elevation_deg** (`list` or `float`):
  + If float: Elevation value in degrees universal for all observations.
  + If list: List of elevation values in degrees for each observation.
* **start_time_utc** (`str`): Start time in UTC (e.g. "2019-04-23 20:41:56.397").

#### Noise and Calibration Parameters (Optional)

* **Tsys_others_TOD** (`array`, optional): Array of the remaining system temperature TOD (shape: nfreq x ntime). Default is None (no other components).
* **background_gain_TOD** (`array`, optional): Array of background gain TOD (shape: nfreq x ntime). Default is None (unity gain).
* **gain_noise_TOD** (`array`, optional): Array of gain noise TOD (shape: nfreq x ntime). Default is None (no gain noise).
* **gain_noise_params** (`list`, optional): List of parameters [f0, fc, alpha] for generating gain noise if gain_noise_TOD is None. Default is [1.335e-5, 1.099e-3, 2].
* **white_noise_var** (`float`, optional): Variance of white noise to be added. Default is None (uses default value of 2.5e-6).
* **return_LSTs** (`bool`, optional): If True, return the LST values along with the TODs. Default is False.
* **nside_hires** (`int`, optional): If provided, upgrade the beam map to this nside before processing. Useful for narrow beams. Default is None.
* **normalize_beam** (`bool`, optional): If True, normalize the beam map to have a sum of 1 before computing the weighted sum. Default is False.
* **truncate_frac_thres** (`float`, optional): Fractional threshold for beam truncation. Pixels below this fraction of the peak are set to zero. Default is 1e-10.

### Output Parameters

* **overall_TOD** (`ndarray`): Complete TOD with all components (nfreq × ntime)
* **sky_TOD** (`ndarray`): Sky signal component only (beam-weighted sum of sky maps, nfreq × ntime)
* **gain_noise_TOD** (`ndarray`): Gain noise component (nfreq × ntime)
* **LST_deg_list** (`ndarray`, optional): LST values in degrees, only returned if `return_LSTs=True`

### TOD Simulation Examples

The [TODSim_examples.ipynb](examples/TODSim_examples.ipynb) provides several more examples on the TOD simulation.

### Theoretical Background

#### Time-Ordered Data (TOD) Model

The complete TOD model implemented in this package follows the equation:

```
TOD(ν,t) = G_bg(ν,t) × [1 + G_noise(ν,t)] × [sky_TOD(ν,t) + Tsys_others(ν,t)] × [1 + η(t)]
```

Where:

* `G_bg(ν,t)`: Background gain pattern
* `G_noise(ν,t)`: Gain noise fluctuations (1/f noise)
* `sky_TOD(ν,t)`: Sky signal convolved with beam
* `Tsys_others(ν,t)`: All the other system temperature components
* `η(t)`: White noise component

#### Sky Signal Computation

The sky signal is computed as the convolution of the beam pattern with the sky brightness temperature:

```
sky_TOD(ν,t) = ∫ B(θ,φ,ν,t) × T_sky(θ,φ,ν) dΩ
```

This is efficiently computed using HEALPix spherical harmonics:

1. Convert beam map to spherical harmonic coefficients: `B_lm = map2alm(B)`
2. Rotate beam to pointing direction using Euler angles
3. Compute beam-weighted sum with sky map

#### Coordinate Transformations

The package handles coordinate transformations between:

1. **Local Telescope frame** (time, Azimuth, Elevation) → **Equatorial frame** (RA, Dec)
2. Representations of the transformation: **ZYZY Euler angles** → **ZYZ Euler angles** for HEALPix rotations

#### Detailed Workflow

Step 1: Scan Specifications → LST Sequence

* Convert UTC timestamps to Local Sidereal Time using telescope location
* Function: `generate_LSTs_deg()`

Step 2: Telescope Pointing → ZYZY Angles 

* Map telescope parameters to natural rotation sequence:
  + α = LST (Earth's rotation tracking)
  + β = 90° - latitude (site location correction)
  + γ = azimuth (local pointing direction)
  + δ = elevation - 90°  (altitude correction)

Step 3: ZYZY → ZYZ Conversion

* Convert to HEALPix-compatible Euler angles using `zyzy2zyz()`
* This handles the mathematical transformation: R_zyzy = R_y(δ)R_z(γ)R_y(β)R_z(α) → R_zyz = R_z(φ)R_y(θ)R_z(ψ)

Step 4: Beam Rotation in Spherical Harmonic Space

* Apply rotation to beam's alm coefficients using `pointing_beam_in_eq_sys()`
* Efficiently rotates beam pattern without pixel-by-pixel calculations
* Function: `_rotate_healpix_map()` calls `healpy.rotate_alm()`

Step 5: Sky Integration

* Compute beam-weighted sum: SKY_TOD_SAMPLE = ∫ B_pointed(θ, φ) × T_sky(θ, φ) dΩ
* Function: `_beam_weighted_sum()`

This approach allows accurate simulation of how the telescope beam tracks celestial sources as the Earth rotates and the telescope points to different directions.

#### Mathematical Conventions

For detailed mathematical formulations, coordinate system definitions, and algorithmic conventions used in this package, please refer to [Mathematical Conventions Document](conventions.pdf).

This document contains:

* Coordinate system definitions and transformations
* Euler angle conventions (ZYZY ↔ ZYZ)
* Spherical harmonics formulations
* Beam convolution algorithms
* Noise model specifications

### API Reference

#### Core Classes

##### `TODSim`

Main simulator class for generating time-ordered data.

```python
class TODSim:
    def __init__(self,
                 ant_latitude_deg=-30.7130,
                 ant_longitude_deg=21.4430,
                 ant_height_m=1054,
                 beam_func=example_beam_map,
                 sky_func=GDSM_sky_model,
                 beam_nside=256,
                 sky_nside=256)
```

**Parameters:**

* `ant_latitude_deg` (float): Antenna latitude in degrees
* `ant_longitude_deg` (float): Antenna longitude in degrees
* `ant_height_m` (float): Antenna height above sea level in meters
* `beam_func` (callable): Function returning beam map given (freq, nside) as keyword arguments
* `sky_func` (callable): Function returning sky map given (freq, nside) as keyword arguments
* `beam_nside` (int): HEALPix resolution parameter for beam maps. Should be large enough to resolve beam features.
* `sky_nside` (int): HEALPix resolution parameter for sky maps. Decides how the sky map is parametrized.

#### Core Functions

##### `generate_TOD()`

Generate complete time-ordered data including all noise components.

```python
def generate_TOD(self,
                freq_list,
                time_list,
                azimuth_deg_list,
                elevation_deg=41.5,
                start_time_utc="2019-04-23 20:41:56.397",
                Tsys_others_TOD=None,
                background_gain_TOD=None,
                gain_noise_TOD=None,
                gain_noise_params=[1.335e-5, 1.099e-3, 2],
                white_noise_var=None,
                return_LSTs=False,
                nside_hires=None,
                normalize_beam=False,
                truncate_frac_thres=1e-10)
```

**Parameters:**

* `freq_list` (array_like): Observation frequencies in MHz
* `time_list` (array_like): Time offsets from start time in seconds
* `azimuth_deg_list` (array_like): Time-ordered azimuth angles in degrees
* `elevation_deg` (float or array_like): Elevation angle(s) in degrees
* `start_time_utc` (str): UTC start time in ISO format
* `Tsys_others_TOD` (array_like, optional): All the other system temperature components
* `background_gain_TOD` (array_like, optional): Background gain variations
* `gain_noise_TOD` (array_like, optional): Pre-computed gain noise
* `gain_noise_params` (list): [f0, fc, alpha] for 1/f noise generation, if gain_noise_TOD is not provided
* `white_noise_var` (float, optional): White noise variance
* `return_LSTs` (bool, optional): If True, return LST values along with the TODs. Default is False.
* `nside_hires` (int, optional): Upgrade beam map to this nside before processing. Useful for narrow beams. Default is None.
* `normalize_beam` (bool, optional): If True, normalize the beam map before computing. Default is False.
* `truncate_frac_thres` (float, optional): Fractional threshold for beam truncation. Default is 1e-10.

**Returns:**

* `overall_TOD` (ndarray): Complete TOD with all components (nfreq × ntime)
* `sky_TOD` (ndarray): Sky signal component only (beam-weighted sum of sky maps, no gain and no noise. Shape: nfreq × ntime)
* `gain_noise_TOD` (ndarray): Gain noise component (nfreq × ntime)
* `LST_deg_list` (ndarray, optional): LST values in degrees, only returned if `return_LSTs=True`

##### `simulate_sky_TOD()`

Generate sky signal component of TOD.

```python
def simulate_sky_TOD(self,
                    freq_list,
                    time_list,
                    azimuth_deg_list,
                    elevation_deg,
                    start_time_utc="2019-04-23 20:41:56.397",
                    return_LSTs=False,
                    nside_hires=None,
                    normalize_beam=False,
                    truncate_frac_thres=1e-10)
```

**Parameters:**

* `freq_list`, `time_list`, `azimuth_deg_list`, `elevation_deg`, `start_time_utc`: Same as `generate_TOD()`
* `return_LSTs` (bool, optional): If True, return LST values along with the TODs. Default is False.
* `nside_hires` (int, optional): Upgrade beam map to this nside before processing. Useful for narrow beams. Default is None.
* `normalize_beam` (bool, optional): If True, normalize the beam map before computing. Default is False.
* `truncate_frac_thres` (float, optional): Fractional threshold for beam truncation. Default is 1e-10.

**Returns:**

* `sky_TOD` (ndarray): Sky signal TOD (nfreq × ntime)
* `LST_deg_list` (ndarray, optional): LST values in degrees, only returned if `return_LSTs=True`

#### Utility Functions

##### `example_scan()`

Generate a simple raster scanning pattern.

```python
def example_scan(az_s=-60.3, az_e=-42.3, dt=2.0, n_repeats=5)
```

**Parameters:**

* `az_s` (float): Starting azimuth in degrees
* `az_e` (float): Ending azimuth in degrees
* `dt` (float): Time step in seconds
* `n_repeats` (int): Number of scan repetitions. Default is 5.

**Returns:**

* `time_list` (ndarray): Time offsets in seconds
* `azimuth_list` (ndarray): Azimuth angles in degrees

##### `generate_LSTs_deg()`

Compute Local Sidereal Time for observation times.

```python
def generate_LSTs_deg(ant_latitude_deg, ant_longitude_deg, ant_height_m,
                     time_list, start_time_utc="2019-04-23 20:41:56.397")
```

**Returns:**

* `LST_deg_list` (ndarray): LST values in degrees

#### Coordinate Transformation Functions

##### `zyzy2zyz()`

Convert ZYZY Euler angles to ZYZ convention for HEALPix rotations.

```python
def zyzy2zyz(alpha, beta, gamma, delta, output_degrees=False)
```

**Parameters:**

* `alpha` (float): First Z rotation angle in degrees
* `beta` (float): First Y rotation angle in degrees  
* `gamma` (float): Second Z rotation angle in degrees
* `delta` (float): Second Y rotation angle in degrees
* `output_degrees` (bool): If True, return angles in degrees; otherwise radians

**Returns:**

* `(psi, theta, phi)` (tuple): ZYZ Euler angles for HEALPix rotation

**Mathematical Background:**

* **ZYZY rotation**: R = R_y(δ) × R_z(γ) × R_y(β) × R_z(α)
* **ZYZ rotation**: R = R_z(φ) × R_y(θ) × R_z(ψ)

This conversion is necessary because telescope pointing naturally follows ZYZY rotations (combining Earth rotation and local pointing), while HEALPix requires ZYZ convention.

##### `zyz_of_pointing()`

Generate ZYZ Euler angles from telescope pointing parameters.

```python
def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg)
```

**Parameters:**

* `LST_deg` (float): Local Sidereal Time in degrees
* `lat_deg` (float): Telescope latitude in degrees
* `azimuth_deg` (float): Pointing azimuth in degrees
* `elevation_deg` (float): Pointing elevation in degrees

**Returns:**

* `(psi, theta, phi)` (tuple): ZYZ Euler angles in radians for `hp.rotate_alm()`

**Algorithm:**

1. Convert pointing parameters to ZYZY angles:
   * α = LST (Earth rotation)
   * β = 90° - lat (latitude correction)
   * γ = azimuth (local pointing direction)
   * δ = 90° - elevation (elevation correction)
2. Transform to ZYZ using `zyzy2zyz()`

This maps the natural telescope coordinate system to the mathematical framework required for spherical harmonic rotations.
```

##### `pointing_beam_in_eq_sys()`

Point a beam pattern to specific telescope coordinates in the equatorial system.

```python
def pointing_beam_in_eq_sys(beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside,
                            normalize=True, truncate_frac_thres=1e-10)
```

**Parameters:**

* `beam_alm` (array): Spherical harmonic coefficients of the beam in its native orientation
* `LST_deg` (float): Local Sidereal Time in degrees
* `lat_deg` (float): Latitude of the observation site in degrees
* `azimuth_deg` (float): Azimuth of the pointing in degrees
* `elevation_deg` (float): Elevation of the pointing in degrees
* `nside` (int): HEALPix resolution parameter
* `normalize` (bool, optional): If True, normalize the pointed beam map to sum to 1. For Stokes Q, U, V, they are scaled by the same factor as Stokes I. Default is True.
* `truncate_frac_thres` (float, optional): Fractional threshold for beam truncation. Pixels below this fraction of the peak are set to zero before normalization. Default is 1e-10.

**Returns:**

* `beam_pointed` (ndarray): The rotated beam map in equatorial coordinates

**Algorithm:**

1. Convert telescope pointing (LST, lat, az, el) to ZYZ Euler angles using `zyz_of_pointing()`
2. Rotate the beam's spherical harmonic coefficients using `_rotate_healpix_map()`
3. Return the pointed beam map in equatorial coordinate system

This function is central to the TOD simulation as it enables the beam pattern to track celestial sources as the Earth rotates.

##### Internal Helper Functions

###### `_rotate_healpix_map()`

Rotate a HEALPix map using Euler angles in spherical harmonic space.

```python
def _rotate_healpix_map(alm, psi_rad, theta_rad, phi_rad, nside, return_map=True)
```

**Parameters:**

* `alm` (array): Spherical harmonic coefficients of the map
* `psi_rad`,      `theta_rad`,  `phi_rad` (float): ZYZ Euler angles in radians
* `nside` (int): HEALPix resolution parameter
* `return_map` (bool): If True, return rotated map; if False, return rotated alm

**Algorithm:**

1. Creates a copy of input spherical harmonic coefficients
2. Applies rotation using `healpy.rotate_alm()` with ZYZ convention
3. Converts back to map format if requested

###### `_beam_weighted_sum()`

Compute the convolution integral of beam and sky.

```python
def _beam_weighted_sum(beam_map, sky_map, normalize=False)
```

**Parameters:**

* `beam_map` (array): HEALPix beam pattern (should be pre-normalized unless `normalize=True`)
* `sky_map` (array): HEALPix sky brightness temperature map
* `normalize` (bool, optional): If True, normalize the beam map before computing. Default is False.

**Returns:**

* `float`: Beam-weighted integral ∫ B(θ, φ) × T_sky(θ, φ) dΩ

This implements the discrete version of the beam convolution integral that produces each TOD sample.

##### Beam and Sky Functions

###### `example_beam_map()`

Generate elliptical Gaussian beam pattern.

```python
def example_beam_map(*, freq, nside, FWHM_major=1.1, FWHM_minor=1.1)
```

###### `GDSM_sky_model()`

Generate sky map using Global Sky Model.

```python
def GDSM_sky_model(*, freq, nside)
```

## Map-Making with HPW_mapmaking

### Overview

The `HPW_mapmaking` class provides a sophisticated map-making pipeline that combines high-pass filtering and Wiener filtering to reconstruct sky maps from Time-Ordered Data (TOD). This implementation handles 1/f noise through high-pass filtering while optimally recovering sky signals using Wiener filtering with optional priors.

### HPW_mapmaking Quick Start

```python
sky_map, sky_uncertainty = mapmaker(
    TOD_group=TOD_group,
    dtime=dtime,
    cutoff_freq_group=cutoff_freq_group,
    gain_group=None,
    known_injection_group=None,
    Tsky_prior_mean=None,
    Tsky_prior_inv_cov_diag=None,
    Tsys_other_prior_mean_group=None,
    Tsys_other_prior_inv_cov_group=None,
    regularization=1e-12,
    return_full_cov=False,
    filter_order=4
)
```

**Parameters:**

* `TOD_group` (list): List of TOD arrays, one per observation
* `dtime` (float): Time sampling interval in seconds
* `cutoff_freq_group` (list): High-pass filter cutoff frequencies in Hz
* `gain_group` (list, optional): Gain calibration factors for each TOD
* `known_injection_group` (list, optional): Known signals to subtract (e.g., calibration diodes)
* `Tsky_prior_mean` (array, optional): Prior mean for sky temperature
* `Tsky_prior_inv_cov_diag` (array, optional): Diagonal of prior inverse covariance matrix
* `Tsys_other_prior_mean_group` (list, optional): Prior means for other system parameters
* `Tsys_other_prior_inv_cov_group` (list, optional): Prior inverse covariances for other system parameters
* `regularization` (float): Regularization parameter for numerical stability
* `return_full_cov` (bool): If True, return full posterior covariance matrix
* `filter_order` (int): Order of the Butterworth high-pass filter. Default is 4.

**Returns:**

* `sky_map` (array): Reconstructed sky map(s) in temperature units
* `sky_uncertainty` (array): Per-pixel uncertainty estimates
* `Tsys_others_estimation_group` (list, optional): Estimated other system parameters
* `Tsys_others_uncertainty_group` (list, optional): Uncertainties for other system parameters

### Map-Making Examples

📓 **For a complete working example, see [examples/mm_example.ipynb](examples/mm_example.ipynb)**

This notebook demonstrates:

* Simulating multiple TOD sets at different elevations
* Initializing the `HPW_mapmaking` class with keyword arguments
* Performing map-making with high-pass + wiener filtering
* Visualizing reconstructed sky maps using `gnomview_patch`

### Theoretical Background

The map-making process follows these key steps:

1. **High-Pass Filtering**: Remove low-frequency drifts and 1/f noise using a Butterworth filter
2. **Forward Modeling**: Build an operator that maps sky parameters to TOD samples
3. **Wiener Filtering**: Solve the inverse problem optimally:

```
   x̂ = (A^T N^{-1} A + S^{-1})^{-1} (A^T N^{-1} d + S^{-1} μ)
   ```

   where:

* `A` : System operator (beam convolution + instrumental effects)
* `N` : Noise covariance matrix
* `S` : Signal prior covariance matrix
* `d` : Measured TOD data
* `μ` : Prior mean for sky parameters

### API Reference

#### `HPW_mapmaking` Class

```python
class HPW_mapmaking:
    def __init__(self, *,
                 beam_map,
                 LST_deg_list_group,
                 lat_deg,
                 azimuth_deg_list_group,
                 elevation_deg_list_group,
                 threshold=0.01,
                 Tsys_others_operator_group=None,
                 nside_hires=None,
                 nside_target=None,
                 beam_truncate_frac_thres=None)
```

**Parameters** (all keyword-only):

* `beam_map` (array): HEALPix beam pattern. Can be:
  + 1D array (length npix) for intensity-only (I)
  + 2D array (3 × npix) for polarization (I, Q, U)
  + 2D array (4 × npix) for full Stokes (I, Q, U, V)
* `LST_deg_list_group` (list): LST values in degrees for each TOD or list of LST lists
* `lat_deg` (float): Observation site latitude in degrees
* `azimuth_deg_list_group` (list): Azimuth angles in degrees for each TOD
* `elevation_deg_list_group` (list): Elevation angles in degrees for each TOD
* `threshold` (float): Fractional beam response threshold for pixel selection (e.g., 0.01 = 1% of peak)
* `Tsys_others_operator_group` (list, optional): List of operators for other system temperature components (e.g., receiver temperature variations)
* `nside_hires` (int, optional): If provided, upgrade the beam map to this nside before processing. Useful for narrow beams.
* `nside_target` (int, optional): Target nside for the output beam map. Should match the convention used in pixel indices.
* `beam_truncate_frac_thres` (float, optional): Fractional threshold for beam truncation. If None, uses `threshold` value. Note the difference: `threshold` selects which pixels to include in map-making, while `beam_truncate_frac_thres` truncates the beam map itself.

## Performance Considerations

### Parallel Processing

The package supports MPI parallelization over frequencies:

```python
# Automatic parallelization over frequency list
tod_array = sim.generate_TOD(freq_list=frequencies, ...)

# Run with MPI
# mpirun -n 4 python simulation_script.py
```

Per-frequency calculation will also benefit from internal threading of `numpy` , 
`scipy` and `healpy` .  We recommends giving each worker 2-4 CPU cores.

### Optimization Tips

1. **Use appropriate N_side**: Balance resolution vs. speed
2. **Batch processing**: Process multiple frequencies together

## Troubleshooting

Common error scenarios and solutions:

* **Installation errors**: make sure to use a Python virtual environment
* **Memory errors**: Reduce `nside` or process in smaller batches
* **Coordinate errors**: Check that Az/El values are within valid ranges
* **Time errors**: Ensure UTC timestamps are properly formatted
* **MPI errors**: Check that all processes have consistent inputs

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Maintainers

limTOD is developed and maintained by members of MeerKLASS collaboration, which currently include:

* Zheng Zhang (University of Manchester)
* Piyanat Kittiwisit (University of the Western Cape)
