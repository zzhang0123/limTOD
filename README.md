# limTOD: Time-Ordered Data Simulator for single-dish (autocorrelation) line intensity mapping measurements

**limTOD** is a Python package for simulating Time-Ordered Data (TOD) from single-dish/autocorrelation observations using asymetric beam. Although it also supports a symmetric beam, it could be unnecessarily slow compared to directly convolving the sky with the healpy smoothing function.

� **For the latest updates and release notes, see [CHANGELOG.md](CHANGELOG.md)**

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Theoretical Background](#theoretical-background)
4. [Mathematical Conventions](#mathematical-conventions)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Performance Considerations](#performance-considerations)



## Citation

If you use limTOD in your research, please cite:

```bibtex
@software{zheng_zhang_2025_17159124,
  author       = {Zheng Zhang},
  title        = {zzhang0123/limTOD: limTOD: Time-Ordered Data
                   Simulator for line intensity mapping experiments
                  },
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.17159124},
  url          = {https://doi.org/10.5281/zenodo.17159124},
  swhid        = {swh:1:dir:6ce550d9495bcca3c7ad611986cdcfbdcf46a2de
                   ;origin=https://doi.org/10.5281/zenodo.17159123;vi
                   sit=swh:1:snp:c55636bc6c217da228dfcef2497edd037de3
                   fb30;anchor=swh:1:rel:d71baa2ce452220ee21a2ca8b114
                   ca9d3b280b86;path=zzhang0123-limTOD-c2c3492
                  },
}
```



## Installation

### Requirements

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

### Install from source

```bash
git clone https://github.com/zzhang0123/limTOD.git
cd limTOD
pip install -e .
```

## Quick Start

```python
import numpy as np
from tod_simulator import TODSim, example_scan

# Initialize the simulator with MeerKAT coordinates
simulator = TODSim(
    ant_latitude_deg=-30.7130,   # MeerKAT latitude
    ant_longitude_deg=21.4430,   # MeerKAT longitude
    ant_height_m=1054,           # MeerKAT altitude
    nside=256                    # HEALPix resolution
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

- **For detailed coordinate system definitions, see [conventions.pdf](conventions.pdf).**
- **For working examples of TOD simulation, see [test/TODsim_examples.ipynb](test/TODsim_examples.ipynb)**
- **For working examples of HighPass+Wiener mapmaking, see map-making workflow ([test/mm_example.ipynb](test/mm_example.ipynb))**


### Input Parameters:

#### Telescope Configuration:
- **ant_latitude_deg** (`float`): Latitude of the antenna/site in degrees.
- **ant_longitude_deg** (`float`): Longitude of the antenna/site in degrees.
- **ant_height_m** (`float`): Height of the antenna/site in meters.
- **beam_func** (`function`): Function that takes _keyword-only_ inputs, two of which must be `freq` (for frequency) and `nside` and returns the HEALPix beam map of shape (npix,). Optional keywords can be passed to the function for customisation.
- **sky_func** (`function`): Function that takes _keyword-only_ inputs, two of which must be `freq` (for frequency) and `nside` and returns the HEALPix sky map of shape (npix,). Optional keywords can be passed to the function for customisation.
- **nside** (`int`, optional): The nside parameter for Healpix maps.

#### Observation Parameters:
- **freq_list** (`list` or `array`): List of frequencies.
- **time_list** (`list` or `array`): List of time offsets in seconds (for each observation time) from start_time_utc.
- **azimuth_deg_list** (`list` or `array`): List of azimuth values in degrees for each observation.
- **elevation_deg** (`list` or `float`): 
  - If float: Elevation value in degrees universal for all observations.
  - If list: List of elevation values in degrees for each observation.
- **start_time_utc** (`str`): Start time in UTC (e.g. "2019-04-23 20:41:56.397").

#### Noise and Calibration Parameters (Optional):
- **Tsys_others_TOD** (`array`, optional): Array of the remaining system temperature TOD (shape: nfreq x ntime). Default is None (no other components).
- **background_gain_TOD** (`array`, optional): Array of background gain TOD (shape: nfreq x ntime). Default is None (unity gain).
- **gain_noise_TOD** (`array`, optional): Array of gain noise TOD (shape: nfreq x ntime). Default is None (no gain noise).
- **gain_noise_params** (`list`, optional): List of parameters [f0, fc, alpha] for generating gain noise if gain_noise_TOD is None. Default is [1.4e-5, 1e-3, 2].
- **white_noise_var** (`float`, optional): Variance of white noise to be added. Default is None (uses default value of 2.5e-6).

### Output Parameters:
- **overall_TOD** (`ndarray`): Complete TOD with all components (nfreq × ntime)
- **sky_TOD** (`ndarray`): Sky signal component only (beam-weighted sum of sky maps, nfreq × ntime)
- **gain_noise_TOD** (`ndarray`): Gain noise component (nfreq × ntime)






## Theoretical Background

### Time-Ordered Data (TOD) Model

The complete TOD model implemented in this package follows the equation:

```
TOD(ν,t) = G_bg(ν,t) × [1 + G_noise(ν,t)] × [sky_TOD(ν,t) + Tsys_others(ν,t)] × [1 + η(t)]
```

Where:
- `G_bg(ν,t)`: Background gain pattern
- `G_noise(ν,t)`: Gain noise fluctuations (1/f noise)
- `sky_TOD(ν,t)`: Sky signal convolved with beam
- `Tsys_others(ν,t)`: All the other system temperature components
- `η(t)`: White noise component

### Sky Signal Computation

The sky signal is computed as the convolution of the beam pattern with the sky brightness temperature:

```
sky_TOD(ν,t) = ∫ B(θ,φ,ν,t) × T_sky(θ,φ,ν) dΩ
```

This is efficiently computed using HEALPix spherical harmonics:

1. Convert beam map to spherical harmonic coefficients: `B_lm = map2alm(B)`
2. Rotate beam to pointing direction using Euler angles
3. Compute beam-weighted sum with sky map

### Coordinate Transformations

The package handles coordinate transformations between:

1. **Local Telescope frame** (time, Azimuth, Elevation) → **Equatorial frame** (RA, Dec)
2. Representations of the transformation: **ZYZY Euler angles** → **ZYZ Euler angles** for HEALPix rotations

#### Detailed Workflow:

**Step 1: Scan Specifications → LST Sequence**
- Convert UTC timestamps to Local Sidereal Time using telescope location
- Function: `generate_LSTs_deg()`

**Step 2: Telescope Pointing → ZYZY Angles**  
- Map telescope parameters to natural rotation sequence:
  - α = LST (Earth's rotation tracking)
  - β = 90° - latitude (site location correction)
  - γ = azimuth (local pointing direction)
  - δ = elevation - 90°  (altitude correction)

**Step 3: ZYZY → ZYZ Conversion**
- Convert to HEALPix-compatible Euler angles using `zyzy2zyz()`
- This handles the mathematical transformation: R_zyzy = R_y(δ)R_z(γ)R_y(β)R_z(α) → R_zyz = R_z(φ)R_y(θ)R_z(ψ)

**Step 4: Beam Rotation in Spherical Harmonic Space**
- Apply rotation to beam's alm coefficients using `pointing_beam_in_eq_sys()`
- Efficiently rotates beam pattern without pixel-by-pixel calculations
- Function: `_rotate_healpix_map()` calls `healpy.rotate_alm()`

**Step 5: Sky Integration**
- Compute beam-weighted sum: SKY_TOD_SAMPLE = ∫ B_pointed(θ,φ) × T_sky(θ,φ) dΩ
- Function: `_beam_weighted_sum()`

This approach allows accurate simulation of how the telescope beam tracks celestial sources as the Earth rotates and the telescope points to different directions.

### Mathematical Conventions

For detailed mathematical formulations, coordinate system definitions, and algorithmic conventions used in this package, please refer to:

** [Mathematical Conventions Document](conventions.pdf)**

This document contains:
- Coordinate system definitions and transformations
- Euler angle conventions (ZYZY ↔ ZYZ)
- Spherical harmonics formulations
- Beam convolution algorithms
- Noise model specifications

## API Reference

### Core Classes

#### `TODSim`

Main simulator class for generating time-ordered data.

```python
class TODSim:
    def __init__(self, 
                 ant_latitude_deg=-30.7130,
                 ant_longitude_deg=21.4430, 
                 ant_height_m=1054,
                 beam_func=example_beam_map,
                 sky_func=GDSM_sky_model,
                 nside=256)
```

**Parameters:**
- `ant_latitude_deg` (float): Antenna latitude in degrees
- `ant_longitude_deg` (float): Antenna longitude in degrees  
- `ant_height_m` (float): Antenna height above sea level in meters
- `beam_func` (callable): Function returning beam map given (freq, nside) as keyword arguments
- `sky_func` (callable): Function returning sky map given (freq, nside) as keyword arguments
- `nside` (int): HEALPix resolution parameter (must be power of 2)

### Core Functions

#### `generate_TOD()`

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
                gain_noise_params=[1.4e-5, 1e-3, 2],
                white_noise_var=None)
```

**Parameters:**
- `freq_list` (array_like): Observation frequencies in MHz
- `time_list` (array_like): Time offsets from start time in seconds
- `azimuth_deg_list` (array_like): Time-ordered azimuth angles in degrees
- `elevation_deg` (float or array_like): Elevation angle(s) in degrees
- `start_time_utc` (str): UTC start time in ISO format
- `Tsys_others_TOD` (array_like, optional): All the other system temperature components
- `background_gain_TOD` (array_like, optional): Background gain variations
- `gain_noise_TOD` (array_like, optional): Pre-computed gain noise
- `gain_noise_params` (list): [f0, fc, alpha] for 1/f noise generation, if gain_noise_TOD is not provided
- `white_noise_var` (float, optional): White noise variance

**Returns:**
- `overall_TOD` (ndarray): Complete TOD with all components (nfreq × ntime)
- `sky_TOD` (ndarray): Sky signal component only (beam-weighted sum of sky maps, no gain and no noise. Shape: nfreq × ntime)
- `gain_noise_TOD` (ndarray): Gain noise component (nfreq × ntime)

#### `simulate_sky_TOD()`

Generate sky signal component of TOD.

```python
def simulate_sky_TOD(self,
                    freq_list,
                    time_list,
                    azimuth_deg_list,
                    elevation_deg,
                    start_time_utc="2019-04-23 20:41:56.397")
```

**Returns:**
- `sky_TOD` (ndarray): Sky signal TOD (nfreq × ntime)

### Utility Functions

#### `example_scan()`

Generate a simple raster scanning pattern.

```python
def example_scan(az_s=-60.3, az_e=-42.3, dt=2.0)
```

**Parameters:**
- `az_s` (float): Starting azimuth in degrees
- `az_e` (float): Ending azimuth in degrees  
- `dt` (float): Time step in seconds

**Returns:**
- `time_list` (ndarray): Time offsets in seconds
- `azimuth_list` (ndarray): Azimuth angles in degrees

#### `generate_LSTs_deg()`

Compute Local Sidereal Time for observation times.

```python
def generate_LSTs_deg(ant_latitude_deg, ant_longitude_deg, ant_height_m,
                     time_list, start_time_utc="2019-04-23 20:41:56.397")
```

**Returns:**
- `LST_deg_list` (ndarray): LST values in degrees

#### Coordinate Transformation Functions

##### `zyzy2zyz()`

Convert ZYZY Euler angles to ZYZ convention for HEALPix rotations.

```python
def zyzy2zyz(alpha, beta, gamma, delta, output_degrees=False)
```

**Parameters:**
- `alpha` (float): First Z rotation angle in degrees
- `beta` (float): First Y rotation angle in degrees  
- `gamma` (float): Second Z rotation angle in degrees
- `delta` (float): Second Y rotation angle in degrees
- `output_degrees` (bool): If True, return angles in degrees; otherwise radians

**Returns:**
- `(psi, theta, phi)` (tuple): ZYZ Euler angles for HEALPix rotation

**Mathematical Background:**
- **ZYZY rotation**: R = R_y(δ) × R_z(γ) × R_y(β) × R_z(α)
- **ZYZ rotation**: R = R_z(φ) × R_y(θ) × R_z(ψ)

This conversion is necessary because telescope pointing naturally follows ZYZY rotations (combining Earth rotation and local pointing), while HEALPix requires ZYZ convention.

##### `zyz_of_pointing()`

Generate ZYZ Euler angles from telescope pointing parameters.

```python
def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg)
```

**Parameters:**
- `LST_deg` (float): Local Sidereal Time in degrees
- `lat_deg` (float): Telescope latitude in degrees
- `azimuth_deg` (float): Pointing azimuth in degrees
- `elevation_deg` (float): Pointing elevation in degrees

**Returns:**
- `(psi, theta, phi)` (tuple): ZYZ Euler angles in radians for `hp.rotate_alm()`

**Algorithm:**
1. Convert pointing parameters to ZYZY angles:
   - α = LST (Earth rotation)
   - β = 90° - lat (latitude correction)
   - γ = azimuth (local pointing direction)
   - δ = 90° - elevation (elevation correction)
2. Transform to ZYZ using `zyzy2zyz()`

This maps the natural telescope coordinate system to the mathematical framework required for spherical harmonic rotations.
def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg)
```

##### `pointing_beam_in_eq_sys()`

Point a beam pattern to specific telescope coordinates in the equatorial system.

```python
def pointing_beam_in_eq_sys(beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside)
```

**Parameters:**
- `beam_alm` (array): Spherical harmonic coefficients of the beam in its native orientation
- `LST_deg` (float): Local Sidereal Time in degrees  
- `lat_deg` (float): Latitude of the observation site in degrees
- `azimuth_deg` (float): Azimuth of the pointing in degrees
- `elevation_deg` (float): Elevation of the pointing in degrees
- `nside` (int): HEALPix resolution parameter

**Returns:**
- `beam_pointed` (ndarray): The rotated beam map in equatorial coordinates

**Algorithm:**
1. Convert telescope pointing (LST, lat, az, el) to ZYZ Euler angles using `zyz_of_pointing()`
2. Rotate the beam's spherical harmonic coefficients using `_rotate_healpix_map()`
3. Return the pointed beam map in equatorial coordinate system

This function is central to the TOD simulation as it enables the beam pattern to track celestial sources as the Earth rotates.

#### Internal Helper Functions

##### `_rotate_healpix_map()`

Rotate a HEALPix map using Euler angles in spherical harmonic space.

```python
def _rotate_healpix_map(alm, psi_rad, theta_rad, phi_rad, nside, return_map=True)
```

**Parameters:**
- `alm` (array): Spherical harmonic coefficients of the map
- `psi_rad`, `theta_rad`, `phi_rad` (float): ZYZ Euler angles in radians
- `nside` (int): HEALPix resolution parameter
- `return_map` (bool): If True, return rotated map; if False, return rotated alm

**Algorithm:**
1. Creates a copy of input spherical harmonic coefficients
2. Applies rotation using `healpy.rotate_alm()` with ZYZ convention
3. Converts back to map format if requested

##### `_beam_weighted_sum()`

Compute the convolution integral of beam and sky.

```python
def _beam_weighted_sum(beam_map, sky_map)
```

**Parameters:**
- `beam_map` (array): HEALPix beam pattern (will be normalized)
- `sky_map` (array): HEALPix sky brightness temperature map

**Returns:**
- `float`: Beam-weighted integral ∫ B(θ,φ) × T_sky(θ,φ) dΩ

This implements the discrete version of the beam convolution integral that produces each TOD sample.

#### Beam and Sky Functions

##### `example_beam_map()`

Generate elliptical Gaussian beam pattern.

```python
def example_beam_map(*, freq, nside, FWHM_major=1.1, FWHM_minor=1.1)
```

##### `GDSM_sky_model()`

Generate sky map using Global Sky Model.

```python
def GDSM_sky_model(*, freq, nside)
```

## Map-Making with HPW_mapmaking

### Overview

The `HPW_mapmaking` class provides a sophisticated map-making pipeline that combines high-pass filtering and Wiener filtering to reconstruct sky maps from Time-Ordered Data (TOD). This implementation handles 1/f noise through high-pass filtering while optimally recovering sky signals using Wiener filtering with optional priors.

### Theoretical Background

The map-making process follows these key steps:

1. **High-Pass Filtering**: Remove low-frequency drifts and 1/f noise using a Butterworth filter
2. **Forward Modeling**: Build an operator that maps sky parameters to TOD samples
3. **Wiener Filtering**: Solve the inverse problem optimally:
   ```
   x̂ = (A^T N^{-1} A + S^{-1})^{-1} (A^T N^{-1} d + S^{-1} μ)
   ```
   where:
   - `A`: System operator (beam convolution + instrumental effects)
   - `N`: Noise covariance matrix
   - `S`: Signal prior covariance matrix
   - `d`: Measured TOD data
   - `μ`: Prior mean for sky parameters

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
                 Tsys_others_operator=None)
```

**Parameters** (all keyword-only):
- `beam_map` (array): HEALPix beam pattern. Can be:
  - 1D array (length npix) for intensity-only (I)
  - 2D array (3 × npix) for polarization (I, Q, U)
  - 2D array (4 × npix) for full Stokes (I, Q, U, V)
- `LST_deg_list_group` (list): LST values in degrees for each TOD or list of LST lists
- `lat_deg` (float): Observation site latitude in degrees
- `azimuth_deg_list_group` (list): Azimuth angles in degrees for each TOD
- `elevation_deg_list_group` (list): Elevation angles in degrees for each TOD
- `threshold` (float): Fractional beam response threshold (e.g., 0.01 = 1% of peak)
- `Tsys_others_operator` (array, optional): Operator for other system components (e.g., receiver temperature variations)

#### Calling the Map-Maker

```python
sky_map, sky_uncertainty = mapmaker(
    TOD_group,
    dtime,
    cutoff_freq_group,
    gain_group=None,
    known_injection_group=None,
    Tsky_prior_mean=None,
    Tsky_prior_inv_cov_diag=None,
    Tsys_other_prior_mean_group=None,
    Tsys_other_prior_inv_cov_group=None,
    regularization=1e-12,
    return_full_cov=False
)
```

**Parameters:**
- `TOD_group` (list): List of TOD arrays, one per observation
- `dtime` (float): Time sampling interval in seconds
- `cutoff_freq_group` (list): High-pass filter cutoff frequencies in Hz
- `gain_group` (list, optional): Gain calibration factors for each TOD
- `known_injection_group` (list, optional): Known signals to subtract (e.g., calibration diodes)
- `Tsky_prior_mean` (array, optional): Prior mean for sky temperature
- `Tsky_prior_inv_cov_diag` (array, optional): Diagonal of prior inverse covariance matrix
- `Tsys_other_prior_mean_group` (list, optional): Prior means for other system parameters
- `Tsys_other_prior_inv_cov_group` (list, optional): Prior inverse covariances for other system parameters
- `regularization` (float): Regularization parameter for numerical stability
- `return_full_cov` (bool): If True, return full posterior covariance matrix

**Returns:**
- `sky_map` (array): Reconstructed sky map(s) in temperature units
- `sky_uncertainty` (array): Per-pixel uncertainty estimates
- `Tsys_others_estimation_group` (list, optional): Estimated other system parameters
- `Tsys_others_uncertainty_group` (list, optional): Uncertainties for other system parameters

### Map-Making Examples

📓 **For a complete working example, see [test/mm_example.ipynb](test/mm_example.ipynb)**

This notebook demonstrates:
- Simulating multiple TOD sets at different elevations
- Initializing the `HPW_mapmaking` class with keyword arguments
- Performing map-making with high-pass + wiener filtering
- Visualizing reconstructed sky maps using `gnomview_patch`


#### Example 1: Basic Map-Making from Simulated TODs

```python
import numpy as np
from limTOD import TODSim, HPW_mapmaking, example_beam_map, GDSM_sky_model

# 1. Simulate TODs (as in previous examples)
simulator = TODSim(
    ant_latitude_deg=-30.7130,  # MeerKAT
    ant_longitude_deg=21.4430,
    ant_height_m=1054,
    nside=64,
    beam_func=example_beam_map,
    sky_func=GDSM_sky_model
)

# Generate multiple scans
TOD_group = []
LST_deg_list_group = []
azimuth_deg_list_group = []
elevation_deg_list_group = []

for elevation in [38.5, 41.5, 44.5]:  # Multiple elevations
    time_list, azimuth_list = example_scan(dt=2.0, n_repeats=7)
    
    tod, _, _, LST_list = simulator.generate_TOD(
        freq_list=[950],  # MHz
        time_list=time_list,
        azimuth_deg_list=azimuth_list,
        elevation_deg=elevation,
        start_time_utc="2019-04-23 20:41:56.397",
        return_LSTs=True
    )
    
    TOD_group.append(tod[0])
    LST_deg_list_group.append(LST_list)
    azimuth_deg_list_group.append(azimuth_list)
    elevation_deg_list_group.append(elevation * np.ones_like(tod[0]))

# 2. Initialize map-maker
beam_map = example_beam_map(freq=950, nside=64)

mapmaker = HPW_mapmaking(
    beam_map=beam_map,
    LST_deg_list_group=LST_deg_list_group,
    lat_deg=simulator.ant_latitude_deg,
    azimuth_deg_list_group=azimuth_deg_list_group,
    elevation_deg_list_group=elevation_deg_list_group,
    threshold=0.05  # Only use pixels with >5% beam response
)

# 3. Perform map-making
cutoff_freqs = [0.001] * len(TOD_group)  # 0.001 Hz high-pass filter

sky_map, sky_unc = mapmaker(
    TOD_group=TOD_group,
    dtime=2.0,  # seconds
    cutoff_freq_group=cutoff_freqs
)

print(f"Reconstructed {len(sky_map)} sky pixels")
print(f"Mean temperature: {np.mean(sky_map):.2f} K")
print(f"Mean uncertainty: {np.mean(sky_unc):.2f} K")
```

#### Example 2: Map-Making with Priors

```python
# Use a previously reconstructed map as a prior
prior_map = ... # e.g., from previous observation or simulation

# Set prior with moderate confidence (inverse variance)
prior_inv_variance = 1.0 / (10.0**2)  # σ_prior = 10 K

sky_map_with_prior, sky_unc_with_prior = mapmaker(
    TOD_group=TOD_group,
    dtime=2.0,
    cutoff_freq_group=cutoff_freqs,
    Tsky_prior_mean=prior_map,
    Tsky_prior_inv_cov_diag=prior_inv_variance * np.ones_like(prior_map)
)
```

#### Example 3: Joint Estimation of Sky and Systematics

```python
# Build operator for receiver temperature variations
# E.g., using Legendre polynomials
from numpy.polynomial.legendre import legval

n_time = len(TOD_group[0])
n_legendre = 5
t_normalized = np.linspace(-1, 1, n_time)

Trec_operator = np.zeros((n_time, n_legendre))
for i in range(n_legendre):
    coeffs = np.zeros(n_legendre)
    coeffs[i] = 1.0
    Trec_operator[:, i] = legval(t_normalized, coeffs)

# Initialize with systematics operator
mapmaker_with_sys = HPW_mapmaking(
    beam_map=beam_map,
    LST_deg_list_group=LST_deg_list_group,
    lat_deg=simulator.ant_latitude_deg,
    azimuth_deg_list_group=azimuth_deg_list_group,
    elevation_deg_list_group=elevation_deg_list_group,
    threshold=0.05,
    Tsys_others_operator=Trec_operator  # Add systematics operator
)

# Prior for receiver temperature coefficients
Trec_prior_mean = [300.0, 0.0, 0.0, 0.0, 0.0]  # ~300 K baseline
Trec_prior_inv_cov = np.diag([1e-6, 1e-2, 1e-2, 1e-2, 1e-2])  # Tight on mean, loose on variations

# Perform joint estimation
sky_map, sky_unc, Trec_est, Trec_unc = mapmaker_with_sys(
    TOD_group=TOD_group,
    dtime=2.0,
    cutoff_freq_group=cutoff_freqs,
    Tsys_other_prior_mean_group=[Trec_prior_mean] * len(TOD_group),
    Tsys_other_prior_inv_cov_group=[Trec_prior_inv_cov] * len(TOD_group)
)

print("Receiver temperature coefficients:")
for i, (est, unc) in enumerate(zip(Trec_est[0], Trec_unc[0])):
    print(f"  Coeff {i}: {est:.3f} ± {unc:.3f}")
```




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

## Error Handling and Validation

Common error scenarios and solutions:

* **Memory errors**: Reduce `nside` or process in smaller batches
* **Coordinate errors**: Check that Az/El values are within valid ranges
* **Time errors**: Ensure UTC timestamps are properly formatted
* **MPI errors**: Check that all processes have consistent inputs

### Development Setup

```bash
git clone https://github.com/zzhang0123/limTOD.git
cd limTOD
pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Maintainers

limTOD is developed and maintained by members of MeerKLASS collaboration, which currently include:
* Zheng Zhang (University of Manchester)
* Piyanat Kittiwisit (University of the Western Cape)
