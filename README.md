# limTOD: Time-ordered Data Simulator for Single-dish Line Intensity Mapping Observation

## Overview

**limTOD** is a Python package for simulating Time-Ordered Data (TOD) from single-dish radio intensity mapping observations. It assumes abritory sky model and beam model, including asymmetric beam, and perform direct convolution to calculate the TOD. The sky and beam models must be passed as Callable functions. A wrapper function of PyGDSM and a function to generate generic Gaussian elliptical beam are provided. The A generic 1/f noise model is also included. Other contributation to the TOD can be passed as arrays. Although symmetric beam is supported, the calculation could be unnecessarily slow due to the use of direct convolution as oppose to using a healpy smoothing function.

📖 **For detailed mathematical conventions and coordinate system definitions, see [conventions.pdf](conventions.pdf).**

### Input Parameters

#### Telescope Configuration

* **ant_latitude_deg** (`float`): Latitude of the antenna/site in degrees.
* **ant_longitude_deg** (`float`): Longitude of the antenna/site in degrees.
* **ant_height_m** (`float`): Height of the antenna/site in meters.
* **beam_func** (`function`): Function that takes frequency and nside as input and returns the beam map.
* **sky_func** (`function`): Function that takes frequency and nside as input and returns the sky map.
* **nside** (`int`, optional): The nside parameter for Healpix maps.

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
* **gain_noise_params** (`list`, optional): List of parameters [f0, fc, alpha] for generating gain noise if gain_noise_TOD is None. Default is [1.4e-5, 1e-3, 2].
* **white_noise_var** (`float`, optional): Variance of white noise to be added. Default is None (uses default value of 2.5e-6).

### Output Parameters

* **overall_TOD** (`ndarray`): Complete TOD with all components (nfreq × ntime)
* **sky_TOD** (`ndarray`): Sky signal component only (beam-weighted sum of sky maps, nfreq × ntime)
* **gain_noise_TOD** (`ndarray`): Gain noise component (nfreq × ntime)

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Theoretical Background](#theoretical-background)
4. [Mathematical Conventions](#mathematical-conventions)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Performance Considerations](#performance-considerations)

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
pip install .
```

## Quick Start

```python
import numpy as np
from limTOD import TODSim, example_scan, example_beam_map, GDSM_sky_model

# Initialize the simulator with MeerKAT coordinates
simulator = TODsim(
    ant_latitude_deg=-30.7130,  # MeerKAT latitude
    ant_longitude_deg=21.4430,  # MeerKAT longitude
    ant_height_m=1054,          # MeerKAT altitude
    nside=64,                   # HEALPix resolution
    beam_func=example_beam_map, # Return HEALPix beam array of shape (hp.nside2npix(nside))
    sky_func=GDSM_sky_model     # Return HEALPix sky map of shape (hp.nside2npix(nside))
)

# Generate a simple scanning pattern
time_list, azimuth_list = example_scan()

# Simulate TOD for multiple frequencies
freq_list = [950, 1000, 1050]  # MHz
tod_array, sky_tod, gain_noise = simulator.generate_TOD(
    freq_list=freq_list,
    time_list=time_list[:100],  # Simulate only the first 100 times
    azimuth_deg_list=azimuth_list,
    elevation_deg=41.5,
)

print(f"Generated TOD shape: {tod_array.shape}")  # (3, n_time)
```

## Theoretical Background

### Time-Ordered Data (TOD) Model

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

### Sky Signal Computation

The sky signal is computed as the convolution of the beam pattern with the sky brightness temperature:

```
sky_TOD(ν,t) = ∫ B(θ,φ,ν,t) × T_sky(θ,φ,ν) dΩ
```

This is efficiently computed using HEALPix spherical harmonics:

1. Convert beam map to spherical harmonic coefficients: `B_lm = map2alm(B)`
2. Rotate beam to pointing direction using Euler angles
3. Revert the roated beam back to map domain
4. Compute beam-weighted sum with sky map

### Coordinate Transformations

The package handles coordinate transformations between:

1. **Local Telescope frame** (time, Azimuth, Elevation) → **Equatorial frame** (RA, Dec)
2. Representations of the transformation: **ZYZY Euler angles** → **ZYZ Euler angles** for HEALPix rotations

#### Detailed Workflow

**Step 1: Scan Specifications → LST Sequence**

* Convert UTC timestamps to Local Sidereal Time using telescope location
* Function: `generate_LSTs_deg()`

**Step 2: Telescope Pointing → ZYZY Angles**  

* Map telescope parameters to natural rotation sequence:
  + α = LST (Earth's rotation tracking)
  + β = 90° - latitude (site location correction)
  + γ = azimuth (local pointing direction)
  + δ = 90° - elevation (altitude correction)

**Step 3: ZYZY → ZYZ Conversion**

* Convert to HEALPix-compatible Euler angles using `zyzy2zyz()`
* This handles the mathematical transformation: R_zyzy = R_y(δ)R_z(γ)R_y(β)R_z(α) → R_zyz = R_z(φ)R_y(θ)R_z(ψ)

**Step 4: Beam Rotation in Spherical Harmonic Space**

* Apply rotation to beam's alm coefficients using `pointing_beam_in_eq_sys()`
* Efficiently rotates beam pattern without pixel-by-pixel calculations
* Function: `_rotate_healpix_map()` calls `healpy.rotate_alm()`

**Step 5: Sky Integration**

* Compute beam-weighted sum: SKY_TOD_SAMPLE = ∫ B_pointed(θ, φ) × T_sky(θ, φ) dΩ
* Function: `_beam_weighted_sum()`

This approach allows accurate simulation of how the telescope beam tracks celestial sources as the Earth rotates and the telescope points to different directions. However, note that accuracy of the TOD will strongly depend on the structure of the sky components and the resolution of the simulation (the `nside` parameter).

## Mathematical Conventions

For detailed mathematical formulations, coordinate system definitions, and algorithmic conventions used in this package, please refer to:

**[Mathematical Conventions Document](conventions.pdf)**

This document contains:

* Coordinate system definitions and transformations
* Euler angle conventions (ZYZY ↔ ZYZ)
* Spherical harmonics formulations
* Beam convolution algorithms
* Noise model specifications

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

* `ant_latitude_deg` (float): Antenna latitude in degrees. [Default to MeerKAT]
* `ant_longitude_deg` (float): Antenna longitude in degrees. [Default to MeerKAT]
* `ant_height_m` (float): Antenna height above sea level in meters. [Default to MeerKAT]
* `beam_func` (callable): Function returning beam map given (freq, nside)
* `sky_func` (callable): Function returning sky map given (freq, nside)
* `nside` (int): HEALPix resolution parameter (must be power of 2)

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

**Returns:**

* `overall_TOD` (ndarray): Complete TOD with all components (nfreq × ntime)
* `sky_TOD` (ndarray): Sky signal component only (beam-weighted sum of sky maps, no gain and no noise. Shape: nfreq × ntime)
* `gain_noise_TOD` (ndarray): Gain noise component (nfreq × ntime)

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

* `sky_TOD` (ndarray): Sky signal TOD (nfreq × ntime)

### Utility Functions

#### `example_scan()`

Generate a simple raster scanning pattern.

```python
def example_scan(az_s=-60.3, az_e=-42.3, dt=2.0)
```

**Parameters:**

* `az_s` (float): Starting azimuth in degrees
* `az_e` (float): Ending azimuth in degrees  
* `dt` (float): Time step in seconds

**Returns:**

* `time_list` (ndarray): Time offsets in seconds
* `azimuth_list` (ndarray): Azimuth angles in degrees

#### `generate_LSTs_deg()`

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
   - α = LST (Earth rotation)
   - β = 90° - lat (latitude correction)
   - γ = azimuth (local pointing direction)
   - δ = 90° - elevation (elevation correction)
2. Transform to ZYZ using `zyzy2zyz()`

This maps the natural telescope coordinate system to the mathematical framework required for spherical harmonic rotations.
def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg)

##### `pointing_beam_in_eq_sys()`

Point a beam pattern to specific telescope coordinates in the equatorial system.

```python
def pointing_beam_in_eq_sys(beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside)
```

**Parameters:**

* `beam_alm` (array): Spherical harmonic coefficients of the beam in its native orientation
* `LST_deg` (float): Local Sidereal Time in degrees  
* `lat_deg` (float): Latitude of the observation site in degrees
* `azimuth_deg` (float): Azimuth of the pointing in degrees
* `elevation_deg` (float): Elevation of the pointing in degrees
* `nside` (int): HEALPix resolution parameter

**Returns:**

* `beam_pointed` (ndarray): The rotated beam map in equatorial coordinates

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

* `alm` (array): Spherical harmonic coefficients of the map
* `psi_rad`,   `theta_rad`,  `phi_rad` (float): ZYZ Euler angles in radians
* `nside` (int): HEALPix resolution parameter
* `return_map` (bool): If True, return rotated map; if False, return rotated alm

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

* `beam_map` (array): HEALPix beam pattern (will be normalized)
* `sky_map` (array): HEALPix sky brightness temperature map

**Returns:**

* `float`: Beam-weighted integral ∫ B(θ, φ) × T_sky(θ, φ) dΩ

This implements the discrete version of the beam convolution integral that produces each TOD sample.

#### Beam and Sky Functions

##### `example_beam_map()`

Generate elliptical Gaussian beam pattern.

```python
def example_beam_map(freq, nside, FWHM_major=1.1, FWHM_minor=1.1)
```

##### `GDSM_sky_model()`

Generate sky map using Global Sky Model.

```python
def GDSM_sky_model(freq, nside)
```

## Examples

See the [example Jupyter notebook](examples.ipynb)

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
