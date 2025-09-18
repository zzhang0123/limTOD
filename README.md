# MeerTOD: Time-Ordered Data Simulator for MeerKLASS single-dish measurements

## Overview

**MeerTOD** is a Python package for simulating Time-Ordered Data (TOD) from MeerKLASS observations using asymetric beam. Although it also supports a symmetric beam, it could be unnecessarily slow compared to directly convolving the sky with the healpy smoothing function.

- Simulating radio telescope observations with realistic beam patterns
- Generating time-ordered data from sky maps using HEALPix projections
- Incorporating various noise models including 1/f gain fluctuations
- Coordinate transformations between telescope and celestial reference frames
- Parallel processing capabilities for large-scale simulations

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Theoretical Background](#theoretical-background)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Performance Considerations](#performance-considerations)
8. [Contributing](#contributing)

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
git clone https://github.com/zzhang0123/meerTOD.git
cd meerTOD
pip install -e .
```

## Quick Start

```python
import numpy as np
from tod_simulator import meerTODsim, example_scan

# Initialize the simulator with MeerKAT coordinates
simulator = meerTODsim(
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

## Theoretical Background

### Time-Ordered Data (TOD) Model

The complete TOD model implemented in this package follows the equation:

```
TOD(ν,t) = G_bg(ν,t) × [1 + G_noise(ν,t)] × [S_sky(ν,t) + T_sys(ν,t)] × [1 + η(t)]
```

Where:
- `G_bg(ν,t)`: Background gain pattern
- `G_noise(ν,t)`: Gain noise fluctuations (1/f noise)
- `S_sky(ν,t)`: Sky signal convolved with beam
- `T_sys(ν,t)`: Residual system temperature
- `η(t)`: White noise component

### Sky Signal Computation

The sky signal is computed as the convolution of the beam pattern with the sky brightness temperature:

```
S_sky(ν,t) = ∫ B(θ,φ,ν,t) × T_sky(θ,φ,ν) dΩ
```

This is efficiently computed using HEALPix spherical harmonics:

1. Convert beam map to spherical harmonic coefficients: `B_lm = map2alm(B)`
2. Rotate beam to pointing direction using Euler angles
3. Compute beam-weighted sum with sky map

### Coordinate Transformations

The package handles coordinate transformations between:

1. **Local Telescope frame** (time, Azimuth, Elevation) → **Equatorial frame** (RA, Dec)
2. Representations of the transformation: **ZYZY Euler angles** → **ZYZ Euler angles** for HEALPix rotations

The transformation follows the sequence:
- Scan specifications → LST sequence
- LST/Azimuth/Elevation → "zyzy" Euler angle representation
-  "zyzy" representation → effective "zyz" Euler angles for spherical harmonic rotation
- Apply rotation to beam pattern in spherical harmonic space

## API Reference

### Core Classes

#### `meerTODsim`

Main simulator class for generating time-ordered data.

```python
class meerTODsim:
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
- `beam_func` (callable): Function returning beam map given (freq, nside)
- `sky_func` (callable): Function returning sky map given (freq, nside)
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
                residual_Tsys_TOD=None,
                background_gain_TOD=None,
                gain_noise_TOD=None,
                gain_noise_params=[1.4e-5, 1e-3, 2],
                white_noise_var=None)
```

**Parameters:**
- `freq_list` (array_like): Observation frequencies in MHz
- `time_list` (array_like): Time offsets from start time in seconds
- `azimuth_deg_list` (array_like): Azimuth angles in degrees
- `elevation_deg` (float or array_like): Elevation angle(s) in degrees
- `start_time_utc` (str): UTC start time in ISO format
- `residual_Tsys_TOD` (array_like, optional): System temperature residuals
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

Convert ZYZY Euler angles to ZYZ convention for HEALPix.

```python
def zyzy2zyz(alpha, beta, gamma, delta, output_degrees=False)
```

##### `zyz_of_pointing()`

Generate ZYZ Euler angles from telescope pointing parameters.

```python
def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg)
```

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

### Example 1: Basic TOD Simulation

```python
import numpy as np
from tod_simulator import meerTODsim, example_scan
import matplotlib.pyplot as plt

# Initialize simulator
sim = meerTODsim(nside=128)  # Lower resolution for speed

# Generate scanning pattern
time_list, az_list = example_scan(dt=1.0)

# Single frequency simulation
tod, sky_tod, gain_noise = sim.generate_TOD(
    freq_list=[1000],  # 1 GHz
    time_list=time_list[:100],  # First 100 time points
    azimuth_deg_list=az_list[:100],
    elevation_deg=45.0
)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

ax1.plot(time_list[:100], tod[0])
ax1.set_ylabel('Total TOD [K]')
ax1.set_title('Complete TOD with all components')

ax2.plot(time_list[:100], sky_tod[0])
ax2.set_ylabel('Sky Signal [K]')
ax2.set_title('Sky component only')

ax3.plot(time_list[:100], gain_noise[0])
ax3.set_ylabel('Gain Noise')
ax3.set_xlabel('Time [s]')
ax3.set_title('Gain noise component')

plt.tight_layout()
plt.show()
```

### Example 2: Multi-frequency Simulation

```python
# Wide frequency range
frequencies = np.linspace(900, 1100, 21)  # 21 channels

tod_multifreq, sky_multifreq, _ = sim.generate_TOD(
    freq_list=frequencies,
    time_list=time_list[:50],
    azimuth_deg_list=az_list[:50],
    elevation_deg=60.0,
    gain_noise_params=[1e-5, 1e-3, 1.8]  # Custom noise parameters
)

# Plot frequency-time waterfall
plt.figure(figsize=(10, 6))
plt.imshow(sky_multifreq, aspect='auto', origin='lower',
           extent=[0, len(time_list[:50]), frequencies[0], frequencies[-1]])
plt.colorbar(label='Temperature [K]')
plt.xlabel('Time sample')
plt.ylabel('Frequency [MHz]')
plt.title('Sky TOD - Frequency vs Time')
plt.show()
```

### Example 3: Custom Beam and Sky Models

```python
def custom_beam(freq, nside):
    """Custom frequency-dependent beam"""
    # Beam size scales with frequency
    fwhm = 70 / freq  # degrees, typical radio telescope scaling
    return example_beam_map(freq, nside, FWHM_major=fwhm, FWHM_minor=fwhm*0.8)

def point_source_sky(freq, nside):
    """Sky with a single point source"""
    npix = hp.nside2npix(nside)
    sky = np.zeros(npix)
    
    # Add point source at specific coordinates
    ra, dec = 180.0, -30.0  # degrees
    theta = np.pi/2 - np.radians(dec)
    phi = np.radians(ra)
    
    ipix = hp.ang2pix(nside, theta, phi)
    sky[ipix] = 100.0  # 100 K source
    
    return sky

# Use custom models
sim_custom = meerTODsim(
    beam_func=custom_beam,
    sky_func=point_source_sky,
    nside=256
)

# Simulate with custom models
tod_custom, _, _ = sim_custom.generate_TOD(
    freq_list=[1000],
    time_list=time_list,
    azimuth_deg_list=az_list,
    elevation_deg=50.0
)
```

### Example 4: High-Resolution Simulation with Caching

```python
import pickle

# High-resolution simulation
sim_hires = meerTODsim(nside=512)

# Generate longer time series
t_long, az_long = example_scan(dt=0.5)  # Higher time resolution

# For computational efficiency, you might want to cache sky maps
def cached_sky_model(freq, nside):
    cache_file = f"sky_{freq}MHz_nside{nside}.pkl"
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        sky = GDSM_sky_model(freq, nside)
        with open(cache_file, 'wb') as f:
            pickle.dump(sky, f)
        return sky

sim_hires.sky_func = cached_sky_model

tod_hires, _, _ = sim_hires.generate_TOD(
    freq_list=np.arange(950, 1051, 10),  # 10 MHz steps
    time_list=t_long,
    azimuth_deg_list=az_long,
    elevation_deg=30.0
)
```

## Mathematical Foundations

### Spherical Harmonic Rotations

The core mathematical operation is rotating spherical harmonics to simulate telescope pointing. Given a function on the sphere represented as:

```
f(θ,φ) = Σ_l Σ_m a_lm Y_lm(θ,φ)
```

A rotation by Euler angles (α,β,γ) transforms the coefficients as:

```
a'_lm = Σ_m' a_lm' D^l_mm'(α,β,γ)
```

Where D^l_mm' are Wigner D-matrices. HEALPix implements this efficiently via `rotate_alm()`.

### Noise Modeling

#### 1/f Gain Noise

Gain fluctuations follow a power spectrum:

```
P(f) = f_0^α / (f_c^2 + f^2)^(α/2)
```

The time-domain correlation function is:

```
C(τ) = (f_0/f_c)^α × Γ(1-α,|f_c τ|) / (π|τ|)
```

This is implemented using incomplete gamma functions from `mpmath`.

#### White Noise

Thermal noise is modeled as uncorrelated Gaussian:

```
⟨η(t)η(t')⟩ = σ²δ(t-t')
```

### Coordinate Systems

The package handles transformations between three coordinate systems:

1. **Horizontal** (Az, El): Telescope-fixed coordinates
2. **Equatorial** (RA, Dec): Earth-fixed celestial coordinates  
3. **HEALPix** (θ, φ): Spherical coordinates for map pixels

The transformation chain:
(Az,El,t) → (RA,Dec) → (θ,φ) → Euler angles → Rotated beam

## Performance Considerations

### Memory Usage

- HEALPix maps scale as O(N_side²) 
- For N_side=512: ~3 million pixels, ~12 MB per map
- Spherical harmonic coefficients: ~(N_side×3)² complex numbers

### Computational Scaling

- `map2alm()`: O(N_side² log N_side)
- `rotate_alm()`: O(N_side²)  
- `alm2map()`: O(N_side² log N_side)

### Parallel Processing

The package supports MPI parallelization over frequencies:

```python
# Automatic parallelization over frequency list
tod_array = sim.generate_TOD(freq_list=frequencies, ...)

# Run with MPI
# mpirun -n 4 python simulation_script.py
```

### Optimization Tips

1. **Use appropriate N_side**: Balance resolution vs. speed
2. **Cache sky maps**: Avoid recomputing identical sky models
3. **Batch processing**: Process multiple frequencies together
4. **Memory mapping**: Use `numpy.memmap` for large datasets

## Error Handling and Validation

Common error scenarios and solutions:

- **Memory errors**: Reduce `nside` or process in smaller batches
- **Coordinate errors**: Check that Az/El values are within valid ranges
- **Time errors**: Ensure UTC timestamps are properly formatted
- **MPI errors**: Check that all processes have consistent inputs


### Development Setup

```bash
git clone https://github.com/zzhang0123/meerTOD.git
cd meerTOD
pip install -e ".[dev]"
pytest tests/
```

## License

This project is licensed under the MIT License - see LICENSE file for details.



## Acknowledgments

- MeerKLASS team 

---

For more information, visit the [project repository](https://github.com/yourusername/meerTOD) or contact the maintainers.
