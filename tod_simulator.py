"""
MeerTOD: Time-Ordered Data Simulator forMeerKLASS

For comprehensive documentation, see README.md.
"""

from typing import Union, List, Tuple
import numpy as np
import healpy as hp
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
import astropy.units as u
import tqdm
from scipy.spatial.transform import Rotation as R



# Enhanced type aliases for better code readability
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]
TimeList = ArrayLike
FrequencyList = ArrayLike
AngleList = ArrayLike

# Enhanced constants with better documentation
DEFAULT_MEERKAT_LATITUDE = -30.7130   # degrees (MeerKAT coordinates)
DEFAULT_MEERKAT_LONGITUDE = 21.4430   # degrees
DEFAULT_MEERKAT_HEIGHT = 1054         # meters above sea level
DEFAULT_START_TIME_UTC = "2019-04-23 20:41:56.397"
DEFAULT_WHITE_NOISE_VAR = 2.5e-6      # Typical thermal noise variance
DEFAULT_GAIN_NOISE_PARAMS = [1.4e-5, 1e-3, 2.0]  # [f0, fc, alpha] for 1/f noise



def example_scan(
        az_s=-60.3, 
        az_e=-42.3, 
        dt=2.0
    ):

    aux = np.linspace(az_s, az_e, 111)
    azimuths = np.concatenate((aux[1:-1][::-1], aux))
    azimuths = np.tile(azimuths, 5)

    # Length of TOD
    ntime = len(azimuths)
    t_list = np.arange(ntime) * dt

    return t_list, azimuths

def zyzy2zyz(alpha, beta, gamma, delta, output_degrees=False):
    '''
    Convert "zyzy" angles to effective "zyz" angles.
    Input angles are in degrees.
    Output angles are in degrees if output_degrees=True, else in radians.

    "zyzy"-rotation: R = R_y(delta) * R_z(gamma) * R_y(beta) * R_z(alpha)
    "zyz"-rotation: R = R_z(phi) * R_y(theta) * R_z(psi)

    Parameters:
    alpha : float
        First "z" rotation angle in degrees.
    beta : float
        First "y" rotation angle in degrees.
    gamma : float
        Second "z" rotation angle in degrees.
    delta : float
        Second "y" rotation angle in degrees.
    output_degrees : bool, optional
        If True, output angles are in degrees. Default is False (radians).

    Returns:
    tuple
        A tuple containing the (psi, theta, phi) angles.
    '''
    r = (
    R.from_euler("y", delta, degrees=True) *
    R.from_euler("z", gamma, degrees=True) *
    R.from_euler("y", beta, degrees=True) *
    R.from_euler("z", alpha, degrees=True) 
    )
    psi, theta, phi = r.as_euler("zyz", degrees=output_degrees) 
    return psi, theta, phi 

def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg):
    '''
    This function generates the effective "zyz"-rotation angles (psi, theta, phi)
    from the pointing parameters: LST, latitude, azimuth, and elevation.
    All input angles are in degrees.
    The output angles (psi, theta, phi) are in radians, as that is the unit used by hp.rotate_alm().

    Parameters:
    LST_deg : float
        The site's Local Sidereal Time in degrees.
    lat_deg : float
        The site's latitude in degrees.
    azimuth_deg : float
        The pointing's azimuth in degrees.
    elevation_deg : float
        The pointing's elevation in degrees.

    Returns:
    tuple
        A tuple containing the (psi, theta, phi) angles in radians.
    '''

    # Convert pointing parameters to "zyzy" angles
    alpha = LST_deg
    beta = 90.0 - lat_deg
    gamma = azimuth_deg
    delta = 90.0 - elevation_deg

    # Convert "zyzy" angles to effective "zyz" angles
    return zyzy2zyz(alpha, beta, gamma, delta)

def generate_LSTs_deg(
        ant_latitude_deg,
        ant_longitude_deg, 
        ant_height_m,
        time_list, 
        start_time_utc="2019-04-23 20:41:56.397"):
    '''
    Generate Local Sidereal Time (LST) values in degrees for a list of time offsets.

    Parameters:
    ant_latitude_deg : float
        Latitude of the antenna/site in degrees.
    ant_longitude_deg : float
        Longitude of the antenna/site in degrees.
    ant_height_m : float
        Height of the antenna/site in meters.
    time_list: list or array of time offsets in seconds from start_time_utc.
    start_time_utc : str
        Start time in UTC (e.g. "2019-04-23 20:41:56.397").

    Returns:
    LST_list_deg : array
        Array of Local Sidereal Time values in degrees corresponding to each time offset.
    '''
    # --- site coordinates ---
    site = EarthLocation(lat=ant_latitude_deg*u.deg, lon=ant_longitude_deg*u.deg, height=ant_height_m*u.m)

    # --- define start time and offsets (sec) ---
    start_time = Time(start_time_utc)                    
    UTC_list = start_time + TimeDelta(time_list, format="sec")

    # --- compute Local Sidereal Time ---
    LST_list = UTC_list.sidereal_time("apparent", longitude=site.lon)

    # convert to degrees (Angle object → value)
    LST_list_deg = LST_list.to(u.deg).value

    return LST_list_deg

def _rotate_healpix_map(alm, psi_rad, theta_rad, phi_rad, nside, return_map=True):
    '''
    Rotate a Healpix map represented by its alm coefficients using given Euler angles (psi, theta, phi).
    The rotation is performed in-place on a copy of the alm coefficients.

    Parameters:
    alm : array
        The alm coefficients of the Healpix map to be rotated.
    psi_rad : float
        The first Euler angle (rotation about z-axis) in radians.
    theta_rad : float
        The second Euler angle (rotation about y-axis) in radians.
    phi_rad : float
        The third Euler angle (rotation about z-axis) in radians.
    nside : int
        The nside parameter of the Healpix map.
    return_map : bool, optional
        If True, return the rotated map. If False, return the rotated alm coefficients. Default is True.

    Returns:
    array
        The rotated Healpix map (if return_map is True) or the rotated alm coefficients (if return_map is False).
    '''

    # Make a copy of alm since hp.rotate_alm operates in-place
    alm_rot = alm.copy()
    hp.rotate_alm(alm_rot, psi_rad, theta_rad, phi_rad)
    if return_map:
        map_pointed = hp.alm2map(alm_rot, nside)
        return map_pointed
    return alm_rot

def _normalize_map(input_map):
    '''
    Normalize a Healpix map to have a sum value of 1.
    
    Parameters:
    input_map : array
        The Healpix map to be normalized.

    Returns:
    array
        The normalized Healpix map.
    '''
    return input_map / np.sum(input_map)

def pointing_beam_in_eq_sys(beam_lm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside):
    '''
    Point the beam in the equatorial coordinate system.
    Parameters:
    beam_lm : array
        The alm coefficients of the beam in its native orientation.
    LST_deg : float
        The Local Sidereal Time in degrees.
    lat_deg : float
        The latitude of the observation site in degrees.
    azimuth_deg : float
        The azimuth of the pointing in degrees.
    elevation_deg : float
        The elevation of the pointing in degrees.
    nside : int
        The nside parameter of the Healpix map.

    Returns:
    array 
        The pointed beam map in the equatorial coordinate system.
    '''
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg)
    beam_pointed = _rotate_healpix_map(beam_lm, psi_rad, theta_rad, phi_rad, nside)
    return beam_pointed

def _beam_weighted_sum(beam_map, sky_map):
    '''
    Compute the beam-weighted sum of the sky map.
    
    Parameters:
    beam_map : array
        The Healpix map of the beam (should be normalized).
    sky_map : array
        The Healpix map of the sky.

    Returns:
    float
        The beam-weighted sum of the sky map.
    '''
    beam_map_normalized = _normalize_map(beam_map)
    return np.sum(beam_map_normalized * sky_map)

def generate_TOD_sky(beam_map, sky_map, LST_deg_list, lat_deg, azimuth_deg_list, elevation_deg_list):
    '''
    Generate Time-Ordered Data (TOD) by simulating observations of a sky map with a given beam pattern.
    Note that the TOD represents the beam-weighted sum of the sky map at each pointing.
    
    Parameters:
    beam_map : array
        The Healpix map of the beam pattern.
    sky_map : array
        The Healpix map of the sky.
    LST_deg_list : array
        List of Local Sidereal Time values in degrees for each observation.
    lat_deg : float
        The latitude of the observation site in degrees.
    azimuth_deg_list : array
        List of azimuth values in degrees for each observation.
    elevation_deg_list : array
        List of elevation values in degrees for each observation.

    Returns:
    array
        The generated Time-Ordered Data (TOD) as a 1D array.
    '''

    beam_alm = hp.map2alm(beam_map)
    nside = hp.get_nside(beam_map)
    tod = []

    #for LST_deg, azimuth_deg, elevation_deg in zip(LST_deg_list, azimuth_deg_list, elevation_deg_list):
    for LST_deg, azimuth_deg, elevation_deg in tqdm.tqdm(zip(LST_deg_list, azimuth_deg_list, elevation_deg_list), total=len(LST_deg_list)):
        beam_pointed = pointing_beam_in_eq_sys(beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside=nside)
        sample = _beam_weighted_sum(beam_pointed, sky_map)
        tod.append(sample)

    return np.array(tod)


import mpiutil 
from flicker_model import sim_noise

def GDSM_sky_model(freq, nside):
    from pygdsm import GlobalSkyModel
    gsm = GlobalSkyModel()
    skymap = gsm.generate(freq)
    skymap = hp.ud_grade(skymap, nside_out=nside)
    return skymap

def example_beam_map(freq, nside, FWHM_major=1.1, FWHM_minor=1.1):
    """
    Generate an asymmetric (elliptical) Gaussian beam map.
    FWHM_major: major axis FWHM in degrees
    FWHM_minor: minor axis FWHM in degrees
    """
    NPIX = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(NPIX))
    # Convert FWHM to sigma (radians)
    sigma_major = np.radians(FWHM_major / (2 * np.sqrt(2 * np.log(2))))
    sigma_minor = np.radians(FWHM_minor / (2 * np.sqrt(2 * np.log(2))))
    angle_rad = 0.0
    # Compute offsets from beam center

    dtheta = theta
    dphi = phi
    # Convert to Cartesian offsets
    x = dtheta * np.cos(phi)
    y = dtheta * np.sin(phi)
    # Rotate by angle
    x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    # Elliptical Gaussian
    beam_map = np.exp(-0.5 * ((x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2))
    # Normalize
    beam_map /= np.max(beam_map)
    # hp.mollview(beam_map, title="Elliptical (Asymmetric) Beam Map")
    return beam_map


class meerTODsim:
    def __init__(self, 
                 ant_latitude_deg=-30.7130, 
                 ant_longitude_deg=21.4430, 
                 ant_height_m=1054, 
                 beam_func=example_beam_map, 
                 sky_func=GDSM_sky_model, 
                 nside=256):
        '''
        Initialize the meerTODsim class.
        Parameters:
        ant_latitude_deg : float
            Latitude of the antenna/site in degrees.
        ant_longitude_deg : float
            Longitude of the antenna/site in degrees.
        ant_height_m : float
            Height of the antenna/site in meters.
        beam_func : function
            Function that takes frequency and nside as input and returns the beam map.
        sky_func : function
            Function that takes frequency and nside as input and returns the sky map.
        nside : int, optional
            The nside parameter for Healpix maps. 
        '''
        self.ant_latitude_deg = ant_latitude_deg
        self.ant_longitude_deg = ant_longitude_deg
        self.ant_height_m = ant_height_m
        self.nside = nside
        self.beam_func = beam_func
        self.sky_func = sky_func

    def simulate_sky_TOD(self, freq_list, time_list, azimuth_deg_list, elevation_deg, start_time_utc="2019-04-23 20:41:56.397"):
        '''
        Simulate sky TOD (beam-weighted sum of sky map) for a list of frequencies and time offsets.

        Parameters:
        freq_list : list or array
            List of frequencies.
        time_list : list or array
            List of time offsets in seconds (for each observation time) from start_time_utc.
        azimuth_deg_list : list or array
            List of azimuth values in degrees for each observation.
        elevation_deg : list or float
            If float: Elevation value in degrees universal for all observations.
            If list: List of elevation values in degrees for each observation.
        start_time_utc : str
            Start time in UTC (e.g. "2019-04-23 20:
        '''
        # the beam-weighted sum of the sky map at each pointing.

        ntime = len(time_list)
        if isinstance(elevation_deg, (int, float)):
            elevation_deg_list = np.full(ntime, elevation_deg)
        else:
            elevation_deg_list = elevation_deg

        LST_deg_list = generate_LSTs_deg(
            self.ant_latitude_deg,
            self.ant_longitude_deg,
            self.ant_height_m,
            time_list,
            start_time_utc=start_time_utc
        )

        def single_freq_sky_TOD(freq):            
            beam_map = self.beam_func(freq, self.nside)
            sky_map = self.sky_func(freq, self.nside)

            tod = generate_TOD_sky(
                beam_map,
                sky_map,
                LST_deg_list,
                self.ant_latitude_deg,
                azimuth_deg_list,
                elevation_deg_list
            )

            return tod

        TOD_array = np.array(mpiutil.parallel_map_gather(single_freq_sky_TOD, freq_list))

        return TOD_array

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
                     white_noise_var=None
                     ):
        '''
        Generate overall TOD including sky signal and other components.

        Data model:
        overall_TOD = background_gain_TOD * (1 + gain_noise_TOD) * (sky_TOD + residual_Tsys_TOD) * (1 + white_noise_TOD)

        Parameters:
        freq_list : list or array
            List of frequencies.
        time_list : list or array
            List of time offsets in seconds (for each observation time) from start_time_utc.
        azimuth_deg_list : list or array
            List of azimuth values in degrees for each observation.
        elevation_deg : list or float
            If float: Elevation value in degrees universal for all observations.
            If list: List of elevation values in degrees for each observation.
        start_time_utc : str
            Start time in UTC (e.g. "2019-04-23 20:41:56.397").
        residual_Tsys_TOD : array, optional
            Array of residual system temperature TOD (shape: nfreq x ntime). Default is None (no residual).
        background_gain_TOD : array, optional
            Array of background gain TOD (shape: nfreq x ntime). Default is None (unity gain).
        gain_noise_TOD : array, optional
            Array of gain noise TOD (shape: nfreq x ntime). Default is None (no gain noise).
        gain_noise_params : list, optional
            List of parameters [f0, fc, alpha] for generating gain noise if gain_noise_TOD is None. Default is [1.4e-5, 1e-3, 2].
        white_noise_var : float, optional
            Variance of white noise to be added. Default is None (uses default value of 2.5e-6).

        Returns:
        overall_TOD : array
            The overall generated TOD (shape: nfreq x ntime).
        sky_TOD : array
            The sky signal TOD (shape: nfreq x ntime).
        gain_noise_TOD : array
            The gain noise TOD (shape: nfreq x ntime).
        '''


        nfreq = len(freq_list)
        ntime = len(time_list)

        if residual_Tsys_TOD is None:
            residual_Tsys_TOD = 0.0
        if background_gain_TOD is None:
            background_gain_TOD = 1.0
        if white_noise_var is None:
            if mpiutil.rank0:
                print("No white noise variance is specified!! Using default value of 2.5e-6 (Dimensionless fractional noise)")
            white_noise_var = 2.5e-6
        
        if gain_noise_TOD is None:
            if gain_noise_params is None:
                gain_noise_TOD = 0.0
            else:
                if mpiutil.rank0:
                    print(f"Generating gain noise with parameters: f0={gain_noise_params[0]}, fc={gain_noise_params[1]}, alpha={gain_noise_params[2]}.  \
                          Note that these 1/f noise are uncorrelated between frequencies.")

                f0, fc, alpha = gain_noise_params
                gain_noise_TOD = sim_noise(f0, fc, alpha, time_list, n_samples=nfreq, white_n_variance=0.0)
        elif gain_noise_params is not None:
            if mpiutil.rank0:
                print("Warning: Both gain_noise_TOD and gain_noise_params are provided. Ignoring gain_noise_params.")


        white_noise_TOD = np.random.normal(0, np.sqrt(white_noise_var), size=(nfreq, ntime))

        sky_TOD = self.simulate_sky_TOD(freq_list, time_list, azimuth_deg_list, elevation_deg, start_time_utc=start_time_utc)
        overall_TOD = background_gain_TOD * (1 + gain_noise_TOD) * (sky_TOD + residual_Tsys_TOD) * (1 + white_noise_TOD)

        return overall_TOD, sky_TOD, gain_noise_TOD
    

# # Example usage:
# TOD_sim_test = meerTODsim()
# t_list, azimuths = example_scan()

# TOD_arr, sky_TOD, gain_noise_TOD = TOD_sim_test.generate_TOD(
#     freq_list=[950, 1050], 
#     time_list=t_list, 
#     azimuth_deg_list=azimuths,
# ) 