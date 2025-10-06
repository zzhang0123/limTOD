"""
limTOD: Time-Ordered Data Simulator for single-dish radio telescopes.

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

import limTOD.mpiutil as mpiutil
import limTOD.sky_model as sky_model
from limTOD.flicker_model import sim_noise


# Enhanced type aliases for better code readability
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]
TimeList = ArrayLike
FrequencyList = ArrayLike
AngleList = ArrayLike

# Enhanced constants with better documentation
DEFAULT_MEERKAT_LATITUDE = -30.7130  # degrees (MeerKAT coordinates)
DEFAULT_MEERKAT_LONGITUDE = 21.4430  # degrees
DEFAULT_MEERKAT_HEIGHT = 1054  # meters above sea level
DEFAULT_START_TIME_UTC = "2019-04-23 20:41:56.397"
DEFAULT_WHITE_NOISE_VAR = 2.5e-6  # Typical thermal noise variance
DEFAULT_GAIN_NOISE_PARAMS = [1.4e-5, 1e-3, 2.0]  # [f0, fc, alpha] for 1/f noise


def example_scan(az_s=-60.3, az_e=-42.3, dt=2.0, n_repeats=5):
    aux = np.linspace(az_s, az_e, 111)
    azimuths = np.concatenate((aux[1:-1][::-1], aux))
    azimuths = np.tile(azimuths, n_repeats)

    # Length of TOD
    ntime = len(azimuths)
    t_list = np.arange(ntime) * dt

    return t_list, azimuths


def zyzy2zyz(alpha, beta, gamma, delta, output_degrees=False):
    """
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
    """
    r = (
        R.from_euler("y", delta, degrees=True)
        * R.from_euler("z", gamma, degrees=True)
        * R.from_euler("y", beta, degrees=True)
        * R.from_euler("z", alpha, degrees=True)
    )
    psi, theta, phi = r.as_euler("zyz", degrees=output_degrees)
    return psi, theta, phi


def zyz_of_pointing(LST_deg, lat_deg, azimuth_deg, elevation_deg):
    """
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
    """

    # Convert pointing parameters to "zyzy" angles
    alpha = LST_deg
    beta = 90.0 - lat_deg
    gamma = (
        -azimuth_deg
    )  # Note the sign convention for azimuth: East of North is positive
    delta = elevation_deg - 90.0

    # Convert "zyzy" angles to effective "zyz" angles
    return zyzy2zyz(alpha, beta, gamma, delta)


def generate_LSTs_deg(
    ant_latitude_deg,
    ant_longitude_deg,
    ant_height_m,
    time_list,
    start_time_utc="2019-04-23 20:41:56.397",
):
    """
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
    """
    # --- site coordinates ---
    site = EarthLocation(
        lat=ant_latitude_deg * u.deg,
        lon=ant_longitude_deg * u.deg,
        height=ant_height_m * u.m,
    )

    # --- define start time and offsets (sec) ---
    start_time = Time(start_time_utc)
    UTC_list = start_time + TimeDelta(time_list, format="sec")

    # --- compute Local Sidereal Time ---
    LST_list = UTC_list.sidereal_time("apparent", longitude=site.lon)

    # convert to degrees (Angle object → value)
    LST_list_deg = LST_list.to(u.deg).value

    return LST_list_deg


def _rotate_healpix_map(alm, psi_rad, theta_rad, phi_rad, nside, return_map=True):
    """
    Rotate a Healpix map represented by its alm coefficients using given Euler angles (psi, theta, phi).
    The rotation is performed in-place on a copy of the alm coefficients.

    Parameters:
    alm : array
        The alm coefficients of the Healpix map to be rotated.
        Input map(s) can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
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
    """

    # Make a copy of alm since hp.rotate_alm operates in-place
    # If input alm is single array or 3-row array, directly rotate
    alm_rot = np.zeros_like(alm, dtype=alm.dtype)

    if alm.ndim == 1 or alm.shape[0] == 3:
        alm_copy = alm.copy()
        hp.rotate_alm(alm_copy, phi_rad, theta_rad, psi_rad)
        alm_rot = alm_copy
        stokes_V = False
    elif alm.shape[0] == 4:
        # If input alm has 4 rows, ignore the V component (4th row)
        alm_copy = alm[:3].copy()
        hp.rotate_alm(alm_copy, phi_rad, theta_rad, psi_rad)
        alm_rot[:3] = alm_copy

        alm_copy = alm[3].copy()
        hp.rotate_alm(alm_copy, phi_rad, theta_rad, psi_rad)
        alm_rot[3] = alm_copy
        stokes_V = True
    else:
        raise ValueError("Input alm must be a 1D array or a 2D array with 3 or 4 rows.")

    if return_map:
        if stokes_V:
            map_pointed = hp.alm2map(alm_rot[:3], nside)
            map_V = hp.alm2map(alm_rot[3], nside)
            map_pointed = np.vstack((map_pointed, map_V))
        else:
            map_pointed = hp.alm2map(alm_rot, nside)
        return map_pointed
    return alm_rot


def _normalize_map(input_map):
    """
    Normalize a Healpix map to have a sum value of 1.

    Parameters:
    input_map : array
        The Healpix map to be normalized.

    Returns:
    array
        The normalized Healpix map.
    """
    return input_map / np.sum(input_map)


def pointing_beam_in_eq_sys(
    beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside
):
    """
    Point the beam in the equatorial coordinate system.
    Parameters:
    beam_alm : array
        The alm coefficients of the beam in its native orientation.
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
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
    """
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(
        LST_deg=LST_deg,
        lat_deg=lat_deg,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )
    beam_pointed = _rotate_healpix_map(beam_alm, psi_rad, theta_rad, phi_rad, nside)
    return beam_pointed


def _beam_weighted_sum(beam_map, sky_map, normalize=False):
    """
    Compute the beam-weighted sum of the sky map.

    Parameters:
    beam_map : array
        The Healpix map of the beam (should be normalized).
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
    sky_map : array
        The Healpix map of the sky.
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
    normalize : bool, optional
        If True, normalize the Stokes-I beam map to have a sum of 1 before computing the weighted sum.
        All other Stokes parameters (Q,U,V) will be scaled by the same factor.

    Returns:
    float
        The beam-weighted sum of the sky map.
    """
    if normalize:
        # Create a copy to avoid modifying the input array
        beam_map = beam_map.copy()
        if beam_map.ndim == 1:
            beam_map = _normalize_map(beam_map)
        elif beam_map.shape[0] in [3, 4]:
            norm_factor = np.sum(beam_map[0])
            beam_map[0] = _normalize_map(beam_map[0])
            # Scale other Stokes parameters by the same factor
            if norm_factor > 0:
                beam_map[1:] = beam_map[1:] / norm_factor
            else:
                print("Warning: Beam normalization factor is zero!")
        else:
            raise ValueError(
                "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
            )

    return np.sum(beam_map * sky_map)


def generate_TOD_sky(
    beam_map, sky_map, LST_deg_list, lat_deg, azimuth_deg_list, elevation_deg_list
):
    """
    Generate Time-Ordered Data (TOD) by simulating observations of a sky map with a given beam pattern.
    Note that the TOD represents the beam-weighted sum of the sky map at each pointing.

    Parameters:
    beam_map : array
        The Healpix map of the beam pattern.
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
    sky_map : array
        The Healpix map of the sky.
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
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
    """
    assert (
        beam_map.shape == sky_map.shape
    ), "Beam map and sky map must have the same shape."

    # Convert beam map to alm coefficients
    if beam_map.ndim == 1 or beam_map.shape[0] == 3:
        beam_alm = hp.map2alm(beam_map)
    elif beam_map.shape[0] == 4:
        beam_alm_IQU = hp.map2alm(beam_map[:3])
        beam_alm_V = hp.map2alm(beam_map[3])
        beam_alm = np.vstack((beam_alm_IQU, beam_alm_V))
    else:
        raise ValueError(
            "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
        )

    if beam_alm.ndim == 1:
        nside = hp.get_nside(beam_map)
    else:
        nside = hp.get_nside(beam_map[0])

    tod = []

    # for LST_deg, azimuth_deg, elevation_deg in zip(LST_deg_list, azimuth_deg_list, elevation_deg_list):
    for LST_deg, azimuth_deg, elevation_deg in tqdm.tqdm(
        zip(LST_deg_list, azimuth_deg_list, elevation_deg_list), total=len(LST_deg_list)
    ):
        beam_pointed = pointing_beam_in_eq_sys(
            beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside=nside
        )
        sample = _beam_weighted_sum(beam_pointed, sky_map)
        tod.append(sample)

    return np.array(tod)


def example_beam_map(*, freq, nside, FWHM_major=1.1, FWHM_minor=1.1):
    """
    Generate an example Gaussian beam map.
    This toy model is achromatic.

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
    beam_map /= np.sum(beam_map)
    # hp.mollview(beam_map, title="Elliptical (Asymmetric) Beam Map")
    return beam_map


def example_symm_beam_map(*, freq, nside, FWHM=1.1):
    """
    Generate a symmetric Gaussian beam map centered at the pole.

    Parameters:
    freq : float
        Frequency (not used in this achromatic model, but kept for API consistency).
    nside : int
        HEALPix nside parameter.
    FWHM : float, optional
        Full Width at Half Maximum of the Gaussian beam in degrees. Default is 1.1.

    Returns:
    beam_map : array
        Normalized Gaussian beam map (1D array, sum = 1).
    """

    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma (degrees)
    sigma_rad = np.radians(sigma)  # Convert to radians

    NPIX = hp.nside2npix(nside)

    # Get HEALPix pixel coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, np.arange(NPIX))

    # Compute Gaussian beam response centered at the north pole (theta=0)
    beam_map = np.exp(-0.5 * (theta / sigma_rad) ** 2)
    # Normalize so that the beam integrates to 1 (sum over all pixels = 1)
    beam_map /= np.sum(beam_map)

    return beam_map


class TODSim:
    def __init__(
        self,
        ant_latitude_deg=-30.7130,
        ant_longitude_deg=21.4430,
        ant_height_m=1054,
        beam_func=example_beam_map,
        sky_func=sky_model.GDSM_sky_model,
        nside=256,
    ):
        """
        Initialize the limTODsim class.
        Parameters:
        ant_latitude_deg : float
            Latitude of the antenna/site in degrees.
        ant_longitude_deg : float
            Longitude of the antenna/site in degrees.
        ant_height_m : float
            Height of the antenna/site in meters.
        beam_func : function
            Function that takes frequency and nside as keyword input and returns the beam map.
            The output map can be:
                a single array is considered I,
                array with 3 rows:[I,Q,U]
                array with 4 rows:[I,Q,U,V]
        sky_func : function
            Function that takes frequency and nside as keyword input and returns the sky map.
            The output map can be:
                a single array is considered I,
                array with 3 rows:[I,Q,U]
                array with 4 rows:[I,Q,U,V]
        nside : int, optional
            The nside parameter for Healpix maps.
        """
        self.ant_latitude_deg = ant_latitude_deg
        self.ant_longitude_deg = ant_longitude_deg
        self.ant_height_m = ant_height_m
        self.nside = nside
        self.beam_func = beam_func
        self.sky_func = sky_func

    def simulate_sky_TOD(
        self,
        freq_list,
        time_list,
        azimuth_deg_list,
        elevation_deg,
        start_time_utc="2019-04-23 20:41:56.397",
        return_LSTs=False,
    ):
        """
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
            Start time in UTC (e.g. "2019-04-23 20:41:56.397").
        """
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
            start_time_utc=start_time_utc,
        )

        def single_freq_sky_TOD(freq):
            beam_map = self.beam_func(freq=freq, nside=self.nside)
            sky_map = self.sky_func(freq=freq, nside=self.nside)

            tod = generate_TOD_sky(
                beam_map,
                sky_map,
                LST_deg_list,
                self.ant_latitude_deg,
                azimuth_deg_list,
                elevation_deg_list,
            )

            return tod

        TOD_array = np.array(
            mpiutil.parallel_map_gather(single_freq_sky_TOD, freq_list)
        )

        if return_LSTs:
            return TOD_array, LST_deg_list
        return TOD_array

    def generate_TOD(
        self,
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
    ):
        """
        Generate overall TOD including sky signal and other components.

        Data model:
        overall_TOD = background_gain_TOD * (1 + gain_noise_TOD) * (sky_TOD + Tsys_others_TOD) * (1 + white_noise_TOD)

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
        Tsys_others_TOD : array, optional
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
        """

        nfreq = len(freq_list)
        ntime = len(time_list)

        if Tsys_others_TOD is None:
            Tsys_others_TOD = 0.0
        if background_gain_TOD is None:
            background_gain_TOD = 1.0
        if white_noise_var is None:
            if mpiutil.rank0:
                print(
                    "No white noise variance is specified!! Using default value of 2.5e-6 (Dimensionless fractional noise)"
                )
            white_noise_var = 2.5e-6

        if gain_noise_TOD is None:
            if gain_noise_params is None:
                gain_noise_TOD = 0.0
            else:
                if mpiutil.rank0:
                    print(
                        "Generating gain noise with parameters: "
                        f"f0={gain_noise_params[0]}, fc={gain_noise_params[1]}, "
                        f"alpha={gain_noise_params[2]}."
                        "\nNote that these 1/f noise are uncorrelated in frequencies."
                    )

                f0, fc, alpha = gain_noise_params
                gain_noise_TOD = sim_noise(
                    f0, fc, alpha, time_list, n_samples=nfreq, white_n_variance=0.0
                )
        elif gain_noise_params is not None:
            if mpiutil.rank0:
                print(
                    "Warning: Both gain_noise_TOD and gain_noise_params are provided. Ignoring gain_noise_params."
                )

        white_noise_TOD = np.random.normal(
            0, np.sqrt(white_noise_var), size=(nfreq, ntime)
        )

        sky_TOD, LST_deg_list = self.simulate_sky_TOD(
            freq_list,
            time_list,
            azimuth_deg_list,
            elevation_deg,
            start_time_utc=start_time_utc,
            return_LSTs=True,
        )

        overall_TOD = (
            background_gain_TOD
            * (1 + gain_noise_TOD)
            * (sky_TOD + Tsys_others_TOD)
            * (1 + white_noise_TOD)
        )
        if return_LSTs:
            return overall_TOD, sky_TOD, gain_noise_TOD, LST_deg_list
        return overall_TOD, sky_TOD, gain_noise_TOD


def truncate_stacked_beam(
    beam_map,
    LST_deg_list,
    lat_deg,
    azimuth_deg_list,
    elevation_deg_list,
    threshold=0.01,
):
    """
    Generate the selected pixel indices based on beam sensitivity.
    The selected pixels are those with beam response above a given threshold in the stacked abs(beam) map.

    Parameters:
    beam_map : array
        The Healpix map of the beam pattern.
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
    LST_deg_list : array
        List of Local Sidereal Time values in degrees for each observation.
    lat_deg : float
        The latitude of the observation site in degrees.
    azimuth_deg_list : array
        List of azimuth values in degrees for each observation.
    elevation_deg_list : array
        List of elevation values in degrees for each observation.
    threshold : float
        The threshold to cut off the fractional beam response np.abs(beam[pixel])/beam_max, default is 0.01.
        e.g., if threshold=0.01, only pixels with beam response larger than 1% of the maximum will be considered.

    Returns:
    pixel_indices : array
        The selected pixel indices based on the beam sensitivity.
    """

    # Convert beam map to alm coefficients
    if beam_map.ndim == 1 or beam_map.shape[0] == 3:
        beam_alm = hp.map2alm(beam_map)
    elif beam_map.shape[0] == 4:
        beam_alm_IQU = hp.map2alm(beam_map[:3])
        beam_alm_V = hp.map2alm(beam_map[3])
        beam_alm = np.vstack((beam_alm_IQU, beam_alm_V))
    else:
        raise ValueError(
            "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
        )

    if beam_alm.ndim == 1:
        nside = hp.get_nside(beam_map)
    else:
        nside = hp.get_nside(beam_map[0])

    # Integrate the beam map as the sum map, select pixels above threshold

    print("\nStep 1: Generating the stacked abs(beam) map ... \n")
    # Generate a initial boolean map with all pixels zero
    bool_map = np.zeros_like(beam_map, dtype=bool)

    for LST_deg, azimuth_deg, elevation_deg in tqdm.tqdm(
        zip(LST_deg_list, azimuth_deg_list, elevation_deg_list), total=len(LST_deg_list)
    ):
        beam_pointed = pointing_beam_in_eq_sys(
            beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside=nside
        )
        norm = np.max(np.abs(beam_pointed))
        if norm > 0:
            beam_pointed = beam_pointed / norm
            bool_map = np.logical_or(bool_map, beam_pointed > threshold)
        else:
            print("Warning: Beam has zero maximum value at this pointing!")

    print("\nStep 2: Selecting pixels above threshold sensitivity ... \n")
    if bool_map.ndim == 2:
        bool_map = np.any(bool_map, axis=0)
    pixel_indices = np.where(bool_map)[0]

    return pixel_indices


def generate_sky2sys_projection(
    beam_map,
    LST_deg_list,
    lat_deg,
    azimuth_deg_list,
    elevation_deg_list,
    pixel_indices,
    normalize=False,
):
    """
    Generate the sky-to-Tsys projection matrix and the selected pixel indices based on beam sensitivity.
    The projection matrix maps the sky pixels to the system temperature.

    Parameters:
    beam_map : array
        The Healpix map of the beam pattern.
        Input map can be:
            a single array is considered I,
            array with 3 rows:[I,Q,U]
            array with 4 rows:[I,Q,U,V]
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
    """

    # Convert beam map to alm coefficients
    if beam_map.ndim == 1 or beam_map.shape[0] == 3:
        beam_alm = hp.map2alm(beam_map)
    elif beam_map.shape[0] == 4:
        beam_alm_IQU = hp.map2alm(beam_map[:3])
        beam_alm_V = hp.map2alm(beam_map[3])
        beam_alm = np.vstack((beam_alm_IQU, beam_alm_V))
    else:
        raise ValueError(
            "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
        )

    if beam_alm.ndim == 1:
        nside = hp.get_nside(beam_map)
    else:
        nside = hp.get_nside(beam_map[0])

    n_data = len(LST_deg_list)
    n_pixels = len(pixel_indices)
    print(f"Number of data points: {n_data}")
    print(f"Number of selected pixels: {n_pixels}")

    if beam_map.ndim == 1:
        sky2sys = np.zeros((n_data, n_pixels))
    else:
        sky2sys = np.zeros((n_data, beam_map.shape[0], n_pixels))

    i = 0
    for LST_deg, azimuth_deg, elevation_deg in tqdm.tqdm(
        zip(LST_deg_list, azimuth_deg_list, elevation_deg_list), total=n_data
    ):
        beam_pointed = pointing_beam_in_eq_sys(
            beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, nside=nside
        )
        if beam_map.ndim == 1:
            if normalize:
                norm = np.sum(beam_pointed[pixel_indices])
                if norm > 0:
                    beam_pointed = beam_pointed / norm
                    sky2sys[i, :] = beam_pointed[pixel_indices]
                else:
                    print("Warning: Beam normalization factor is zero!")
            else:
                sky2sys[i, :] = beam_pointed[pixel_indices]
        else:
            if normalize:
                norm = np.sum(beam_pointed[0, pixel_indices])
                if norm > 0:
                    beam_pointed = beam_pointed / norm
                    sky2sys[i, :, :] = beam_pointed[:, pixel_indices]
                else:
                    print("Warning: Beam normalization factor is zero!")
            else:
                sky2sys[i, :, :] = beam_pointed[:, pixel_indices]
        i += 1

    result = sky2sys.reshape(i, -1)  # shape: ntime x (npol * npix)

    # # Debugging: print the shape of the result matrix
    # print(f"Sky-to-Tsys projection matrix shape: {result.shape}")
    # # Check the rank of the result matrix
    # rank = np.linalg.matrix_rank(result)
    # print(f"Rank of the projection matrix: {rank}")
    return result
