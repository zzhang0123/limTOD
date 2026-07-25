"""
limTOD: Time-Ordered Data Simulator for single-dish radio telescopes.

For comprehensive documentation, see README.md.
"""

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

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

logger = logging.getLogger(__name__)


# Enhanced type aliases for better code readability
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]
TimeList = ArrayLike
FrequencyList = ArrayLike
AngleList = ArrayLike

# Enhanced constants with better documentation
DEFAULT_MEERKAT_LATITUDE = -30.7130  # degrees, South is negative (EarthLocation convention: N+, S-)
DEFAULT_MEERKAT_LONGITUDE = 21.4430  # degrees, East is positive (EarthLocation convention: E+, W-)
DEFAULT_MEERKAT_HEIGHT = 1054  # meters above sea level
DEFAULT_START_TIME_UTC = "2019-04-23 20:41:56.397"
DEFAULT_WHITE_NOISE_VAR = 2.5e-6  # Typical thermal noise variance
# (f0, fc, alpha) for 1/f noise — the default of TODSim.generate_TOD.
# A tuple (immutable) so the shared default cannot be mutated across calls;
# pass gain_noise_params=None to disable gain-noise generation entirely.
DEFAULT_GAIN_NOISE_PARAMS = (1.335e-5, 1.099e-3, 2)


def example_scan(
    az_s: float = -60.3, az_e: float = -42.3, dt: float = 2.0, n_repeats: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    aux = np.linspace(az_s, az_e, 111)
    azimuths = np.concatenate((aux[1:-1][::-1], aux))
    azimuths = np.tile(azimuths, n_repeats)

    # Length of TOD
    ntime = len(azimuths)
    t_list = np.arange(ntime) * dt

    return t_list, azimuths


def zyzyz2zyz(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    chi: float,
    output_degrees: bool = False,
) -> Tuple[float, float, float]:
    """
    Convert "zyzyz" angles to effective "zyz" angles.
    Input angles are in degrees.
    Output angles are in degrees if output_degrees=True, else in radians.

    "zyzyz"-rotation: R = R_z(chi) R_y(delta) * R_z(gamma) * R_y(beta) * R_z(alpha)
    "zyz"-rotation: R = R_z(phi) * R_y(theta) * R_z(psi)

    Parameters
    ----------
    alpha : float
        First "z" rotation angle in degrees.
    beta : float
        First "y" rotation angle in degrees.
    gamma : float
        Second "z" rotation angle in degrees.
    delta : float
        Second "y" rotation angle in degrees.
    chi: float
        Third "z" rotation angle in degrees.
    output_degrees : bool, optional
        If True, output angles are in degrees. Default is False (radians).

    Returns
    -------
    tuple
        A tuple containing the (psi, theta, phi) angles.
    """
    r = (
        R.from_euler("z", chi, degrees=True)
        * R.from_euler("y", delta, degrees=True)
        * R.from_euler("z", gamma, degrees=True)
        * R.from_euler("y", beta, degrees=True)
        * R.from_euler("z", alpha, degrees=True)
    )
    psi, theta, phi = r.as_euler("zyz", degrees=output_degrees)
    return psi, theta, phi


def zyz_of_pointing(
    LST_deg: float,
    lat_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    selfrot_deg: float,
) -> Tuple[float, float, float]:
    """
    This function generates the effective "zyz"-rotation angles (psi, theta, phi)
    from the pointing parameters: LST, latitude, azimuth, and elevation.
    All input angles are in degrees.
    The output angles (psi, theta, phi) are in radians, as that is the unit used by hp.rotate_alm().

    Parameters
    ----------
    LST_deg : float
        The site's Local Sidereal Time in degrees.
    lat_deg : float
        The site's latitude in degrees.
    azimuth_deg : float
        The pointing's azimuth in degrees.
    elevation_deg : float
        The pointing's elevation in degrees.
    selfrot_deg: float
        The antenna's self-rotation (with respect to the beam centre) in degrees.

    Returns
    -------
    tuple
        A tuple containing the (psi, theta, phi) angles in radians.
    """

    # Convert pointing parameters to "zyzyz" angles
    alpha = LST_deg
    beta = 90.0 - lat_deg
    gamma = (
        -azimuth_deg
    )  # Note the sign convention for azimuth: East of North is positive
    delta = elevation_deg - 90.0
    chi = selfrot_deg

    # Convert "zyzyz" angles to effective "zyz" angles
    return zyzyz2zyz(alpha, beta, gamma, delta, chi)


def generate_LSTs_deg(
    ant_latitude_deg: float,
    ant_longitude_deg: float,
    ant_height_m: float,
    time_list: TimeList,
    start_time_utc: str = "2019-04-23 20:41:56.397",
) -> np.ndarray:
    """
    Generate Local Sidereal Time (LST) values in degrees for a list of time offsets.

    Parameters
    ----------
    ant_latitude_deg : float
        Latitude of the antenna/site in degrees.
    ant_longitude_deg : float
        Longitude of the antenna/site in degrees.
    ant_height_m : float
        Height of the antenna/site in meters.
    time_list: list or array of time offsets in seconds from start_time_utc.
    start_time_utc : str
        Start time in UTC (e.g. "2019-04-23 20:41:56.397").

    Returns
    -------
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


def _check_pointing_lengths(
    LST_deg_list: AngleList,
    azimuth_deg_list: AngleList,
    elevation_deg_list: AngleList,
    selfrot_deg_list: AngleList,
) -> None:
    """Raise if the per-sample pointing arrays disagree in length (zip would
    otherwise silently truncate to the shortest)."""
    lengths = {
        "LST_deg_list": len(LST_deg_list),
        "azimuth_deg_list": len(azimuth_deg_list),
        "elevation_deg_list": len(elevation_deg_list),
        "selfrot_deg_list": len(selfrot_deg_list),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Pointing arrays must have equal lengths, got {lengths}")


def _rotate_healpix_map(
    alm: np.ndarray,
    psi_rad: float,
    theta_rad: float,
    phi_rad: float,
    nside: int,
    return_map: bool = True,
) -> np.ndarray:
    """
    Rotate a Healpix map represented by its alm coefficients using given Euler angles (psi, theta, phi).
    The rotation is performed in-place on a copy of the alm coefficients.

    Parameters
    ----------
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

    Returns
    -------
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
        # 4-row input: rotate I,Q,U jointly (spin-weighted), then rotate the
        # V component separately as a scalar field.
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

def _normalize_map(input_map: np.ndarray) -> np.ndarray:
    """
    Normalize a Healpix map to have a sum value of 1.

    Parameters
    ----------
    input_map : array
        The Healpix map to be normalized.

    Returns
    -------
    array
        The normalized Healpix map.

    Raises
    ------
    ValueError
        If the map sums to (numerically) zero — dividing would silently
        fill the pipeline with NaN/Inf samples.
    """
    total = np.sum(input_map)
    if not np.isfinite(total) or np.abs(total) < np.finfo(np.float64).tiny:
        raise ValueError(
            "Cannot normalize a beam map whose pixel sum is zero or non-finite "
            f"(sum={total!r}); check the beam model, mask, and truncation "
            "settings for this pointing."
        )
    return input_map / total

def _truncate_map(input_map: np.ndarray, frac_thres: float = 1e-10) -> np.ndarray:
    """
    Truncate a Healpix map by setting all pixels with values below a certain fraction of the maximum pixel value to zero.

    Parameters
    ----------
    input_map : array
        The Healpix map to be truncated.
    frac_thres : float, optional
        The fractional threshold value for beam truncation.
        If specified, set all pixels with values below this fraction of the maximum pixel value to zero. Default is 0.0.

    Returns
    -------
    array
        The truncated Healpix map.
    """
    result = input_map.copy()
    if frac_thres > 0.0:
        if result.ndim == 1:
            max_val = np.max(result)
            threshold = frac_thres * max_val
            result[result < threshold] = 0.0
        else:
            # Use Stokes I (first row) to determine the pixel mask
            max_val = np.max(result[0])
            threshold = frac_thres * max_val
            mask = result[0] < threshold
            result[:, mask] = 0.0
    return result

def pointing_beam_in_eq_sys(
    beam_alm: np.ndarray,
    LST_deg: float,
    lat_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    selfrot_deg: float,
    nside: int,
    normalize: bool = True,
    horizontal_mask: Optional[np.ndarray] = None,
    truncate_frac_thres: float = 1e-10
) -> np.ndarray:
    """
    Point the beam in the equatorial coordinate system.

    The beam's native orientation follows the limTOD beam convention
    (docs/theory.md): boresight at the map north pole; the phi = 0
    meridian is carried to the direction of increasing elevation and
    phi = 90 deg to the direction of increasing azimuth at the pointing,
    rotated further by selfrot_deg about the boresight.

    Parameters
    ----------
    beam_alm : array
        The alm coefficients of the beam in its native orientation.
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
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
    normalize : bool, optional
        If True, normalize the pointed beam map to have a sum of 1.
        For Stokes Q, U, V parameters, they will be scaled by the same factor as Stokes I. Default is True.
    horizontal_mask : array, optional
        A Healpix map (1D array) representing a horizontal mask in the local horizontal coordinate system.
            If provided, the mask will be rotated to the equatorial coordinate system and applied to the pointed beam map before normalization. Default is None (no mask).
            1 indicates unmasked pixels, 0 indicates masked pixels. 
    truncate_frac_thres : float, optional
        The fractional threshold value for (rotated-)beam truncation.
        If specified, set all pixels with values below this fraction of the maximum pixel value to zero before normalization. Default is 1e-10.

    Returns
    -------
    array
        The pointed beam map in the equatorial coordinate system.
    """
    psi_rad, theta_rad, phi_rad = zyz_of_pointing(
        LST_deg=LST_deg,
        lat_deg=lat_deg,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        selfrot_deg=selfrot_deg
    )
    beam_pointed = _rotate_healpix_map(beam_alm, psi_rad, theta_rad, phi_rad, nside)

    if horizontal_mask is not None:
        # The horizontal-frame mask has its pole at the ZENITH (like beams
        # have theirs at boresight), so horizontal -> equatorial is the
        # zenith pointing: elevation=90 makes delta = el-90 vanish and the
        # chain reduce to the pure LST/lat transform (psi'=LST, theta'=
        # 90-lat, phi'=0). elevation=0 here was a bug that tipped the mask
        # 90 deg onto the horizon (fixed in v1.3.0; regression-tested).
        mask_psi_rad, mask_theta_rad, mask_phi_rad = zyz_of_pointing(
            LST_deg=LST_deg,
            lat_deg=lat_deg,
            azimuth_deg=0,
            elevation_deg=90,
            selfrot_deg=0
        )
        mask_alm = hp.map2alm(horizontal_mask)
        mask_pointed = _rotate_healpix_map(mask_alm, mask_psi_rad, mask_theta_rad, mask_phi_rad, nside)
        mask_pointed = np.where(mask_pointed >= 0.5, 1.0, 0.0)
        beam_pointed = beam_pointed * mask_pointed[np.newaxis, :] if beam_pointed.ndim == 2 else beam_pointed * mask_pointed
    else:
        beam_pointed = _truncate_map(beam_pointed, frac_thres=truncate_frac_thres)
    if normalize:
        if beam_pointed.ndim == 1:
            beam_pointed = _normalize_map(beam_pointed)
        elif beam_pointed.shape[0] in [3, 4]:
            # _normalize_map raises on a zero-sum Stokes I, so norm_factor
            # is guaranteed usable for the remaining Stokes rows.
            norm_factor = np.sum(beam_pointed[0])
            beam_pointed[0] = _normalize_map(beam_pointed[0])
            beam_pointed[1:] = beam_pointed[1:] / norm_factor
        else:
            raise ValueError(
                "Pointed beam map must be a 1D array or a 2D array with 3 or 4 rows."
            )
    return beam_pointed


def _beam_weighted_sum(
    beam_map: np.ndarray, sky_map: np.ndarray, normalize: bool = False
) -> float:
    """
    Compute the beam-weighted sum of the sky map.

    Parameters
    ----------
    beam_map : array
        The Healpix map of the beam (should be normalized).
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
    sky_map : array
        The Healpix map of the sky.
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
    normalize : bool, optional
        If True, normalize the Stokes-I beam map to have a sum of 1 before computing the weighted sum.
        All other Stokes parameters (Q,U,V) will be scaled by the same factor.

    Returns
    -------
    float
        The beam-weighted sum of the sky map.
    """
    if normalize:
        # Create a copy to avoid modifying the input array
        beam_map = beam_map.copy()
        if beam_map.ndim == 1:
            beam_map = _normalize_map(beam_map)
        elif beam_map.shape[0] in [3, 4]:
            # _normalize_map raises on a zero-sum Stokes I, so norm_factor
            # is guaranteed usable for the remaining Stokes rows.
            norm_factor = np.sum(beam_map[0])
            beam_map[0] = _normalize_map(beam_map[0])
            beam_map[1:] = beam_map[1:] / norm_factor
        else:
            raise ValueError(
                "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
            )

    # For multi-row (Stokes) inputs this sums I_b*I_s + Q_b*Q_s + U_b*U_s
    # (+ V_b*V_s) over pixels — a total-detected-power convention with unit
    # Mueller response and no polarization leakage terms.
    return np.sum(beam_map * sky_map)


def generate_TOD_sky(
    beam_map: np.ndarray,
    sky_map: np.ndarray,
    LST_deg_list: AngleList,
    lat_deg: float,
    azimuth_deg_list: AngleList,
    elevation_deg_list: AngleList,
    selfrot_deg_list: AngleList,
    nside_hires: Optional[int] = None,
    normalize_beam: bool = False,
    horizontal_mask: Optional[np.ndarray] = None,
    truncate_frac_thres: float = 1e-10
) -> np.ndarray:
    """
    Generate Time-Ordered Data (TOD) by simulating observations of a sky map with a given beam pattern.
    Note that the TOD represents the beam-weighted sum of the sky map at each pointing.

    Parameters
    ----------
    beam_map : array
        The Healpix map of the beam pattern, in the limTOD beam
        convention (docs/theory.md): boresight at the map north pole;
        phi = 0 -> direction of increasing elevation at the pointing,
        phi = 90 deg -> direction of increasing azimuth.
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
    sky_map : array
        The Healpix map of the sky.
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
    LST_deg_list : array
        List of Local Sidereal Time values in degrees for each observation.
    lat_deg : float
        The latitude of the observation site in degrees.
    azimuth_deg_list : array
        List of azimuth values in degrees for each observation.
    elevation_deg_list : array
        List of elevation values in degrees for each observation.
    selfrot_deg_list : array
        List of self-rotation values in degrees for each observation.
    nside_hires : int, optional
        If provided, upgrade the beam map to this nside before processing.
        This can help improve accuracy when the beam is narrow. Default is None.
    normalize_beam : bool, optional
        If True, normalize the beam map to have a sum of 1 before computing the weighted sum.
        Default is False.
    horizontal_mask : array, optional
        A Healpix map (1D array) representing a horizontal mask in the local horizontal coordinate system.
            If provided, the mask will be rotated to the equatorial coordinate system and applied to the pointed beam map before normalization. Default is None (no mask).
            1 indicates unmasked pixels, 0 indicates masked pixels.
    truncate_frac_thres : float, optional
        The fractional threshold value for beam truncation. 
        It is ignored if horizontal_mask is provided.
        If specified, and horizontal_mask is provided, set all pixels with values below this fraction of the maximum pixel value to zero before normalization. Default is 1e-10.

    Returns
    -------
    array
        The generated Time-Ordered Data (TOD) as a 1D array.
    """
    
    nside_beam = hp.get_nside(beam_map) if beam_map.ndim == 1 else hp.get_nside(beam_map[0])
    if nside_hires is not None and nside_hires != nside_beam:
        beam_map_hires = hp.ud_grade(beam_map, nside_hires)
    else:
        beam_map_hires = beam_map
    nside_sky = hp.get_nside(sky_map) if sky_map.ndim == 1 else hp.get_nside(sky_map[0])
    _check_pointing_lengths(LST_deg_list, azimuth_deg_list, elevation_deg_list, selfrot_deg_list)

    # Convert beam map to alm coefficients
    if beam_map_hires.ndim == 1 or beam_map_hires.shape[0] == 3:
        beam_alm = hp.map2alm(beam_map_hires)
    elif beam_map_hires.shape[0] == 4:
        beam_alm_IQU = hp.map2alm(beam_map_hires[:3])
        beam_alm_V = hp.map2alm(beam_map_hires[3])
        beam_alm = np.vstack((beam_alm_IQU, beam_alm_V))
    else:
        raise ValueError(
            "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
        )

    tod = []

        

    for LST_deg, azimuth_deg, elevation_deg, selfrot_deg in tqdm.tqdm(
        zip(LST_deg_list, azimuth_deg_list, elevation_deg_list, selfrot_deg_list), total=len(LST_deg_list)
    ):
        beam_pointed = pointing_beam_in_eq_sys(
            beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, selfrot_deg,
            nside=nside_sky, normalize=normalize_beam, 
            horizontal_mask=horizontal_mask,
            truncate_frac_thres=truncate_frac_thres
        )
        sample = _beam_weighted_sum(beam_pointed, sky_map)
        tod.append(sample)

    return np.array(tod)


def example_beam_map(
    *, freq: float, nside: int, FWHM_major: float = 1.1, FWHM_minor: float = 1.1
) -> np.ndarray:
    """
    Generate an example Gaussian beam map.
    This toy model is achromatic.

    FWHM_major: major axis FWHM in degrees, along the beam-map meridian
    phi = 0 — carried to the direction of increasing elevation under
    limTOD's beam coordinate convention (docs/theory.md).
    FWHM_minor: minor axis FWHM in degrees, along phi = 90 deg —
    carried to the direction of increasing azimuth.
    """
    NPIX = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(NPIX))
    # Convert FWHM to sigma (radians)
    sigma_major = np.radians(FWHM_major / (2 * np.sqrt(2 * np.log(2))))
    sigma_minor = np.radians(FWHM_minor / (2 * np.sqrt(2 * np.log(2))))
    angle_rad = 0.0
    # Offsets from the beam center (at the pole) in Cartesian projection
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    # Rotate by angle
    x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    # Elliptical Gaussian
    beam_map = np.exp(-0.5 * ((x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2))
    # Normalize
    beam_map /= np.sum(beam_map)
    # hp.mollview(beam_map, title="Elliptical (Asymmetric) Beam Map")
    return beam_map


def example_symm_beam_map(*, freq: float, nside: int, FWHM: float = 1.1) -> np.ndarray:
    """
    Generate a symmetric Gaussian beam map centered at the pole.

    Parameters
    ----------
    freq : float
        Frequency (not used in this achromatic model, but kept for API consistency).
    nside : int
        HEALPix nside parameter.
    FWHM : float, optional
        Full Width at Half Maximum of the Gaussian beam in degrees. Default is 1.1.

    Returns
    -------
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
        ant_latitude_deg: float = -30.7130,
        ant_longitude_deg: float = 21.4430,
        ant_height_m: float = 1054,
        beam_func: Callable[..., np.ndarray] = example_beam_map,
        sky_func: Callable[..., np.ndarray] = sky_model.GDSM_sky_model,
        beam_nside: int = 256,
        sky_nside: int = 256,
    ) -> None:
        """
        Initialize the limTODsim class.

        Parameters
        ----------
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
            Orientation (limTOD beam convention, see docs/theory.md):
            boresight at the map's north pole (theta = 0); for a pointing
            (az, el) the meridian phi = 0 is carried to the direction of
            increasing elevation (e_el = d b_hat / d el) and phi = 90 deg
            to the direction of increasing azimuth (e_az); positive
            selfrot rotates the pattern from e_el toward e_az. Pinned by
            tests/test_beam_orientation.py.
        sky_func : function
            Function that takes frequency and nside as keyword input and returns the sky map.
            The output map can be:
                a single array is considered I,
                array with 3 rows:[I,Q,U]
                array with 4 rows:[I,Q,U,V]
        beam_nside : int, optional
            The nside parameter for the beam Healpix maps.
            Note: beam_nside and sky_nside can be different. 
            beam_nside is used when generating the beam map, it should be large enough to resolve the beam features.
        sky_nside : int, optional
            The nside parameter for the sky Healpix maps.
            It decides how the sky map is parametrized.
        """
        self.ant_latitude_deg = ant_latitude_deg
        self.ant_longitude_deg = ant_longitude_deg
        self.ant_height_m = ant_height_m
        self.beam_nside = beam_nside
        self.sky_nside = sky_nside
        self.beam_func = beam_func
        self.sky_func = sky_func

    def simulate_sky_TOD(
        self,
        freq_list: FrequencyList,
        time_list: TimeList,
        azimuth_deg_list: AngleList,
        elevation_deg: Union[float, AngleList],
        selfrot_deg_list: Optional[AngleList] = None,
        start_time_utc: str = "2019-04-23 20:41:56.397",
        return_LSTs: bool = False,
        nside_hires: Optional[int] = None,
        normalize_beam: bool = False,
        horizontal_mask: Optional[np.ndarray] = None,
        truncate_frac_thres: float = 1e-10
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate sky TOD (beam-weighted sum of sky map) for a list of frequencies and time offsets.

        Parameters
        ----------
        freq_list : list or array
            List of frequencies.
        time_list : list or array
            List of time offsets in seconds (for each observation time) from start_time_utc.
        azimuth_deg_list : list or array
            List of azimuth values in degrees for each observation.
        elevation_deg : list or float
            If float: Elevation value in degrees universal for all observations.
            If list: List of elevation values in degrees for each observation.
        selfrot_deg_list : list or array, optional
            Antenna self-rotation angles in degrees for each observation.
            Default is None (zero self-rotation).
        start_time_utc : str
            Start time in UTC (e.g. "2019-04-23 20:41:56.397").
        nside_hires : int, optional
            The nside parameter for high-resolution Healpix maps used in the simulation.
        normalize_beam : bool, optional
            If True, normalize the beam map to have a sum of 1 before computing the weighted sum.
            Default is False.
        horizontal_mask : array, optional
            Binary HEALPix mask in the local horizontal coordinate system
            (1 = keep, 0 = masked), rotated with the pointing and applied to
            the pointed beam. Default is None (no mask).
        truncate_frac_thres : float, optional
            The fractional threshold value for beam truncation.
            If specified, set all pixels with values below this fraction of the maximum pixel value to zero before normalization. Default is 1e-10.
        return_LSTs : bool, optional
            If True, return the LST values along with the TODs. Default is False.

        Returns
        -------
        TOD_array : array
            The simulated sky TOD array with shape (nfreq, ntime).
            Each element represents the simulated TOD sample.

        LST_deg_list : array, optional
            If return_LSTs is True, also return the array of LST values in degrees.
        """
        # the beam-weighted sum of the sky map at each pointing.
        #
        # Parallel strategy: partition the *time* axis contiguously across
        # MPI ranks. Each rank computes its local time slice for every
        # frequency (outer serial loop over freq_list, inner parallel slice
        # over time), then allgathers along the time axis so every rank
        # returns the full (nfreq, ntime) array. Time is always the
        # dominant axis (typically thousands of samples) while nfreq is
        # often 1, which is why partitioning here beats the previous
        # freq-axis parallelism.

        ntime = len(time_list)
        nfreq = len(freq_list)

        if isinstance(elevation_deg, (int, float)):
            elevation_deg_list: np.ndarray = np.full(ntime, elevation_deg)
        else:
            elevation_deg_list = np.asarray(elevation_deg)

        if selfrot_deg_list is None:
            selfrot_deg_list = np.zeros(ntime)
        else:
            selfrot_deg_list = np.asarray(selfrot_deg_list)

        azimuth_deg_list = np.asarray(azimuth_deg_list)

        LST_deg_list = generate_LSTs_deg(
            self.ant_latitude_deg,
            self.ant_longitude_deg,
            self.ant_height_m,
            time_list,
            start_time_utc=start_time_utc,
        )

        # Contiguous time-axis partition. In serial (size == 1) this
        # returns all ntime indices on the single rank.
        idx_mine = np.asarray(
            mpiutil.partition_list_mpi(list(range(ntime)), method="con"),
            dtype=int,
        )
        n_local = len(idx_mine)

        lst_mine = LST_deg_list[idx_mine]
        az_mine = azimuth_deg_list[idx_mine]
        el_mine = elevation_deg_list[idx_mine]
        sr_mine = selfrot_deg_list[idx_mine]

        TOD_local = np.empty((nfreq, n_local), dtype=np.float64)
        for i, freq in enumerate(freq_list):
            beam_map = self.beam_func(freq=freq, nside=self.beam_nside)
            sky_map = self.sky_func(freq=freq, nside=self.sky_nside)
            TOD_local[i] = generate_TOD_sky(
                beam_map,
                sky_map,
                lst_mine,
                self.ant_latitude_deg,
                az_mine,
                el_mine,
                sr_mine,
                nside_hires=nside_hires,
                normalize_beam=normalize_beam,
                horizontal_mask=horizontal_mask,
                truncate_frac_thres=truncate_frac_thres,
            )

        if mpiutil.size == 1:
            TOD_array = TOD_local
        else:
            # allgather preserves the "all ranks return the full array"
            # contract of this method. Concatenation along the last axis
            # reassembles the global time order because partition_list_mpi
            # hands out contiguous slices in rank order.
            pieces = mpiutil.world.allgather(TOD_local)
            TOD_array = np.concatenate(pieces, axis=-1)

        if return_LSTs:
            return TOD_array, LST_deg_list
        return TOD_array

    def generate_TOD(
        self,
        freq_list: FrequencyList,
        time_list: TimeList,
        azimuth_deg_list: AngleList,
        selfrot_deg_list: Optional[AngleList] = None,
        elevation_deg: Union[float, AngleList] = 41.5,
        start_time_utc: str = "2019-04-23 20:41:56.397",
        Tsys_others_TOD: Optional[Union[float, np.ndarray]] = None,
        background_gain_TOD: Optional[Union[float, np.ndarray]] = None,
        # Typed Any (not Optional[float | ndarray]): the rank-0/bcast dance
        # below means mypy cannot prove it is non-None where it is consumed.
        gain_noise_TOD: Any = None,
        gain_noise_params: Optional[Sequence[float]] = DEFAULT_GAIN_NOISE_PARAMS,
        white_noise_var: Optional[float] = None,
        return_LSTs: bool = False,
        nside_hires: Optional[int] = None,
        normalize_beam: bool = False,
        horizontal_mask: Optional[np.ndarray] = None,
        truncate_frac_thres: float = 1e-10
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float]],
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float], np.ndarray],
    ]:
        """
        Generate overall TOD including sky signal and other components.

        Data model:
        overall_TOD = background_gain_TOD * (1 + gain_noise_TOD) * (sky_TOD + Tsys_others_TOD) * (1 + white_noise_TOD)

        Parameters
        ----------
        freq_list : list or array
            List of frequencies.
        time_list : list or array
            List of time offsets in seconds (for each observation time) from start_time_utc.
        azimuth_deg_list : list or array
            List of azimuth values in degrees for each observation.
        elevation_deg : list or float
            If float: Elevation value in degrees universal for all observations.
            If list: List of elevation values in degrees for each observation.
        selfrot_deg_list : list or array, optional
            Antenna self-rotation angles in degrees for each observation.
            Default is None (zero self-rotation).
        start_time_utc : str
            Start time in UTC (e.g. "2019-04-23 20:41:56.397").
        Tsys_others_TOD : array, optional
            Array of residual system temperature TOD (shape: nfreq x ntime). Default is None (no residual).
        background_gain_TOD : array, optional
            Array of background gain TOD (shape: nfreq x ntime). Default is None (unity gain).
        gain_noise_TOD : array, optional
            Array of gain noise TOD (shape: nfreq x ntime). Default is None (no gain noise).
        gain_noise_params : tuple or list, optional
            Parameters (f0, fc, alpha) for generating gain noise if gain_noise_TOD is None.
            Default is (1.335e-5, 1.099e-3, 2); pass None to disable gain noise.
        white_noise_var : float, optional
            Variance of white noise to be added. Default is None (uses default value of 2.5e-6).
        return_LSTs : bool, optional
            If True, return the LST values along with the TODs. Default is False.
        nside_hires : int, optional
            If provided, upgrade the beam map to this nside before processing.
            This can help improve accuracy when the beam is narrow. Default is None.
        normalize_beam : bool, optional
            If True, normalize the beam map to have a sum of 1 before computing the weighted sum.
            Default is False.
        horizontal_mask : array, optional
            Binary HEALPix mask in the local horizontal coordinate system
            (1 = keep, 0 = masked), rotated with the pointing and applied to
            the pointed beam. Default is None (no mask).
        truncate_frac_thres : float, optional
            The fractional threshold value for beam truncation.
            If specified, set all pixels with values below this fraction of the maximum pixel value to zero before normalization. Default is 1e-10.

        Returns
        -------
        overall_TOD : array
            The overall generated TOD (shape: nfreq x ntime).
        sky_TOD : array
            The sky signal TOD (shape: nfreq x ntime).
        gain_noise_TOD : array
            The gain noise TOD (shape: nfreq x ntime).
        LST_deg_list : array, optional
            Only when return_LSTs is True (making the return a 4-tuple).
        """

        nfreq = len(freq_list)
        ntime = len(time_list)

        if Tsys_others_TOD is None:
            Tsys_others_TOD = 0.0
        if background_gain_TOD is None:
            background_gain_TOD = 1.0
        if white_noise_var is None:
            if mpiutil.rank0:
                logger.info(
                    "No white noise variance specified; using the default "
                    "2.5e-6 (dimensionless fractional noise)."
                )
            white_noise_var = 2.5e-6

        # Generate 1/f gain noise on rank 0 only, then broadcast to every
        # rank. The 1/f sequence is correlated in time, so every rank must
        # see the *same* realisation — a per-rank re-draw would give a
        # rank-dependent overall_TOD and silently corrupt MPI runs.
        if gain_noise_TOD is None:
            if gain_noise_params is None:
                gain_noise_TOD = 0.0
            else:
                if mpiutil.rank0:
                    logger.info(
                        "Generating gain noise with parameters f0=%s, fc=%s, "
                        "alpha=%s (1/f realizations are uncorrelated across "
                        "frequencies).",
                        gain_noise_params[0], gain_noise_params[1],
                        gain_noise_params[2],
                    )
                    f0, fc, alpha = gain_noise_params
                    gain_noise_TOD = sim_noise(
                        f0, fc, alpha, cast(np.ndarray, time_list),
                        n_samples=nfreq,
                        white_n_variance=0.0,
                    )
                else:
                    gain_noise_TOD = None
                if mpiutil.size > 1:
                    gain_noise_TOD = mpiutil.world.bcast(
                        gain_noise_TOD, root=0)
        elif gain_noise_params is not None:
            if mpiutil.rank0:
                logger.warning(
                    "Both gain_noise_TOD and gain_noise_params were provided; "
                    "ignoring gain_noise_params."
                )

        # White noise follows the same rank-0-then-bcast pattern. Without
        # the broadcast each rank draws from its own (different) RNG state
        # and overall_TOD would differ by rank after the multiplicative
        # combine below.
        white_noise_TOD: Any  # non-rank0 implies size > 1, so the bcast below always replaces the None
        if mpiutil.rank0:
            white_noise_TOD = np.random.normal(
                0, np.sqrt(white_noise_var), size=(nfreq, ntime),
            )
        else:
            white_noise_TOD = None
        if mpiutil.size > 1:
            white_noise_TOD = mpiutil.world.bcast(white_noise_TOD, root=0)

        sky_TOD, LST_deg_list = self.simulate_sky_TOD(
            freq_list,
            time_list,
            azimuth_deg_list,
            elevation_deg,
            selfrot_deg_list=selfrot_deg_list,
            start_time_utc=start_time_utc,
            return_LSTs=True,
            nside_hires=nside_hires,
            normalize_beam=normalize_beam,
            horizontal_mask=horizontal_mask,
            truncate_frac_thres=truncate_frac_thres
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
    beam_map: np.ndarray,
    LST_deg_list: AngleList,
    lat_deg: float,
    azimuth_deg_list: AngleList,
    elevation_deg_list: AngleList,
    selfrot_deg_list: AngleList,
    horizontal_mask: Optional[np.ndarray] = None,
    threshold: float = 0.0111,
    nside_hires: Optional[int] = None,
    nside_target: Optional[int] = None,
) -> np.ndarray:
    """
    Generate the selected pixel indices based on beam sensitivity.
    The selected pixels are those with beam response above a given threshold in the stacked abs(beam) map.

    Parameters
    ----------
    beam_map : array
        The Healpix map of the beam pattern.
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
    LST_deg_list : array
        List of Local Sidereal Time values in degrees for each observation.
    lat_deg : float
        The latitude of the observation site in degrees.
    azimuth_deg_list : array
        List of azimuth values in degrees for each observation.
    elevation_deg_list : array
        List of elevation values in degrees for each observation.
    selfrot_deg_list : array
        List of self-rotation values in degrees for each observation.
    horizontal_mask : array, optional
        A Healpix map (1D array) representing a horizontal mask in the local horizontal coordinate system. If provided, the mask will be rotated to the equatorial coordinate system and applied to the pointed beam map before normalization. Default is None (no mask).
        1 indicates unmasked pixels, 0 indicates masked pixels.
    threshold : float
        The threshold to cut off the fractional beam response np.abs(beam[pixel])/beam_max, default is 0.0111.
        e.g., if threshold=0.01, only pixels with beam response larger than 1% of the maximum will be considered.
        Note that this is the threshold for singling out pixels.
    nside_hires : int, optional
        If provided, upgrade the beam map to this nside before processing.
        This can help improve accuracy when the beam is narrow. Default is None.
    nside_target : int, optional
        The target nside for the output beam map. This should match the convention used in pixel_indices.   

    Returns
    -------
    pixel_indices : array
        The selected pixel indices based on the beam sensitivity.
    """

    nside_beam = hp.get_nside(beam_map) if beam_map.ndim == 1 else hp.get_nside(beam_map[0])
    if nside_hires is not None and nside_hires != nside_beam:
        beam_map_hires = hp.ud_grade(beam_map, nside_hires)
    else:
        beam_map_hires = beam_map

    if nside_target is None:
        nside_target = nside_beam
        if mpiutil.rank0:
            logger.info("nside_target not provided; using the beam map nside as target.")


    # Convert beam map to alm coefficients
    if beam_map_hires.ndim == 1 or beam_map_hires.shape[0] == 3:
        beam_alm = hp.map2alm(beam_map_hires)
    elif beam_map_hires.shape[0] == 4:
        beam_alm_IQU = hp.map2alm(beam_map_hires[:3])
        beam_alm_V = hp.map2alm(beam_map_hires[3])
        beam_alm = np.vstack((beam_alm_IQU, beam_alm_V))
    else:
        raise ValueError(
            "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
        )


    _check_pointing_lengths(LST_deg_list, azimuth_deg_list, elevation_deg_list, selfrot_deg_list)

    # Integrate the beam map as the sum map, select pixels above threshold

    if mpiutil.rank0:
        logger.info("Step 1: generating the stacked abs(beam) map ...")
    # Generate a initial boolean map with all pixels zero
    bool_map = np.zeros(hp.nside2npix(nside_target), dtype=bool)

    for LST_deg, azimuth_deg, elevation_deg, selfrot_deg in tqdm.tqdm(
        zip(LST_deg_list, azimuth_deg_list, elevation_deg_list, selfrot_deg_list), total=len(LST_deg_list)
    ):
        beam_pointed = pointing_beam_in_eq_sys(
            beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, selfrot_deg,
            horizontal_mask=horizontal_mask,
            nside=nside_target
        )
        norm = np.max(np.abs(beam_pointed))
        if norm > 0:
            beam_pointed = beam_pointed / norm
            bool_map = np.logical_or(bool_map, beam_pointed > threshold)
        else:
            logger.warning("Beam has zero maximum value at this pointing.")

    if mpiutil.rank0:
        logger.info("Step 2: selecting pixels above threshold sensitivity ...")
    if bool_map.ndim == 2:
        bool_map = np.any(bool_map, axis=0)
    pixel_indices = np.where(bool_map)[0]

    return pixel_indices


def generate_sky2sys_projection(
    beam_map: np.ndarray,
    LST_deg_list: AngleList,
    lat_deg: float,
    azimuth_deg_list: AngleList,
    elevation_deg_list: AngleList,
    selfrot_deg_list: AngleList,
    pixel_indices: np.ndarray,
    horizontal_mask: Optional[np.ndarray] = None,
    normalize_beam: bool = False,
    nside_hires: Optional[int] = None,
    nside_target: Optional[int] = None,
    truncate_frac_thres: float = 1e-10
) -> np.ndarray:
    """
    Generate the sky-to-Tsys projection matrix and the selected pixel indices based on beam sensitivity.
    The projection matrix maps the sky pixels to the system temperature.

    Parameters
    ----------
    beam_map : array
        The Healpix map of the beam pattern.
        Input can be a single array (Stokes I), an array
        with 3 rows [I, Q, U], or with 4 rows [I, Q, U, V].
    LST_deg_list : array
        List of Local Sidereal Time values in degrees for each observation.
    lat_deg : float
        The latitude of the observation site in degrees.
    azimuth_deg_list : array
        List of azimuth values in degrees for each observation.
    elevation_deg_list : array
        List of elevation values in degrees for each observation.
    selfrot_deg_list : array
        List of self-rotation values in degrees for each observation.
    pixel_indices : array
        The selected pixel indices based on the beam sensitivity.
    horizontal_mask : array, optional
        A Healpix map (1D array) representing a horizontal mask in the local horizontal coordinate
        system. If provided, the mask will be rotated to the equatorial coordinate system and applied to the pointed beam map before normalization. Default is None (no mask).
        1 indicates unmasked pixels, 0 indicates masked pixels.
    normalize_beam : bool, optional
        If True, normalize the beam map to have a sum of 1 before processing.
        Default is False.
    truncate_frac_thres : float, optional
        The fractional threshold value for beam truncation. 
        If specified, set all pixels with values below this fraction of the maximum pixel value to zero before normalization. Default is 1e-10.
    nside_hires : int, optional
        If provided, upgrade the beam map to this nside before processing.
        This can help improve accuracy when the beam is narrow. Default is None.
    nside_target : int, optional
        The target nside for the output beam map. This should match the convention used in pixel_indices.

    Returns
    -------
    array
        The generated Time-Ordered Data (TOD) as a 1D array.
    """
    nside_beam = hp.get_nside(beam_map) if beam_map.ndim == 1 else hp.get_nside(beam_map[0])
    if nside_hires is not None and nside_hires != nside_beam:
        beam_map_hires = hp.ud_grade(beam_map, nside_hires)
    else:
        beam_map_hires = beam_map
    if nside_target is None:
        nside_target = nside_beam

    # Convert beam map to alm coefficients
    if beam_map_hires.ndim == 1 or beam_map_hires.shape[0] == 3:
        beam_alm = hp.map2alm(beam_map_hires)
    elif beam_map_hires.shape[0] == 4:
        beam_alm_IQU = hp.map2alm(beam_map_hires[:3])
        beam_alm_V = hp.map2alm(beam_map_hires[3])
        beam_alm = np.vstack((beam_alm_IQU, beam_alm_V))
    else:
        raise ValueError(
            "Input beam_map must be a 1D array or a 2D array with 3 or 4 rows."
        )

    _check_pointing_lengths(LST_deg_list, azimuth_deg_list, elevation_deg_list, selfrot_deg_list)

    n_data = len(LST_deg_list)
    n_pixels = len(pixel_indices)
    if mpiutil.rank0:
        logger.info("Number of data points: %d", n_data)
        logger.info("Number of selected pixels: %d", n_pixels)

    if beam_map.ndim == 1:
        sky2sys: np.ndarray = np.zeros((n_data, n_pixels))
    else:
        sky2sys = np.zeros((n_data, beam_map.shape[0], n_pixels))

    i = 0
    for LST_deg, azimuth_deg, elevation_deg, selfrot_deg in tqdm.tqdm(
        zip(LST_deg_list, azimuth_deg_list, elevation_deg_list, selfrot_deg_list), total=n_data
    ):
        beam_pointed = pointing_beam_in_eq_sys(
            beam_alm, LST_deg, lat_deg, azimuth_deg, elevation_deg, selfrot_deg,
            horizontal_mask=horizontal_mask,
            nside=nside_target, normalize=normalize_beam,
            truncate_frac_thres=truncate_frac_thres
        )

        if beam_map.ndim == 1:
            sky2sys[i, :] = beam_pointed[pixel_indices]
        else:
            sky2sys[i, :, :] = beam_pointed[:, pixel_indices]
        i += 1

    result = sky2sys.reshape(n_data, -1)  # shape: ntime x (npol * npix)

    return result
