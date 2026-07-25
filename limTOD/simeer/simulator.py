"""
Drop-in replacement for :class:`limTOD.TODSim` whose sky-TOD step uses
the MeerKLASS-native (l, m) beam path instead of HEALPix spherical
harmonic rotation.

The class inherits everything else from limTOD: LST generation, gain
noise, flicker noise, white noise, and the overall TOD assembly via
``generate_TOD``. Only :meth:`simulate_sky_TOD` is overridden. Users who
only need the lower-level :mod:`limTOD.simeer.sky_integrator` API can
bypass this class entirely.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

from limTOD.simulator import (
    AngleList,
    FrequencyList,
    TimeList,
    TODSim as _BaseTODSim,
    generate_LSTs_deg,
)

from .beam import MeerKLASSBeam
from .sky_integrator import integrate_tod, materialize_sky_cube

SkyFunc = Callable[..., np.ndarray]


class SimeerTODSim(_BaseTODSim):
    """MeerKLASS-aware TOD simulator.

    Inherits gain/noise/white-noise modelling and the overall
    ``generate_TOD`` assembly from :class:`limTOD.TODSim`. Only the
    sky-signal step is replaced with the disc-based (l, m) interpolation
    path implemented in :mod:`simeer.sky_integrator`.

    Parameters
    ----------
    beam : MeerKLASSBeam
        Pre-loaded beam wrapper.
    sky_func : callable
        ``sky_func(freq=..., nside=...) -> ndarray`` of shape ``(npix,)``,
        matching the limTOD convention.
    sky_nside : int
        HEALPix resolution of the sky model.
    disc_radius_deg : float
        Radius of the integration disc. Default 8 degrees (a small
        margin beyond the +/-6 degree MeerKLASS U-band beam grid).
    polarization : str
        ``'HH'`` or ``'VV'``.
    ant_latitude_deg, ant_longitude_deg, ant_height_m : float
        Telescope site as in :class:`limTOD.TODSim`. Defaults are the
        MeerKAT values.
    n_jobs : int
        Number of joblib worker processes for the sample loop. ``1``
        runs serially.
    """

    def __init__(
        self,
        *,
        beam: MeerKLASSBeam,
        sky_func: SkyFunc,
        sky_nside: int,
        disc_radius_deg: float = 8.0,
        polarization: str = "HH",
        ant_latitude_deg: float = -30.7130,
        ant_longitude_deg: float = 21.4430,
        ant_height_m: float = 1054.0,
        n_jobs: int = 1,
    ):
        # NB: limTOD.TODSim.__init__ uses positional kwargs (beam_func,
        # sky_func, etc.). We bypass its beam_func contract and instead
        # carry our own beam wrapper. The base class is still useful for
        # the noise and gain machinery downstream.
        def _unused_beam_func(**_: object) -> np.ndarray:
            raise RuntimeError(
                "SimeerTODSim replaces the HEALPix beam_func path; this "
                "placeholder should never be called."
            )

        super().__init__(
            ant_latitude_deg=ant_latitude_deg,
            ant_longitude_deg=ant_longitude_deg,
            ant_height_m=ant_height_m,
            beam_func=_unused_beam_func,
            sky_func=sky_func,
            beam_nside=sky_nside,  # unused but limTOD validates
            sky_nside=sky_nside,
        )

        self.beam = beam
        self.disc_radius_deg = float(disc_radius_deg)
        self.polarization = polarization.upper()
        self.n_jobs = int(n_jobs)

    # ------------------------------------------------------------------ #
    # Override sky-TOD step                                              #
    # ------------------------------------------------------------------ #
    def simulate_sky_TOD(  # noqa: N802 - matches limTOD's name
        self,
        freq_list: FrequencyList,
        time_list: TimeList,
        azimuth_deg_list: AngleList,
        elevation_deg: Union[float, AngleList],
        selfrot_deg_list: Optional[AngleList] = None,
        start_time_utc: str = "2019-04-23 20:41:56.397",
        return_LSTs: bool = False,
        nside_hires: Optional[int] = None,  # accepted for API parity, ignored
        normalize_beam: bool = False,  # accepted for API parity, ignored
        horizontal_mask: Optional[np.ndarray] = None,
        truncate_frac_thres: float = 1e-10,  # accepted for API parity, ignored
        progress: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Sky-TOD integration via the Simeer (l, m) disc path.

        The signature mirrors :meth:`limTOD.TODSim.simulate_sky_TOD` so
        callers can swap one class for the other. Parameters specific to
        the HEALPix-SH path (``nside_hires``, ``normalize_beam``,
        ``truncate_frac_thres``) are accepted but ignored -- the new
        path interpolates the beam at native resolution and normalises
        through ``beam_solid_angle`` instead.
        """
        time_arr = np.asarray(time_list, dtype=np.float64)
        ntime = len(time_arr)
        if isinstance(elevation_deg, (int, float)):
            el_arr: np.ndarray = np.full(ntime, float(elevation_deg))
        else:
            el_arr = np.asarray(elevation_deg, dtype=np.float64)

        az_arr = np.asarray(azimuth_deg_list, dtype=np.float64)

        lst_deg_list = generate_LSTs_deg(
            self.ant_latitude_deg,
            self.ant_longitude_deg,
            self.ant_height_m,
            time_arr,
            start_time_utc=start_time_utc,
        )

        # Build the sky cube once (one HEALPix map per output channel).
        sky_cube = materialize_sky_cube(self.sky_func, freq_list, nside=self.sky_nside)

        tod = integrate_tod(
            lst_deg_list=lst_deg_list,
            az_deg_list=az_arr,
            el_deg_list=el_arr,
            lat_deg=self.ant_latitude_deg,
            beam=self.beam,
            sky_maps=sky_cube,
            freq_MHz=freq_list,
            disc_radius_deg=self.disc_radius_deg,
            polarization=self.polarization,
            horizontal_mask=horizontal_mask,
            n_jobs=self.n_jobs,
            progress=progress,
        )

        if return_LSTs:
            return tod, lst_deg_list
        return tod
