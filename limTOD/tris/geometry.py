"""Pointing geometry for the parked-zenith TRIS drift scan.

Everything here was checked numerically against ``limTOD.simulator`` rather
than taken on trust (see ``docs/tris.md`` for the measurements):

* the boresight lands at ``RA = LST``, ``dec = latitude`` for every LST;
* with ``selfrot = 0`` the intrinsic **E** plane (``phi = 0/180``) lands
  exactly on the meridian (position angle 180/0 deg) and the **H** plane
  (``phi = 90/270``) exactly east-west;
* with ``selfrot = -7`` both rotate by ``+7`` degrees in position angle, i.e.
  the E plane sits 7 degrees **east** of the meridian -- which is what
  ``TRIS_Beam_Profile.txt`` and both ring headers state.

Because the E plane is the narrow one (19.2 deg vs 23.4 deg), the TRIS beam is
narrow in declination and wide in right ascension.
"""

from dataclasses import dataclass

import numpy as np

from ._validate import (
    _readonly_finite_array,
    _RealScalarInput,
    _validate_finite_scalar,
    _validate_latitude,
    _VectorLike,
)

#: Campo Imperatore, from TRIS I (Zannoni et al. 2008): 42 deg 26 arcmin N.
TRIS_SITE_LATITUDE_DEG = 42.0 + 26.0 / 60.0

#: The declination label printed in the ring headers ("Declination=+42 Degrees").
TRIS_DECLINATION_LABEL_DEG = 42.0

#: Archive-stated tilt of the E plane east of the meridian, in degrees.
TRIS_E_PLANE_EAST_OF_MERIDIAN_DEG = 7.0


@dataclass(frozen=True, init=False, eq=False)
class TRISZenithGeometry:
    """Immutable zenith geometry for the TRIS drift-ring bridge."""

    lst_deg: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    selfrot_deg: np.ndarray
    latitude_deg: float

    def __init__(
        self,
        lst_deg: _VectorLike,
        azimuth_deg: _VectorLike,
        elevation_deg: _VectorLike,
        selfrot_deg: _VectorLike,
        latitude_deg: _RealScalarInput,
    ) -> None:
        object.__setattr__(self, "lst_deg", lst_deg)
        object.__setattr__(self, "azimuth_deg", azimuth_deg)
        object.__setattr__(self, "elevation_deg", elevation_deg)
        object.__setattr__(self, "selfrot_deg", selfrot_deg)
        object.__setattr__(self, "latitude_deg", latitude_deg)
        self.__post_init__()

    def __post_init__(self) -> None:
        latitude = _validate_latitude(self.latitude_deg, "latitude_deg")
        arrays = (
            ("lst_deg", self.lst_deg),
            ("azimuth_deg", self.azimuth_deg),
            ("elevation_deg", self.elevation_deg),
            ("selfrot_deg", self.selfrot_deg),
        )
        validated = [
            (name, _readonly_finite_array(array, name)) for name, array in arrays
        ]
        lengths = {array.size for _name, array in validated}
        if len(lengths) != 1:
            raise ValueError("TRIS geometry arrays must have the same length")
        for name, array in validated:
            object.__setattr__(self, name, array)
        object.__setattr__(self, "latitude_deg", latitude)

    @property
    def boresight_ra_deg(self) -> np.ndarray:
        """Boresight right ascension per sample: identical to ``lst_deg``.

        Exact, not approximate: the antenna is parked at the zenith, so the
        boresight hour angle is zero and its RA is the LST by definition.
        """
        return self.lst_deg

    @property
    def boresight_dec_deg(self) -> float:
        """Boresight declination: the site latitude, for a zenith park."""
        return self.latitude_deg


def tris_zenith_geometry(
    ra_deg: _VectorLike,
    *,
    latitude_deg: _RealScalarInput = TRIS_SITE_LATITUDE_DEG,
    e_plane_east_of_meridian_deg: _RealScalarInput = (
        TRIS_E_PLANE_EAST_OF_MERIDIAN_DEG
    ),
) -> TRISZenithGeometry:
    """Translate TRIS RA labels into the parked-zenith limTOD geometry.

    Supplied RA samples are preserved verbatim as LST samples -- for a zenith
    park that identification is exact, not an approximation.  The park is
    azimuth zero and elevation 90 degrees (azimuth is degenerate at the
    zenith, so it carries no information there and the roll is owned entirely
    by ``selfrot``).  The E plane lies east of the meridian, and limTOD's roll
    convention therefore needs ``selfrot = -e_plane_east_of_meridian_deg``;
    this sign was verified numerically at the real site latitude, not only at
    the degenerate equator.

    The default latitude is the measured 42 deg 26 arcmin site latitude.  Pass
    ``latitude_deg=42.0`` to use the rounded declination label instead; the
    0.43-degree difference is small against a 23-degree beam but it is a
    choice, so make it explicitly.

    This function owns the pointing orientation.  Do not additionally rotate
    the beam map.
    """
    latitude = _validate_latitude(latitude_deg, "latitude_deg")
    e_plane_offset = _validate_finite_scalar(
        e_plane_east_of_meridian_deg, "e_plane_east_of_meridian_deg"
    )
    lst_deg = _readonly_finite_array(ra_deg, "ra_deg")
    ntime = lst_deg.size
    return TRISZenithGeometry(
        lst_deg=lst_deg,
        azimuth_deg=np.zeros(ntime),
        elevation_deg=np.full(ntime, 90.0),
        selfrot_deg=np.full(ntime, -e_plane_offset),
        latitude_deg=latitude,
    )
