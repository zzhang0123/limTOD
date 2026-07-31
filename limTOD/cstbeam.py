"""CST Studio far-field exports -> HEALPix beam maps.

The sibling of :mod:`limTOD.uvbeam` for the other format a measured or
simulated horn arrives in: CST Microwave Studio far-field ASCII exports, one
file per frequency. Three entry points:

* :func:`read_cst_farfield` — one file into a regular ``(theta, phi)`` grid.
* :func:`cst_beam_maps` — a directory of them onto HEALPix maps in limTOD's
  beam-map convention (boresight at the pole, RING ordering), interpolated in
  frequency.
* :func:`cst_beam_func` — the same wrapped as a ``beam_func(freq=...,
  nside=...)`` callable satisfying :class:`limTOD.TODSim`'s contract, so a CST
  horn drops straight into a simulation.

Needs only ``healpy`` and ``scipy``, both base dependencies — there is no extra
to install for this module.

Conventions
-----------

Stated because getting one wrong returns a finite, correctly shaped, wrong
beam.

* **Theta.** CST's ``Theta`` is measured from the model's ``+z`` axis and maps
  directly onto the HEALPix colatitude: the boresight sits at the pole, which
  is what limTOD's beam maps mean by beam-local.
* **The quantity.** ``Abs(Dir.)`` is total directivity in dBi — a POWER
  quantity. Maps come back as ``10 ** (dBi / 10)``, which is the ``B`` of
  ``int(B T) / int(B)``. **Nothing is normalized here**; ask the consumer to
  divide by its own quadrature ``int(B)``, which is the only way the band limit
  cancels exactly.
* **Phi, which is not derivable from the file.** CST's ``Phi`` is measured from
  the model's ``+x`` axis; limTOD's beam-map ``phi = 0`` is carried to the
  direction of increasing elevation. Which physical direction the CST ``+x``
  axis points is a fact about how the horn was built and mounted, and it is not
  in the export — so it cannot be recovered here. ``phi0_deg`` and
  ``phi_sense`` expose the two degrees of freedom (an offset and a handedness).
  Their defaults are the identity mapping, which is **an assumption to check
  against the as-built horn, not a result**. For a beam with real azimuthal
  structure the handedness is not a detail: RHINO's horn varies by 30-60 %
  around the ``theta = 30`` deg ring, so getting it backwards mirrors that
  structure into the wrong half of the sky while leaving every integral,
  every peak and every symmetric diagnostic unchanged.

  Unlike :mod:`limTOD.uvbeam`, whose azimuth convention *is* fixed by pyuvdata
  and is therefore locked numerically by a test, this one cannot be: the file
  does not contain the information. What the tests here lock instead is that
  the knobs do the right thing — that ``phi_sense`` is a reflection about
  ``phi = 0`` rather than a relabelling, and that ``phi0_deg`` is a rotation
  that conserves the integral.

* **Frequency.** In MHz throughout, as elsewhere in limTOD, and read from the
  trailing number of each filename's stem: ``HornDry70.5.txt`` is 70.5 MHz.
  Interpolation between bracketing files is linear in linear power.
  Extrapolation is refused — a beam invented outside the simulated band is not
  a beam.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Union

import healpy as hp
import numpy as np

#: Column index of ``Abs(Dir.)[dBi]`` in a CST far-field ASCII export.
_CST_DIRECTIVITY_COLUMN = 2

#: Trailing frequency in MHz of a CST filename, e.g. ``HornDry70.5.txt``.
_FREQ_IN_NAME = re.compile(r"([0-9]+(?:\.[0-9]+)?)$")

__all__ = [
    "read_cst_farfield",
    "cst_frequency_table",
    "cst_beam_maps",
    "cst_beam_func",
]


def read_cst_farfield(
    path: Union[str, Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one CST far-field export into a regular ``(theta, phi)`` grid.

    Parameters
    ----------
    path : str or Path
        A CST Studio far-field ASCII export: two header lines, then
        ``Theta Phi Abs(Dir.) ...`` rows on a regular grid.

    Returns
    -------
    theta_deg : ndarray, shape (n_theta,)
    phi_deg : ndarray, shape (n_phi,)
    directivity : ndarray, shape (n_theta, n_phi)
        Linear power, ``10 ** (dBi / 10)`` — not dB.

    Raises
    ------
    ValueError
        If the rows do not fill a complete regular grid. An incomplete export
        would otherwise reshape into a plausible-looking beam with the samples
        in the wrong places.
    """
    table = np.loadtxt(path, skiprows=2)
    if table.ndim != 2 or table.shape[1] <= _CST_DIRECTIVITY_COLUMN:
        raise ValueError(
            f"{path}: expected a CST far-field table with at least "
            f"{_CST_DIRECTIVITY_COLUMN + 1} columns, got shape {table.shape}."
        )
    theta_deg = np.unique(table[:, 0])
    phi_deg = np.unique(table[:, 1])
    if theta_deg.size * phi_deg.size != table.shape[0]:
        raise ValueError(
            f"{path}: {table.shape[0]} rows do not fill the "
            f"{theta_deg.size} x {phi_deg.size} (theta, phi) grid they span; "
            "the export is incomplete or not on a regular grid."
        )
    # Rows run theta-fastest within each phi block, so (n_phi, n_theta) is the
    # natural reshape; transpose to the (theta, phi) the interpolator wants.
    directivity = 10.0 ** (
        table[:, _CST_DIRECTIVITY_COLUMN]
        .reshape(phi_deg.size, theta_deg.size)
        .T
        / 10.0
    )
    return theta_deg, phi_deg, directivity


def cst_frequency_table(
    directory: Union[str, Path], *, suffix: str = ".txt"
) -> Dict[float, Path]:
    """Map frequency [MHz] to file for a directory of CST exports.

    The frequency is the trailing number of the stem, in MHz —
    ``HornDry70.5.txt`` is 70.5. Files whose stem does not end in a number are
    ignored, so a ``README.txt`` alongside the exports is harmless.

    Raises
    ------
    ValueError
        If the directory holds no matching file.
    """
    directory = Path(directory).expanduser()
    table: Dict[float, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        match = _FREQ_IN_NAME.search(path.stem)
        if match is not None:
            table[float(match.group(1))] = path
    if not table:
        raise ValueError(
            f"No CST exports found in {directory} (looked for '*{suffix}' whose "
            "stem ends in a frequency in MHz, e.g. 'HornDry70.5.txt')."
        )
    return table


def _sample_to_healpix(
    path: Path, nside: int, phi0_deg: float, sign: float
) -> np.ndarray:
    """One CST file onto a HEALPix RING map in the beam-local frame."""
    from scipy.interpolate import RegularGridInterpolator

    theta_deg, phi_deg, directivity = read_cst_farfield(path)
    # Close the azimuth circle so the interpolator wraps instead of clamping
    # the last degree onto a boundary value.
    phi_closed = np.append(phi_deg, phi_deg[0] + 360.0)
    grid = np.concatenate([directivity, directivity[:, :1]], axis=1)
    interp = RegularGridInterpolator(
        (theta_deg, phi_closed), grid,
        method="linear", bounds_error=False, fill_value=None,
    )
    theta_hp, phi_hp = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    phi_cst = np.rad2deg(sign * phi_hp) + phi0_deg
    return interp(
        np.stack(
            [
                np.rad2deg(theta_hp),
                np.mod(phi_cst - phi_deg[0], 360.0) + phi_deg[0],
            ],
            axis=-1,
        )
    )


def _phi_sign(phi_sense: str) -> float:
    if phi_sense not in ("ccw", "cw"):
        raise ValueError(f"phi_sense must be 'ccw' or 'cw', got {phi_sense!r}.")
    return 1.0 if phi_sense == "ccw" else -1.0


def _interp_frequency(
    freq: float, grid: np.ndarray, maps: np.ndarray
) -> np.ndarray:
    """Linear interpolation of ``maps`` (n_grid, npix) at ``freq``."""
    if grid.size == 1:
        return maps[0]
    upper = int(np.clip(np.searchsorted(grid, freq, side="left"), 1, grid.size - 1))
    lower = upper - 1
    span = grid[upper] - grid[lower]
    weight = 0.0 if span == 0.0 else (freq - grid[lower]) / span
    return (1.0 - weight) * maps[lower] + weight * maps[upper]


def cst_beam_maps(
    directory: Union[str, Path],
    freq_MHz,
    *,
    nside: int,
    suffix: str = ".txt",
    phi0_deg: float = 0.0,
    phi_sense: str = "ccw",
    _cache: Union[Dict, None] = None,
) -> np.ndarray:
    """Sample a directory of CST exports onto HEALPix maps at given frequencies.

    Parameters
    ----------
    directory : str or Path
        Directory of per-frequency CST exports (see :func:`cst_frequency_table`).
    freq_MHz : array_like, shape (n_freq,)
        Output frequencies [MHz].
    nside : int
        HEALPix resolution of the output maps (RING ordering).
    suffix : str
        File extension of the exports.
    phi0_deg : float
        CST azimuth that lands on the beam-map ``phi = 0`` meridian.
    phi_sense : {"ccw", "cw"}
        Whether CST azimuth increases with beam-map ``phi``. See the module
        docstring: this is a fact about the horn, not about the file.

    Returns
    -------
    ndarray, shape (n_freq, 12 * nside ** 2)
        Linear-power beam maps, **unnormalized**.

    Raises
    ------
    ValueError
        On an unknown ``phi_sense``, or a requested frequency outside the range
        the directory covers.

    Notes
    -----
    Only the files actually bracketing a requested frequency are read: a
    production directory holds dozens, each a 65k-row parse.
    """
    sign = _phi_sign(phi_sense)
    freq_MHz = np.atleast_1d(np.asarray(freq_MHz, dtype=float))
    table = cst_frequency_table(directory, suffix=suffix)
    available = np.array(sorted(table))
    if freq_MHz.min() < available[0] or freq_MHz.max() > available[-1]:
        raise ValueError(
            f"Requested {freq_MHz.min():.3f}-{freq_MHz.max():.3f} MHz but "
            f"{Path(directory).expanduser()} covers only "
            f"{available[0]:.3f}-{available[-1]:.3f} MHz. Extrapolating a beam "
            "outside its simulated band would return a plausible, unsupported "
            "answer."
        )

    needed = sorted(
        {
            available[index]
            for f in freq_MHz
            for index in (
                max(int(np.searchsorted(available, f, side="right")) - 1, 0),
                min(int(np.searchsorted(available, f, side="left")), available.size - 1),
            )
        }
    )
    sampled = []
    for f in needed:
        key = (table[f], nside, phi0_deg, sign)
        if _cache is not None and key in _cache:
            sampled.append(_cache[key])
            continue
        one = _sample_to_healpix(table[f], nside, phi0_deg, sign)
        if _cache is not None:
            _cache[key] = one
        sampled.append(one)

    grid = np.asarray(needed)
    stack = np.stack(sampled)
    return np.stack([_interp_frequency(f, grid, stack) for f in freq_MHz])


def cst_beam_func(
    directory: Union[str, Path],
    *,
    suffix: str = ".txt",
    phi0_deg: float = 0.0,
    phi_sense: str = "ccw",
) -> Callable[..., np.ndarray]:
    """Wrap a directory of CST exports as a limTOD ``beam_func``.

    The returned callable satisfies :class:`limTOD.TODSim`'s contract,
    ``beam_func(freq=..., nside=...) -> (npix,)`` with ``freq`` in MHz, so a
    CST horn drops straight into a simulation::

        sim = TODSim(beam_func=cst_beam_func(horn_dir), sky_func=..., ...)

    Chromatic by construction: each channel interpolates between the two
    bracketing exports.

    Configuration errors — an unreadable directory, an unknown ``phi_sense`` —
    surface here rather than at the first call, and the per-file HEALPix
    resampling is **cached across calls**, so a simulation sweeping 200
    channels through a 61-file directory parses each file once instead of
    hundreds of times.
    """
    _phi_sign(phi_sense)                      # fail now, not on the first channel
    cst_frequency_table(directory, suffix=suffix)
    cache: Dict = {}

    def beam_func(*, freq: float, nside: int) -> np.ndarray:
        return cst_beam_maps(
            directory, freq, nside=nside, suffix=suffix,
            phi0_deg=phi0_deg, phi_sense=phi_sense, _cache=cache,
        )[0]

    return beam_func
