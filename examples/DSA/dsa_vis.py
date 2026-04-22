"""Publication-quality patch visualization + shared helpers for the DSA
survey notebooks.

Provides the patch-image plotting used in the proposal figures, plus a
handful of helpers (site constants, coordinate conversion, beam loader)
that the three DSA notebooks share. Keeping these here avoids copy-paste
drift across ``dsa_scan_demo.ipynb``, ``dsa_meerklass_scan.ipynb`` and
``dsa_steer_and_stare.ipynb``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# DSA site constants (Nevada — Owens Valley Radio Observatory proxy location
# quoted in the proposal documents)
# ---------------------------------------------------------------------------
DSA_LAT = 39.553969   # deg
DSA_LON = -114.423973  # deg
DSA_HGT = 1746.51      # m


def radec_to_azel(ra_deg, dec_deg, time_list_sec, start_time_utc, location):
    """Convert a fixed sky target to per-sample local (Az, El).

    Used by stop-and-stare pointings where the telescope tracks an
    ICRS-fixed (RA, Dec) — the horizontal coordinates drift as the Earth
    rotates, so the function evaluates the transform at every time sample.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Target ICRS coordinates in degrees (scalar).
    time_list_sec : array-like
        Time offsets in seconds from ``start_time_utc``.
    start_time_utc : str
        UTC start time parseable by ``astropy.time.Time`` (e.g.
        ``"2024-04-15 04:00:00"``).
    location : astropy.coordinates.EarthLocation
        Observatory location.

    Returns
    -------
    az_deg, el_deg : np.ndarray
        Azimuth (east of north) and elevation in degrees, one element per
        entry in ``time_list_sec``.
    """
    # Local imports so importing dsa_vis at module scope is cheap in
    # notebooks that only need plot_patch.
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time, TimeDelta
    from astropy import units as u

    start = Time(start_time_utc)
    times = start + TimeDelta(np.asarray(time_list_sec), format="sec")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = target.transform_to(AltAz(obstime=times, location=location))
    return altaz.az.deg, altaz.alt.deg


@lru_cache(maxsize=8)
def _load_beam_native() -> np.ndarray:
    """Load the DSA zenith beam (HEALPix nside=128) once and cache it."""
    here = os.path.dirname(os.path.abspath(__file__))
    fits_path = os.path.join(here, "beam_map_zenith.fits")
    return hp.read_map(fits_path)


def dsa_beam_func(*, freq, nside):
    """Return the DSA zenith beam as a sum-normalised HEALPix map.

    Signature matches ``limTOD.TODSim``'s ``beam_func`` protocol
    (``beam_func(*, freq, nside) -> 1-D HEALPix map``). ``freq`` is
    accepted for API compatibility; the zenith beam is achromatic in
    this dataset so the value is ignored.

    The beam is loaded at its native nside=128, downgraded/upgraded to
    the requested nside with ``hp.ud_grade``, then normalised to
    ``sum == 1`` (matches the convention of
    ``limTOD.simulator.example_symm_beam_map``).
    """
    del freq  # achromatic zenith beam — kept for API compatibility
    beam_native = _load_beam_native()
    beam = beam_native if nside == 128 else hp.ud_grade(beam_native, nside)
    beam = np.asarray(beam, dtype=float).copy()
    beam /= beam.sum()
    return beam


def _patch_extent(nside: int, pixel_indices: np.ndarray, pad_frac: float = 0.08):
    """Return (ra_min, ra_max, dec_min, dec_max, ra_center, dec_center)
    covering all `pixel_indices` with a small padding margin."""
    theta, phi = hp.pix2ang(nside, pixel_indices)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    # Handle possible RA wrap (patches that cross RA=0)
    if ra.max() - ra.min() > 180:
        ra = np.where(ra > 180, ra - 360, ra)

    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    dra = ra_max - ra_min
    ddec = dec_max - dec_min
    ra_center = 0.5 * (ra_min + ra_max)
    dec_center = 0.5 * (dec_min + dec_max)

    return (
        ra_min - pad_frac * dra,
        ra_max + pad_frac * dra,
        dec_min - pad_frac * ddec,
        dec_max + pad_frac * ddec,
        ra_center,
        dec_center,
    )


def plot_patch(
    *,
    map_vec: np.ndarray,
    nside: int,
    pixel_indices: np.ndarray,
    ax=None,
    title: str = "",
    unit: str = "K",
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    npix_x: int = 600,
    pad_frac: float = 0.08,
    show_cbar: bool = True,
    xlabel: str = "Right Ascension [deg]",
    ylabel: str = "Declination [deg]",
    fontsize: int = 12,
    rasterized: bool = True,
) -> plt.Axes:
    """Plot a HEALPix patch as a tight, RA/Dec-labelled image.

    Parameters
    ----------
    map_vec : (N,) array
        Values at ``pixel_indices`` (in `map_vec` order), not a full sky map.
    nside : int
        HEALPix nside of the full sky grid.
    pixel_indices : (N,) int array
        HEALPix RING pixel indices corresponding to ``map_vec``.
    ax : matplotlib Axes, optional
        If None, creates a new figure and axes.
    title, unit, cmap : str
    vmin, vmax : float or None
        Colour-scale limits. If ``None`` they are set to the 2/98 percentile
        of the data so outliers do not wash out the image.
    npix_x : int
        Number of image columns (rows chosen from aspect ratio).
    pad_frac : float
        Padding around the bounding box, as a fraction of the patch extent.
    show_cbar : bool
        Draw a horizontal colourbar beneath the image.
    rasterized : bool
        Rasterise the raster image only; axes and text stay vector.

    Returns
    -------
    ax : matplotlib Axes
    """
    # Build a full-sky map and fill known pixels
    full = np.full(hp.nside2npix(nside), np.nan, dtype=float)
    full[pixel_indices] = map_vec

    # Get the patch's RA/Dec bounding box
    ra_lo, ra_hi, dec_lo, dec_hi, ra_c, dec_c = _patch_extent(
        nside, pixel_indices, pad_frac=pad_frac
    )
    dra = ra_hi - ra_lo
    ddec = dec_hi - dec_lo

    # Sample on a regular RA/Dec grid at this nside's native resolution
    # Aspect: correct for cos(dec) so circles look round
    cos_dec = np.cos(np.radians(dec_c))
    aspect = (ddec) / (dra * cos_dec)
    npix_y = max(100, int(round(npix_x * aspect)))

    ra_grid = np.linspace(ra_lo, ra_hi, npix_x)
    dec_grid = np.linspace(dec_lo, dec_hi, npix_y)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)

    # Convert to HEALPix coordinates
    theta = np.radians(90.0 - DEC)
    phi = np.radians(RA % 360.0)
    pix = hp.ang2pix(nside, theta, phi)
    img = full[pix]

    # Default colour limits: robust percentile
    valid = np.isfinite(img)
    if vmin is None or vmax is None:
        if valid.any():
            lo_default, hi_default = np.percentile(img[valid], [2, 98])
        else:
            lo_default, hi_default = 0.0, 1.0
        if vmin is None:
            vmin = lo_default
        if vmax is None:
            vmax = hi_default

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, max(4.0, 7.5 * aspect)))
    else:
        fig = ax.figure

    # Masked array so NaN pixels show as background
    img_masked = np.ma.masked_invalid(img)
    cmap_obj = mpl.cm.get_cmap(cmap).copy() if hasattr(mpl.cm, "get_cmap") else plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="0.85")

    im = ax.imshow(
        img_masked,
        origin="lower",
        extent=[ra_hi, ra_lo, dec_lo, dec_hi],  # RA increases to the left
        aspect=1.0 / cos_dec,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=rasterized,
    )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize - 1)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="white")

    if show_cbar:
        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            pad=0.02, fraction=0.045)
        cbar.set_label(unit, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize - 1)

    return ax


def plot_map_compare(
    *,
    sky_est: np.ndarray,
    sky_truth: np.ndarray,
    nside: int,
    pixel_indices: np.ndarray,
    freq_mhz: float,
    strategy_name: str,
    cmap: str = "inferno",
    savepath: str | None = None,
) -> plt.Figure:
    """Side-by-side comparison of truth vs. recovered sky for one strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Use the same colour scale so the comparison is visually fair
    lo, hi = np.percentile(sky_truth, [2, 98])

    plot_patch(
        map_vec=sky_truth,
        nside=nside,
        pixel_indices=pixel_indices,
        ax=axes[0],
        title=f"Input GDSM sky  ({freq_mhz:.0f} MHz)",
        unit="K",
        cmap=cmap,
        vmin=lo,
        vmax=hi,
    )
    plot_patch(
        map_vec=sky_est,
        nside=nside,
        pixel_indices=pixel_indices,
        ax=axes[1],
        title=f"Recovered map  ({strategy_name})",
        unit="K",
        cmap=cmap,
        vmin=lo,
        vmax=hi,
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig
