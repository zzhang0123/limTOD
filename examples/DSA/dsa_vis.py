"""Publication-quality patch visualization for DSA survey notebooks.

Replaces limTOD.visual.gnomview_patch with a tighter, better-labelled
version whose axes are real RA/Dec in degrees and whose figure extent
matches the observed patch (no wasted blank space).
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize


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
