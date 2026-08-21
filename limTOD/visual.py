from typing import Optional

import healpy as hp
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover — matplotlib is not a base dep
    raise ImportError(
        "limTOD.visual needs matplotlib (pip install matplotlib); "
        "it is not part of limTOD's base dependencies."
    ) from exc


def view_patch_map(map: np.ndarray, pixel_indices: np.ndarray) -> np.ndarray:
    """
    Blank everything outside a patch so healpy draws the patch alone.

    Parameters
    ----------
    map : np.ndarray
        Full-sky HEALPix map (RING ordering).
    pixel_indices : np.ndarray
        Indices of the pixels belonging to the patch.

    Returns
    -------
    np.ndarray
        A copy of `map` in which every pixel outside `pixel_indices` is
        `healpy.UNSEEN`. healpy renders those with the `badcolor` of the
        plotting call rather than as a value, so the patch is the only thing
        carrying the colour scale.
    """
    # Create a new map with just the patch (other pixels set to UNSEEN)
    patch_only_map = np.full(len(map), hp.UNSEEN)
    patch_only_map[pixel_indices] = map[pixel_indices]
    return patch_only_map

def gnomview_patch(*,
                   map: np.ndarray,
                   nside: int,
                   pixel_indices: np.ndarray,
                   sky_min: Optional[float] = None,
                   sky_max: Optional[float] = None,
                   res: float = 5,
                   title: str = " ",
                   save_path: Optional[str] = None,
                   cmap: str = 'jet',
                   cbar: bool = True,
                   xtick: bool = False,
                   ytick: bool = False,
                   unit: str = 'K',
                   turn_into_map: bool = True,
                   fts: float = 16,
                   xsize: Optional[int] = None,
                   ysize: Optional[int] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   ) -> None:
    """
    Gnomonic plot of a patch, framed on the patch itself.

    Wraps `healpy.gnomview` with the framing and annotation a disc-restricted
    map needs: the projection is centred on the median longitude and latitude
    of `pixel_indices`, so the patch lands in the middle wherever on the sky
    it sits; everything outside it is drawn grey through
    :func:`view_patch_map`; and a 10-degree graticule is added. Draws into the
    current matplotlib figure and returns None.

    Parameters
    ----------
    map : np.ndarray
        Patch values in the order of `pixel_indices` (the default, with
        `turn_into_map=True`) -- the form a patch-based map-maker returns --
        or a full-sky map when `turn_into_map=False`.
    nside : int
        HEALPix resolution of the full sky the patch belongs to.
    pixel_indices : np.ndarray
        Indices of the patch pixels.
    sky_min, sky_max : float, optional
        Colour-scale limits; healpy autoscales when omitted.
    res : float, optional
        Projected resolution in arcmin per pixel (healpy's `reso`).
        Default 5.
    title : str, optional
        Plot title; a blank title draws none.
    save_path : str, optional
        When given, the figure is written here with a tight bounding box.
    cmap : str, optional
        Matplotlib colormap name. Default 'jet'.
    cbar : bool, optional
        Draw the colourbar. Its presence shifts where the tick and axis
        labels below are placed. Default True.
    xtick, ytick : bool, optional
        Annotate the projection centre's longitude / latitude.
    unit : str, optional
        Colourbar unit label. Default 'K'.
    turn_into_map : bool, optional
        Whether `map` holds patch values (True, default) or a full sky
        (False).
    fts : float, optional
        Base font size for the annotations. Default 16.
    xsize, ysize : int, optional
        Size of the projected image in pixels; healpy's defaults when
        omitted.
    xlabel, ylabel : str, optional
        Axis labels, placed with `figure.text` because gnomview draws no
        axes of its own.

    Returns
    -------
    None
    """
    NPIX = hp.nside2npix(nside)
    if turn_into_map:
        aux_map = np.zeros(NPIX, dtype=float)
        aux_map[pixel_indices] = map
    else:
        aux_map = map
    patch_only_map = view_patch_map(aux_map, pixel_indices)

    # middle_pix_index = pixel_indices[len(pixel_indices)//2]
    # theta, phi = hp.pix2ang(nside, middle_pix_index)
    theta, phi = hp.pix2ang(nside, pixel_indices)
    lon, lat = np.degrees(phi), 90 - np.degrees(theta)
    lon_center, lat_center = np.median(lon), np.median(lat)


    hp.gnomview( patch_only_map, rot=(lon_center, lat_center), 
           xsize=xsize, ysize=ysize,
           reso=res, title=title, 
           unit=unit, cmap=cmap, min=sky_min, max=sky_max,
           notext=True,
           coord=['C'], 
           cbar=cbar, 
           badcolor='gray')
    cb = plt.gcf().axes[-1]  # Get the colorbar axis (usually the last one)
    cb.tick_params(labelsize=fts)  # Set the font size to 18 (adjust as needed)
    hp.graticule(dpar=10, dmer=10, coord=['C'], local=True)  # Add graticule lines; separation in degrees
    plt.gca().set_facecolor('gray')  # Set background to gray

    # Add axis labels using plt.text
    fig = plt.gcf()
    ax = plt.gca()
    if title and title.strip():  # Only if title is not empty
        ax.set_title(title, fontsize=fts-1, pad=5)

    if cbar:
        if xtick:
            fig.text(0.5, 0.185, str(lon_center)[:7], ha='center', fontsize=fts-1)
        if ytick:
            fig.text(0.045, 0.37, str(lat_center)[:5], va='center', rotation='vertical', fontsize=fts-1)
        if xlabel is not None:
            fig.text(0.5, 0.155, xlabel, ha='center', fontsize=fts-1)
        if ylabel is not None:
            fig.text(0.01, 0.4, ylabel, va='center', rotation='vertical', fontsize=fts-1)
    else:
        if xtick:
            fig.text(0.5, 0.31, str(lon_center)[:7], ha='center', fontsize=fts-1)
        if ytick:
            fig.text(0.045, 0.5, str(lat_center)[:5], va='center', rotation='vertical', fontsize=fts-1)
        if xlabel is not None:
            fig.text(0.5, 0.28, xlabel, ha='center', fontsize=fts-1)
        if ylabel is not None:
            fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=fts-1)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', 
                pad_inches=0.1)
    
    pass
