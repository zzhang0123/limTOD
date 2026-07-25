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
