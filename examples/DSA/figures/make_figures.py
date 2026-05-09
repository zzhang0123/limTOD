"""Generate publication-quality figures for the DSA survey strategy comparison."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time, TimeDelta
import astropy.units as u
import healpy as hp
from limTOD import example_scan, GDSM_sky_model

# ---------- Publication styling ----------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.3",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Palette — colourblind-friendly
C_SETTING = "#1f77b4"  # blue
C_RISING = "#d62728"  # red
C_GRID = "#2ca02c"  # green
C_ACCENT = "#ff7f0e"  # orange

# ---------- DSA / observation setup ----------
DSA_LAT, DSA_LON, DSA_HGT = 39.553969, -114.423973, 1746.51
dsa_loc = EarthLocation(lat=DSA_LAT * u.deg, lon=DSA_LON * u.deg, height=DSA_HGT * u.m)
BEAM_FWHM = 4.85  # deg


def beam_ellipse(ra: float, dec: float, fwhm_deg: float, **kwargs) -> Ellipse:
    """Build an ``Ellipse`` that renders as a round beam footprint on the sky.

    At declination ``dec`` a circle of angular radius ``r`` spans
    ``2 * r / cos(dec)`` in RA coordinates and ``2 * r`` in Dec — so we
    construct the ellipse with those ``width`` and ``height``. When the plot
    uses ``set_aspect(1/cos(dec_center))``, the ellipse renders visually round.
    """
    r = fwhm_deg / 2.0
    width_ra = 2.0 * r / np.cos(np.radians(dec))  # RA-coordinate extent
    height_dec = 2.0 * r  # Dec-coordinate extent
    return Ellipse((ra, dec), width=width_ra, height=height_dec, **kwargs)


def gdsm_background(
    ax, ra_lo: float, ra_hi: float, dec_lo: float, dec_hi: float,
    freq_mhz: float = 1000.0, nside: int = 128, n_grid: int = 400,
    cmap: str = "bone", alpha: float = 0.85,
) -> None:
    """Draw the GDSM sky temperature as a background image on ``ax``.

    ``ax`` must already have its aspect/xlim/ylim/inverted-RA set. We sample
    the GDSM HEALPix map on a regular (RA, Dec) grid and ``imshow`` it.
    """
    sky = GDSM_sky_model(freq=freq_mhz, nside=nside)
    ra_grid = np.linspace(min(ra_lo, ra_hi), max(ra_lo, ra_hi), n_grid)
    dec_grid = np.linspace(dec_lo, dec_hi, n_grid)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    theta = np.radians(90.0 - DEC)
    phi = np.radians(RA % 360.0)
    pix = hp.ang2pix(nside, theta, phi)
    img = sky[pix]

    vmin, vmax = np.percentile(img, [2, 98])
    extent = [ra_grid[0], ra_grid[-1], dec_grid[0], dec_grid[-1]]
    im = ax.imshow(
        img, extent=extent, origin="lower", aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, zorder=0,
        interpolation="bilinear", rasterized=True,
    )
    return im


def optical_background(
    ax, ra_center: float, dec_center: float,
    fov_deg: float, width_px: int = 1600, height_px: int = 800,
    hips: str = "CDS/P/DSS2/color", cache_path: str | None = None,
) -> None:
    """Draw a HiPS optical sky cutout as the background of ``ax``.

    Uses the CDS ``hips2fits`` web service (``astroquery``). The default
    HiPS dataset is the DSS2 colour composite — a tasteful all-sky
    optical image well suited to proposal figures.

    The result is cached on disk so the figure script can be re-run
    offline once the cutout has been downloaded.
    """
    from PIL import Image

    if cache_path and os.path.exists(cache_path):
        img = np.asarray(Image.open(cache_path))
    else:
        from astroquery.hips2fits import hips2fits
        # `hips2fits.query(format="png")` returns a numpy (H, W, 4) RGBA
        # array directly (no PIL roundtrip needed).
        img = hips2fits.query(
            hips=hips,
            width=width_px, height=height_px,
            ra=ra_center * u.deg, dec=dec_center * u.deg,
            fov=fov_deg * u.deg,
            projection="TAN",
            coordsys="icrs",
            format="png",
        )
        img = np.asarray(img)
        if cache_path:
            Image.fromarray(img).save(cache_path)

    # The TAN projection at this FOV is an excellent linear approximation
    # over our small (~30°) patch.  Draw it edge-to-edge with a tangent
    # plane assumption: ±fov/2 in RA·cos(dec) and ±fov_dec/2 in Dec.
    aspect = height_px / width_px
    half_w = fov_deg / 2.0
    half_h = fov_deg * aspect / 2.0
    # Convert tangent-plane half-width back to RA span at dec_center
    half_ra = half_w / np.cos(np.radians(dec_center))
    extent = [
        ra_center - half_ra, ra_center + half_ra,
        dec_center - half_h, dec_center + half_h,
    ]
    ax.imshow(
        img, extent=extent, origin="upper", aspect="auto", zorder=0,
        interpolation="bilinear", rasterized=True,
    )


def azel_to_radec(az_deg, el_deg, time_list_sec, start_utc):
    t0 = Time(start_utc)
    ts = t0 + TimeDelta(time_list_sec, format="sec")
    altaz = AltAz(obstime=ts, location=dsa_loc, az=az_deg * u.deg, alt=el_deg * u.deg)
    ic = SkyCoord(altaz).icrs
    return ic.ra.deg, ic.dec.deg


def radec_to_azel(ra_deg, dec_deg, time_list_sec, start_utc):
    t0 = Time(start_utc)
    ts = t0 + TimeDelta(time_list_sec, format="sec")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = target.transform_to(AltAz(obstime=ts, location=dsa_loc))
    return altaz.az.deg, altaz.alt.deg


# ---------- Scan data ----------
EL_SCAN = 55.0
time_setting, az_setting = example_scan(az_s=-60.3, az_e=-42.3, dt=2.0, n_repeats=13)
time_rising, az_rising = example_scan(az_s=42.3, az_e=60.3, dt=2.0, n_repeats=13)
start_setting = "2024-04-15 08:25:05"  # tuned so RA-mean = 180° at el=55°
start_rising = "2024-04-15 02:03:50"   # tuned so RA-mean = 180° at el=55°

ra_set, dec_set = azel_to_radec(az_setting, EL_SCAN, time_setting, start_setting)
ra_ris, dec_ris = azel_to_radec(az_rising, EL_SCAN, time_rising, start_rising)

# Hex pointing grid
center_ra, center_dec = 180.0, 52.0
d_dec = 3.5
d_ra = 3.5 / np.cos(np.radians(center_dec))
pointings = []
for i_row, dec_off in enumerate([-d_dec, 0, d_dec]):
    ra_shift = d_ra / 2 if i_row % 2 == 1 else 0
    for ra_off in [-d_ra, 0, d_ra]:
        pointings.append((center_ra + ra_off + ra_shift, center_dec + dec_off))
pointings = np.array(pointings)

# ---------- FIGURE 1: Both strategies on sky (proposal-ready, compact) ----------
fig, ax = plt.subplots(figsize=(7.6, 5.3))

# --- Background: DSS2 colour optical sky (cached after first download) ---
optical_background(
    ax, ra_center=180.0, dec_center=52.5, fov_deg=32.0,
    width_px=1600, height_px=800, hips="CDS/P/DSS2/color",
    cache_path=os.path.join(os.path.dirname(__file__), "_dss2_cache.png"),
)

# Azimuthal-scan tracks — brighter colours so they pop against optical sky
ax.plot(
    ra_set, dec_set, ".", ms=1.0, color="#4ec1ff", alpha=0.55, rasterized=True,
    zorder=2,
)
ax.plot(
    ra_ris, dec_ris, ".", ms=1.0, color="#ff8a8a", alpha=0.55, rasterized=True,
    zorder=2,
)

# Stop-and-stare pointings — bright lime green, clearly visible on dark sky
C_BEAM = "#5dff5d"
for ra_p, dec_p in pointings:
    ax.add_patch(beam_ellipse(
        ra_p, dec_p, BEAM_FWHM,
        fill=False, edgecolor=C_BEAM, lw=1.7, alpha=0.95, zorder=3,
    ))
    ax.plot(ra_p, dec_p, "+", color=C_BEAM, ms=8, mew=1.6, zorder=4)

# FWHM annotation with a subtle dark-bg halo
ax.text(
    center_ra, 46.0,
    rf"stop-and-stare beam FWHM $= {BEAM_FWHM:.1f}^\circ$",
    ha="center", va="bottom", fontsize=9, color=C_BEAM,
    fontweight="bold", zorder=5,
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.5, pad=2.5),
)

# Custom legend (compact, in-axes)
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4ec1ff",
           markersize=6, label="Azimuthal scan (setting)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff8a8a",
           markersize=6, label="Azimuthal scan (rising)"),
    Line2D([0], [0], marker="o", color=C_BEAM, markerfacecolor="none",
           markersize=12, mew=1.4, linestyle="", label="Stop-and-stare pointing"),
]
ax.legend(
    handles=legend_elements,
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=3, framealpha=0.0, handlelength=1.4,
    borderpad=0.3, columnspacing=1.8, handletextpad=0.5,
)

ax.set_xlabel("Right Ascension [deg]")
ax.set_ylabel("Declination [deg]")
# No explicit title: the flat legend above the axes serves as the caption.
ax.set_aspect(1 / np.cos(np.radians(center_dec)))
ax.invert_xaxis()
ax.set_xlim(195, 165)
ax.set_ylim(45, 60)

# Tiny corner credit so reviewers know what the background is
ax.text(
    0.99, 0.01, "background: DSS2 colour (CDS HiPS)",
    transform=ax.transAxes, ha="right", va="bottom",
    color="white", fontsize=7.5, alpha=0.65, zorder=5,
)

plt.tight_layout(pad=0.6)
plt.savefig("both_strategies.pdf")
plt.savefig("both_strategies.png", dpi=300)
plt.close()
print("Saved both_strategies.{pdf,png}")

# ---------- FIGURE 2: Tracking Az/El for a single pointing ----------
ra_t, dec_t = 180.0, 52.0
dt = 2.0
tp = 21 * 60
tl = np.arange(0, tp, dt)
az, el = radec_to_azel(ra_t, dec_t, tl, "2024-04-15 04:00:00")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)

ax1.plot(tl / 60, az, color=C_SETTING, lw=1.8)
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Azimuth [deg]")
ax1.set_title(rf"Azimuth track  (RA $= {ra_t:.0f}^\circ$, Dec $= {dec_t:.0f}^\circ$)",
              fontsize=11)
ax1.text(
    0.04, 0.93, rf"$\Delta$az $= {az.max() - az.min():.1f}^\circ$",
    transform=ax1.transAxes, fontsize=10, va="top",
    bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=3),
)

ax2.plot(tl / 60, el, color=C_RISING, lw=1.8)
ax2.set_xlabel("Time [min]")
ax2.set_ylabel("Elevation [deg]")
ax2.set_title("Elevation track", fontsize=11)
ax2.text(
    0.04, 0.93, rf"$\Delta$el $= {el.max() - el.min():.2f}^\circ$",
    transform=ax2.transAxes, fontsize=10, va="top",
    bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=3),
)

fig.suptitle(
    "Stop-and-stare: Az and El both vary during a 21 min pointing",
    y=1.00, fontsize=12,
)
plt.tight_layout(pad=0.6)
plt.savefig("tracking_azel.pdf")
plt.savefig("tracking_azel.png", dpi=300)
plt.close()
print("Saved tracking_azel.{pdf,png}")

# ---------- FIGURE 3: Hex pointing grid (standalone, labelled) ----------
fig, ax = plt.subplots(figsize=(6.2, 5.6))

for i, (ra_p, dec_p) in enumerate(pointings):
    ax.add_patch(beam_ellipse(
        ra_p, dec_p, BEAM_FWHM,
        fill=False, edgecolor=C_GRID, lw=1.6, alpha=0.9, zorder=3,
    ))
    ax.plot(ra_p, dec_p, "+", color=C_GRID, ms=9, mew=1.7, zorder=4)
    ax.annotate(
        f"PC{i}", (ra_p, dec_p - 1.1),
        ha="center", va="center", fontsize=8.5, color="0.25", zorder=5,
    )

# Neighbour separation annotation: PC1 (Dec row -3.5) → PC4 (Dec row 0),
# drawn on the physical sky. Δ = 3.5° (Dec spacing), overlap = FWHM - Δ.
pc1_ra, pc1_dec = pointings[1]
pc4_ra, pc4_dec = pointings[4]
ax.plot(
    [pc1_ra, pc4_ra], [pc1_dec, pc4_dec],
    color=C_ACCENT, lw=1.3, ls="--", zorder=2,
)
ax.annotate(
    rf"$\Delta = {d_dec:.1f}^\circ$  "
    rf"({(1 - d_dec / BEAM_FWHM) * 100:.0f}% overlap)",
    ((pc1_ra + pc4_ra) / 2, (pc1_dec + pc4_dec) / 2),
    xytext=(10, 0), textcoords="offset points",
    fontsize=9, color=C_ACCENT, fontweight="bold", va="center",
)

ax.set_xlabel("Right Ascension [deg]")
ax.set_ylabel("Declination [deg]")
ax.set_title(rf"Stop-and-stare 3×3 hex grid  •  FWHM $= {BEAM_FWHM:.1f}^\circ$",
             fontsize=12)
ax.set_aspect(1 / np.cos(np.radians(center_dec)))
ax.invert_xaxis()
ax.set_xlim(center_ra + 1.7 * d_ra, center_ra - 1.7 * d_ra)
ax.set_ylim(center_dec - 1.6 * d_dec, center_dec + 1.6 * d_dec)

plt.tight_layout(pad=0.5)
plt.savefig("hex_pointings.pdf")
plt.savefig("hex_pointings.png", dpi=300)
plt.close()
print("Saved hex_pointings.{pdf,png}")

# ---------- FIGURE 4: Azimuthal scan — Az vs time + sky coverage ----------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1]}
)

# Az pattern — first ~15 min for legibility
n_show = 2 * 220  # two back-and-forth sweeps
ax1.plot(
    time_setting[:n_show] / 60, az_setting[:n_show],
    color=C_SETTING, lw=1.4, label="Setting (az -60°..-42°)",
)
ax1.plot(
    time_rising[:n_show] / 60, az_rising[:n_show],
    color=C_RISING, lw=1.4, label="Rising (az +42°..+60°)",
)
ax1.axhline(0, color="0.35", ls="--", lw=0.8, label="Due North")
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Azimuth [deg]")
ax1.set_title("Scan pattern (first ~15 min)", fontsize=11)
ax1.legend(loc="center right", ncol=1, framealpha=0.9, fontsize=8.5,
           handlelength=1.6, borderpad=0.4, labelspacing=0.3)

# RA vs Dec coverage
ax2.plot(ra_set, dec_set, ".", ms=0.9, color=C_SETTING, alpha=0.30,
         rasterized=True, label="Setting")
ax2.plot(ra_ris, dec_ris, ".", ms=0.9, color=C_RISING, alpha=0.30,
         rasterized=True, label="Rising")
ov_lo = max(ra_set.min(), ra_ris.min())
ov_hi = min(ra_set.max(), ra_ris.max())
ax2.axvspan(ov_lo, ov_hi, color=C_GRID, alpha=0.14,
            label=rf"RA overlap (${ov_hi - ov_lo:.0f}^\circ$)")
ax2.set_xlabel("Right Ascension [deg]")
ax2.set_ylabel("Declination [deg]")
ax2.set_title(rf"Sky coverage  (el $= {EL_SCAN:.0f}^\circ$)", fontsize=11)
ax2.invert_xaxis()
ax2.set_xlim(195, 165)  # RA increasing to the left
ax2.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=8.5,
           handlelength=1.2, borderpad=0.3, columnspacing=1.0, handletextpad=0.4)
ax2.set_aspect(1 / np.cos(np.radians(center_dec)))

plt.tight_layout(pad=0.6)
plt.savefig("azimuthal_scan.pdf")
plt.savefig("azimuthal_scan.png", dpi=300)
plt.close()
print("Saved azimuthal_scan.{pdf,png}")

print("\nAll figures generated.")
