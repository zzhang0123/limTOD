"""Generate the drift-scan engine figures in _static/ (run MANUALLY, ~10 min).

Question (docs/driftscan.md): the m-mode path claims to return the *same*
numbers as the generic per-sample path for a fraction of the work. Both
halves of that claim are visual — an agreement plot and a scaling plot — and
neither should be taken on trust from prose.

Protocol: a zenith drift scan at latitude 53.2 deg (Jodrell Bank, the RHINO
site), a chromatic Gaussian beam (FWHM ~ lambda), and the GSM16 sky over
50-100 MHz. Over one sidereal day the beam sweeps the whole ``dec = +53.2``
circle, so the data runs from the bright Galactic-plane crossings (LST 20h-4h)
down to the cold minimum near LST 12.5h, where the north Galactic pole passes
overhead.

Four figures, all written to ``docs/_static``:

* ``engine-waterfall.png``  — the product: one sidereal day of TOD
* ``engine-mmodes.svg``     — the same day in m-space
* ``engine-agreement.svg``  — m-mode vs generic, and the residual
* ``engine-scaling.svg``    — wall clock against band-limit, plus speed-up

Every timed call takes the sky as MAPS, so the numbers include analysing the
sky into harmonic space — the honest cost of one forward evaluation from the
data a caller usually holds. ``engine-benchmark.json`` additionally records
``fast_alms``, the same call with the sky already in harmonic space, which is
what the "hoist the transform" advice on the page is measured against.

Not run at documentation build time: this needs ``limTOD[jax]`` plus healpy
and pygdsm, none of which the Read the Docs environment installs. The outputs
are committed. Single (light) variants only — the RTD theme has no dark mode.

Run:  python docs/make_engine_figures.py             (~10 min)
      python docs/make_engine_figures.py --smoke     (fast plumbing check)
      python docs/make_engine_figures.py --replot    (redraw from the cache)
"""

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import limtod_jax as ltj  # noqa: E402

STATIC = Path(__file__).parent / "_static"
CACHE = Path(__file__).parent / "_engine-figure-data.npz"  # git-ignored
LAT_DEG = 53.2  # Jodrell Bank — the RHINO site
AZ_DEG, EL_DEG = 0.0, 90.0  # zenith drift scan: traces the dec = LAT circle
FWHM_REF_DEG, FREQ_REF_MHZ = 12.0, 70.0  # chromatic beam: FWHM ~ lambda

# Okabe-Ito-adjacent, and legible on the RTD theme's white page.
C = {"fg": "#24292f", "muted": "#57606a", "grid": "#d0d7de",
     "accent": "#0072B2", "warm": "#D55E00", "good": "#1a7f37"}

STYLE = {
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
    "text.color": C["fg"], "axes.labelcolor": C["fg"],
    "axes.edgecolor": C["grid"], "xtick.color": C["muted"],
    "ytick.color": C["muted"], "grid.color": C["grid"],
    "axes.titlecolor": C["fg"], "legend.frameon": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "grid.alpha": 0.55, "grid.linewidth": 0.6,
}


def save(fig, name: str, ext: str) -> None:
    out = STATIC / f"{name}.{ext}"
    fig.savefig(out, bbox_inches="tight", transparent=True,
                dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {out.relative_to(STATIC.parent.parent)}")


# ------------------------------------------------------------ ingredients ---
def beam_alms(nside: int, lmax: int, freqs_mhz: np.ndarray) -> jnp.ndarray:
    """Chromatic Gaussian beam, beam-local alms — as numpy limTOD builds them.

    Normalized to unit pixel sum (limTOD's ``example_beam_map`` convention), so
    the un-normalized beam-weighted sum this path returns is already close to a
    beam-weighted AVERAGE in kelvin rather than an arbitrary scale. Close, not
    equal — that gap is its own section on the page.
    """
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    rows = []
    for f in freqs_mhz:
        fwhm = np.deg2rad(FWHM_REF_DEG * FREQ_REF_MHZ / f)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        beam = np.exp(-0.5 * (theta / sigma) ** 2)
        rows.append(hp.map2alm(beam / beam.sum(), lmax=lmax))
    return jnp.asarray(np.array(rows))


def gsm_sky(nside: int, freqs_mhz: np.ndarray) -> tuple[jnp.ndarray, str]:
    try:
        from pygdsm import GlobalSkyModel16

        gsm = GlobalSkyModel16()
        maps = np.array([hp.ud_grade(gsm.generate(f), nside) for f in freqs_mhz])
        return jnp.asarray(maps), "GSM16"
    except Exception as exc:  # noqa: BLE001 — any pygdsm failure falls back
        print(f"  ! GSM unavailable ({exc!r}); using a synthetic sky")
        rng = np.random.default_rng(0)
        base = 200.0 * np.abs(rng.normal(size=hp.nside2npix(nside)))
        base = hp.smoothing(base, fwhm=np.deg2rad(10.0))
        spec = (freqs_mhz / FREQ_REF_MHZ) ** -2.6
        return jnp.asarray(spec[:, None] * base[None, :]), "synthetic"


def zyz_rows(lst_deg: np.ndarray) -> jnp.ndarray:
    """(n_time, 3) ZYZ angles for the SAME drift scan, for the generic path."""
    n = lst_deg.shape[0]
    return jnp.stack(
        ltj.zyz_of_pointing(
            jnp.asarray(lst_deg), LAT_DEG,
            jnp.full(n, AZ_DEG), jnp.full(n, EL_DEG), jnp.zeros(n),
        ),
        axis=-1,
    )


# --------------------------------------------------------------- the paths --
# Every forward below is SINGLE-CHANNEL and jitted once, then looped over
# frequency in Python. Not vmapped: the generic path holds O(lmax^3) of Wigner
# intermediates per sample, so batching the frequency axis multiplies that by
# n_freq — and at lmax 191 that is the difference between fitting in memory
# and not. Each takes the sky as a MAP, so the timing includes analysing it.
def analyse(sky_map, *, nside, lmax):
    return ltj.map2alm_quad(sky_map, nside=nside, lmax=lmax)


def make_generic(zyz, *, nside, lmax):
    """Generic per-sample path: one Wigner rotation of the beam per sample."""

    @eqx.filter_jit
    def forward(beam_alm, sky_map):
        sky_alm = analyse(sky_map, nside=nside, lmax=lmax)
        return ltj.generate_tod_sky(beam_alm, sky_alm, zyz, lmax=lmax)

    return forward


def make_mmode(lst_deg, *, nside, lmax, uniform):
    """m-mode path, operator built INSIDE the call — pays the rotation on every call."""

    @eqx.filter_jit
    def forward(beam_alm, sky_map):
        sky_alm = analyse(sky_map, nside=nside, lmax=lmax)
        op = ltj.DriftScanMmode.from_pointing(
            beam_alm, lst_deg, LAT_DEG, AZ_DEG, EL_DEG, lmax=lmax,
            lst_ref_deg=0.0, uniform_sampling=uniform,
        )
        return op(sky_alm)

    return forward


def make_cached(*, nside, lmax):
    """m-mode path with the reference rotation already paid (see build_ops)."""

    @eqx.filter_jit
    def forward(op, sky_map):
        return op(analyse(sky_map, nside=nside, lmax=lmax))

    return forward


@eqx.filter_jit
def cached_forward_alms(op, sky_alm):
    """The same call with the sky ALREADY in harmonic space."""
    return op(sky_alm)


def build_ops(beam_alm, lst_deg, *, lmax, uniform):
    """One reference-frame operator per frequency — the O(lmax^3) step, once."""
    return [
        ltj.DriftScanMmode.from_pointing(
            beam_alm[i], lst_deg, LAT_DEG, AZ_DEG, EL_DEG, lmax=lmax,
            lst_ref_deg=0.0, uniform_sampling=uniform,
        )
        for i in range(beam_alm.shape[0])
    ]


def best_of(fn, *args, repeats: int = 3) -> float:
    jax.block_until_ready(fn(*args))  # compile
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        out.append(time.perf_counter() - t0)
    return min(out)


# ---------------------------------------------------------------- figures ---
def figure_waterfall(lst, freqs_mhz, tod, sky_kind: str) -> None:
    """The product: one sidereal day of drift-scan data, 2-panel.

    The colour scale is logarithmic because the sky is a steep power law: on a
    linear scale the lowest channel saturates the map and the Galactic transit
    — the actual signal — is invisible everywhere else.
    """
    from matplotlib.colors import LogNorm

    lst_h = lst / 15.0
    with plt.rc_context(STYLE):
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(7.6, 5.6), sharex=True, height_ratios=[2.1, 1],
            constrained_layout=True,
        )
        im = ax0.pcolormesh(lst_h, freqs_mhz, tod.T, cmap="magma",
                            shading="auto",
                            norm=LogNorm(vmin=tod.min(), vmax=tod.max()))
        ax0.set_ylabel("frequency  [MHz]")
        ax0.set_title(f"One sidereal day of drift-scan data — {sky_kind} sky, "
                      f"zenith beam at latitude {LAT_DEG:g}°", loc="left")
        cb = fig.colorbar(im, ax=ax0, pad=0.015)
        cb.set_label("$T_{\\rm ant}$  [K]")
        cb.outline.set_edgecolor(C["grid"])
        cb.ax.tick_params(color=C["grid"])

        picks = [0, len(freqs_mhz) // 2, len(freqs_mhz) - 1]
        for i, idx in enumerate(picks):
            ax1.semilogy(lst_h, tod[:, idx], lw=1.5,
                         color=[C["accent"], C["good"], C["warm"]][i],
                         label=f"{freqs_mhz[idx]:.0f} MHz")
        ax1.set_xlabel("local sidereal time  [hours]")
        ax1.set_ylabel("$T_{\\rm ant}$  [K]")
        ax1.set_xlim(lst_h[0], lst_h[-1])
        ax1.grid(True, which="both")
        ax1.legend(ncol=3, loc="lower left", fontsize=8)
        ax1.set_title(f"Every channel sees the same sky drift past the "
                      f"$\\delta={LAT_DEG:g}°$ circle", loc="left", fontsize=9)
        save(fig, "engine-waterfall", "png")


def figure_agreement(lst, freqs_mhz, tod_generic, tod_mmode) -> None:
    """Same physics, two paths: overlay + residual at float64 roundoff."""
    lst_h = lst / 15.0
    scale = np.max(np.abs(tod_generic))
    resid = np.abs(tod_mmode - tod_generic) / scale
    worst = float(resid.max())
    with plt.rc_context(STYLE):
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(7.6, 4.6), sharex=True, height_ratios=[2, 1],
            constrained_layout=True,
        )
        step = max(1, len(lst_h) // 26)  # sparse enough to read as markers
        for i in range(tod_generic.shape[1]):
            ax0.plot(lst_h, tod_generic[:, i], lw=1.6, color=C["muted"],
                     label="generic per-sample path" if i == 0 else None, zorder=1)
            ax0.plot(lst_h[::step], tod_mmode[::step, i], ls="none", marker="o",
                     ms=4.5, mfc="none", mew=1.3, color=C["accent"],
                     label="m-mode path" if i == 0 else None, zorder=2)
        ax0.set_ylabel("$T_{\\rm ant}$  [K]")
        ax0.set_title("The m-mode path is not an approximation", loc="left")
        ax0.grid(True)
        ax0.legend(loc="upper right")

        for i in range(resid.shape[1]):
            ax1.semilogy(lst_h, np.maximum(resid[:, i], 1e-18), lw=0.9,
                         color=C["accent"], alpha=0.75)
        # A legend entry, not floating text: the residual band spans the whole
        # panel width, so any in-plot label lands on top of data.
        ax1.axhline(2.22e-16, ls="--", lw=1.0, color=C["warm"],
                    label="float64 eps")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_ylim(1e-18, 1e-11)
        ax1.set_ylabel("$|\\Delta| / \\max|T|$")
        ax1.set_xlabel("local sidereal time  [hours]")
        ax1.set_xlim(lst_h[0], lst_h[-1])
        ax1.grid(True)
        ax1.set_title(f"worst disagreement {worst:.1e} — float64 roundoff",
                      loc="left", fontsize=9)
        save(fig, "engine-agreement", "svg")


def figure_scaling(bench: list[dict]) -> None:
    """Wall-clock vs band-limit: O(n_t*lmax^3) against O(lmax^3 + n_t*lmax)."""
    lmax = np.array([b["lmax"] for b in bench])
    gen = np.array([b["generic"] for b in bench]) * 1e3
    mm = np.array([b["mmode"] for b in bench]) * 1e3
    fast = np.array([b["fast"] for b in bench]) * 1e3
    with plt.rc_context(STYLE):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 3.5),
                                       constrained_layout=True)
        ax0.loglog(lmax, gen, "o-", color=C["warm"], lw=1.8, ms=5,
                   label="generic per-sample  $O(n_t\\,\\ell_{\\max}^3)$")
        ax0.loglog(lmax, mm, "s-", color=C["accent"], lw=1.8, ms=5,
                   label="m-mode  $O(\\ell_{\\max}^3 + n_t\\ell_{\\max})$")
        ax0.loglog(lmax, fast, "^--", color=C["good"], lw=1.6, ms=5,
                   label="m-mode + cached beam + FFT")
        ax0.set_xlabel("harmonic band-limit  $\\ell_{\\max}$")
        ax0.set_ylabel("one forward evaluation  [ms]")
        ax0.set_title("Cost of one sidereal day", loc="left")
        ax0.grid(True, which="both")
        # A log axis over 23-191 otherwise labels only the decade, 10^2. Label
        # the band-limits actually measured instead.
        ax0.set_xticks(lmax, [str(v) for v in lmax], minor=False)
        ax0.set_xticks([], minor=True)
        ax0.legend(fontsize=8, loc="upper left")

        x = np.arange(len(lmax))
        ax1.bar(x - 0.19, gen / mm, width=0.36, color=C["accent"],
                label="m-mode")
        ax1.bar(x + 0.19, gen / fast, width=0.36, color=C["good"],
                label="+ cached beam + FFT")
        for xi, (a, b) in enumerate(zip(gen / mm, gen / fast, strict=True)):
            ax1.text(xi - 0.19, a, f"{a:.0f}×", ha="center", va="bottom",
                     fontsize=8, color=C["fg"])
            ax1.text(xi + 0.19, b, f"{b:.0f}×", ha="center", va="bottom",
                     fontsize=8, color=C["fg"])
        ax1.set_xticks(x, [str(v) for v in lmax])
        ax1.set_xlabel("harmonic band-limit  $\\ell_{\\max}$")
        ax1.set_ylabel("speed-up over the generic path")
        ax1.set_title("Same numbers, a fraction of the work", loc="left")
        ax1.margins(x=0.06, y=0.18)  # x: keep the last "N×" label inside
        ax1.grid(True, axis="y")
        ax1.legend(fontsize=8, loc="upper left")
        save(fig, "engine-scaling", "svg")


def figure_mmodes(mmodes: np.ndarray, freqs_mhz: np.ndarray) -> None:
    """|V_m| — what the drift scan actually measures."""
    m = np.arange(mmodes.shape[1])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
        colors = [C["accent"], C["good"], C["warm"]]
        picks = [0, mmodes.shape[0] // 2, mmodes.shape[0] - 1]
        for i, idx in enumerate(picks):
            amp = np.abs(mmodes[idx])
            ax.semilogy(m, np.maximum(amp / amp[0], 1e-12), lw=1.4,
                        color=colors[i], label=f"{freqs_mhz[idx]:.0f} MHz")
        ax.set_xlabel("m-mode index  $m$")
        ax.set_ylabel("$|\\tilde V_m| / |\\tilde V_0|$")
        ax.set_title("The drift scan measures a handful of m-modes", loc="left")
        ax.set_xlim(0, m[-1])
        ax.grid(True, which="both")
        ax.legend(ncol=3)
        save(fig, "engine-mmodes", "svg")


# ------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny configuration: check the plumbing, not the physics")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cached run — no physics, no waiting. "
                         "Use for any change that is purely visual.")
    args = ap.parse_args()

    if args.replot:
        cache = np.load(CACHE)
        figure_waterfall(cache["lst"], cache["freqs_mhz"], cache["tod_mmode"],
                         str(cache["sky_kind"]))
        figure_agreement(cache["lst"], cache["freqs_shown"],
                         cache["tod_generic_shown"], cache["tod_mmode_shown"])
        figure_mmodes(cache["mmodes"], cache["freqs_mhz"])
        figure_scaling(json.loads(str(cache["bench"])))
        print("  redrawn from cache")
        return

    if args.smoke:
        nside, n_time, n_freq = 8, 64, 4
        bench_nsides = (4, 8)
    else:
        nside, n_time, n_freq = 64, 512, 32
        bench_nsides = (8, 16, 32, 64)

    lmax = 3 * nside - 1
    freqs_mhz = np.linspace(50.0, 100.0, n_freq)
    lst = np.linspace(0.0, 360.0, n_time, endpoint=False)
    uniform = 2 * lmax < n_time

    print(f"main run: nside={nside} lmax={lmax} n_time={n_time} n_freq={n_freq}")
    alms = beam_alms(nside, lmax, freqs_mhz)
    sky, sky_kind = gsm_sky(nside, freqs_mhz)
    zyz = zyz_rows(lst)

    print("  running the m-mode path ...")
    ops = build_ops(alms, lst, lmax=lmax, uniform=uniform)
    cached = make_cached(nside=nside, lmax=lmax)
    tod_mmode = np.stack(
        [np.asarray(cached(op, sky[i])) for i, op in enumerate(ops)], axis=-1)

    # The generic path costs ~a minute PER CHANNEL at this band-limit, so it
    # runs only on the channels the agreement figure actually draws.
    idx = list(range(0, n_freq, max(1, n_freq // 3)))
    print(f"  running the generic path on {len(idx)} channels (the slow one) ...")
    generic = make_generic(zyz, nside=nside, lmax=lmax)
    tod_generic_shown = np.stack(
        [np.asarray(generic(alms[i], sky[i])) for i in idx], axis=-1)
    tod_mmode_shown = tod_mmode[:, idx]
    worst = float(np.max(np.abs(tod_mmode_shown - tod_generic_shown))
                  / np.max(np.abs(tod_generic_shown)))
    print(f"  agreement: {worst:.3e}")

    mmodes = np.stack([np.asarray(op.mmodes(analyse(sky[i], nside=nside,
                                                    lmax=lmax)))
                       for i, op in enumerate(ops)])

    figure_waterfall(lst, freqs_mhz, tod_mmode, sky_kind)
    figure_agreement(lst, freqs_mhz[idx], tod_generic_shown, tod_mmode_shown)
    figure_mmodes(mmodes, freqs_mhz)

    # ---- scaling benchmark: one frequency, the band-limit is the variable ---
    bench = []
    for ns in bench_nsides:
        lm = 3 * ns - 1
        if 2 * lm >= n_time:
            print(f"  ! skipping nside={ns}: 2*lmax={2 * lm} >= n_time={n_time}")
            continue
        a1 = beam_alms(ns, lm, freqs_mhz[:1])[0]
        s1 = gsm_sky(ns, freqs_mhz[:1])[0][0]
        o1 = build_ops(a1[None], lst, lmax=lm, uniform=True)[0]
        sa1 = analyse(s1, nside=ns, lmax=lm)
        row = {"lmax": lm, "nside": ns,
               "generic": best_of(make_generic(zyz, nside=ns, lmax=lm), a1, s1),
               "mmode": best_of(make_mmode(lst, nside=ns, lmax=lm, uniform=True),
                                a1, s1),
               "fast": best_of(make_cached(nside=ns, lmax=lm), o1, s1),
               "fast_alms": best_of(cached_forward_alms, o1, sa1)}
        bench.append(row)
        print(f"  lmax={lm:4d}  generic {row['generic'] * 1e3:9.1f} ms   "
              f"m-mode {row['mmode'] * 1e3:7.2f} ms  "
              f"({row['generic'] / row['mmode']:.0f}x)   "
              f"fast {row['fast'] * 1e3:7.2f} ms "
              f"({row['generic'] / row['fast']:.0f}x)   "
              f"alms {row['fast_alms'] * 1e3:6.2f} ms")
    figure_scaling(bench)

    (STATIC / "engine-benchmark.json").write_text(json.dumps(
        {"config": {"nside": nside, "lmax": lmax, "n_time": n_time,
                    "n_freq": n_freq, "sky": sky_kind, "lat_deg": LAT_DEG,
                    "az_deg": AZ_DEG, "el_deg": EL_DEG,
                    "uniform_sampling": bool(uniform)},
         "agreement": worst, "scaling": bench}, indent=2) + "\n")
    print("  wrote docs/_static/engine-benchmark.json")

    # Cache everything the figures need: a purely visual change should never
    # cost another full generic-path run.
    np.savez_compressed(
        CACHE, lst=lst, freqs_mhz=freqs_mhz, tod_mmode=tod_mmode,
        freqs_shown=freqs_mhz[idx], tod_generic_shown=tod_generic_shown,
        tod_mmode_shown=tod_mmode_shown, mmodes=mmodes,
        sky_kind=sky_kind, bench=json.dumps(bench),
    )
    print(f"  wrote {CACHE.name} (redraw with --replot)")


if __name__ == "__main__":
    main()
