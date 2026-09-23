"""The sidereal times a drift scan visited, and what they can resolve.

Everything downstream is set by the window: the Rayleigh width ``2 pi / T``
is the narrowest feature in ``m`` the arc can separate, the coverage fraction
decides whether the low modes are separable at all, and the Shannon number
``(2M + 1) x coverage`` counts how many of the ``2M + 1`` real unknowns below
a truncation ``M`` the data actually constrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["LSTWindow"]

_FULL_TURN = 360.0
#: A uniformly sampled arc whose wrap-around gap equals one cadence to this
#: relative tolerance is closed: the smoothing is then periodic and no edge is
#: discarded. Anything else is an open arc, however small the gap -- a seam
#: treated as periodic with the wrong spacing biases every mode.
_CLOSED_RTOL = 1e-3
#: Cadence agreement required for a window to count as uniformly sampled.
_UNIFORM_RTOL = 1e-6


@dataclass(frozen=True, eq=False)
class LSTWindow:
    """Local sidereal time of every sample, in degrees, with optional weights.

    Parameters
    ----------
    lst_deg : ndarray, shape (n,)
        Sidereal time of each sample **in time order**. Need not be uniform or
        contiguous: several nights stacked, or a night with gaps, are one
        window. A scan that crosses 0 h may be given either unwrapped
        (``350, 355, 360, 365``) or wrapped (``350, 355, 0, 5``); one downward
        jump is read as the wrap. Uniformity is required only by the
        smoothing kernel, which says so.
    weights : ndarray, shape (n,), optional
        Inverse-variance weight per sample, e.g. how many times an LST bin was
        visited. Strictly positive: a sample with no weight is not a sample.
        Default: all ones.
    t_ref_deg : float
        The LST the beam harmonics were evaluated at; phases are measured
        from it so that ``d_m`` matches ``B_lm(t_ref)``.

    Equality is identity: two windows built from the same arrays are
    different objects.
    """

    lst_deg: np.ndarray
    weights: Optional[np.ndarray] = None
    t_ref_deg: float = 0.0
    _phase: np.ndarray = field(init=False, repr=False)
    _unwrapped: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lst = np.asarray(self.lst_deg, dtype=float).ravel()
        if lst.size < 2:
            raise ValueError("An LST window needs at least two samples.")
        if not np.all(np.isfinite(lst)):
            raise ValueError("LST values must be finite.")
        w = np.ones(lst.size) if self.weights is None else np.asarray(self.weights, dtype=float).ravel()
        if w.shape != lst.shape:
            raise ValueError(f"weights has shape {w.shape}, expected {lst.shape}.")
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and strictly positive; drop unweighted samples.")
        object.__setattr__(self, "lst_deg", lst)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "_phase", np.deg2rad(lst - self.t_ref_deg))
        object.__setattr__(self, "_unwrapped", _unwrap(lst))

    @classmethod
    def uniform_arc(cls, lst_start_deg: float, span_deg: float, cadence_s: float = 120.0,
                    t_ref_deg: float = 0.0) -> "LSTWindow":
        """One contiguous arc at a fixed cadence in *sidereal* seconds.

        ``span_deg >= 360`` gives a closed scan: the last sample stops one
        cadence short of the first so no LST is counted twice.
        """
        step = cadence_s / 86164.0905 * _FULL_TURN          # sidereal seconds -> degrees
        if step <= 0 or span_deg <= 0:
            raise ValueError("cadence_s and span_deg must be positive.")
        if span_deg >= _FULL_TURN:
            n = int(round(_FULL_TURN / step))
            lst = lst_start_deg + np.arange(n) * (_FULL_TURN / n)
        else:
            n = int(round(span_deg / step)) + 1
            lst = np.linspace(lst_start_deg, lst_start_deg + span_deg, n)
        return cls(lst, t_ref_deg=t_ref_deg)

    @classmethod
    def stacked_nights(cls, nights: int, lst_start_deg: float, span_deg: float,
                       cadence_s: float = 120.0, drift_deg_per_day: float = 0.9856,
                       bin_deg: float = 1.0, t_ref_deg: float = 0.0) -> "LSTWindow":
        """The same wall-clock window every night, binned in LST with visit counts as weights.

        The window drifts by ``drift_deg_per_day`` in sidereal time, so
        integrating longer is also covering more of the circle. The bins are
        returned in LST order, uniformly spaced only once every bin is hit.
        """
        if nights < 1:
            raise ValueError("nights must be at least 1.")
        per_night = cls.uniform_arc(0.0, span_deg, cadence_s).n
        bins = np.zeros(int(round(_FULL_TURN / bin_deg)))
        offsets = np.linspace(0.0, span_deg, per_night)
        for k in range(nights):
            idx = (np.floor((lst_start_deg + drift_deg_per_day * k + offsets) / bin_deg)
                   % bins.size).astype(int)
            np.add.at(bins, idx, 1.0)
        seen = np.nonzero(bins)[0]
        return cls((seen + 0.5) * bin_deg, weights=bins[seen], t_ref_deg=t_ref_deg)

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.lst_deg.size

    @property
    def phase(self) -> np.ndarray:
        """``t_i - t_ref`` in radians."""
        return self._phase

    @property
    def cadence_deg(self) -> Optional[float]:
        """Sample spacing in degrees if the window is uniform *and in time order*, else ``None``."""
        d = np.diff(self._unwrapped)
        if d.size == 0 or d[0] <= 0:
            return None
        return float(d[0]) if np.allclose(d, d[0], rtol=_UNIFORM_RTOL, atol=1e-9) else None

    @property
    def closed(self) -> bool:
        """Whether the arc meets itself exactly one cadence after its last sample."""
        c = self.cadence_deg
        if c is None:
            return False
        gap = _FULL_TURN - (self._unwrapped[-1] - self._unwrapped[0])
        return abs(gap - c) <= _CLOSED_RTOL * c

    @property
    def span_deg(self) -> float:
        """Extent of the arc, degrees (a closed scan reports 360)."""
        if self.closed:
            return _FULL_TURN
        return float(min(np.ptp(self._unwrapped), _FULL_TURN))

    @property
    def coverage(self) -> float:
        """Fraction of the sidereal circle within half a sample spacing of a sample.

        Each sample claims the smaller of its two gaps to its neighbours on
        the circle and one nominal spacing (the median gap, at least 1 deg),
        so a uniformly sampled closed scan is exactly 1 at any cadence and a
        sparse window is not credited for the space between its samples.
        """
        s = np.unique(np.round(np.mod(self.lst_deg, _FULL_TURN), 9))    # revisits of one LST count once
        if s.size == 1:
            return 1.0 / _FULL_TURN
        gaps = np.diff(np.concatenate([s, [s[0] + _FULL_TURN]]))
        nominal = max(1.0, float(np.median(gaps)))
        frac = np.minimum(gaps, nominal).sum() / _FULL_TURN
        return 1.0 if frac > 1.0 - 1e-9 else float(frac)

    @property
    def nyquist_m(self) -> float:
        """The highest ``m`` the sampling can represent: ``180 / (smallest spacing in degrees)``."""
        d = np.diff(np.sort(np.mod(self.lst_deg, _FULL_TURN)))
        d = d[d > 1e-9]
        return float(180.0 / d.min()) if d.size else float("inf")

    @property
    def rayleigh(self) -> float:
        """``2 pi / T``: the narrowest feature in ``m`` the arc can resolve."""
        return _FULL_TURN / self.span_deg

    def shannon_number(self, m_trunc: int) -> float:
        """How many of the ``2M + 1`` real unknowns the window constrains."""
        return (2 * m_trunc + 1) * self.coverage


def _unwrap(lst: np.ndarray) -> np.ndarray:
    """Time-ordered LST made monotone: one downward jump of more than half a turn is a 0 h crossing."""
    d = np.diff(lst)
    if np.any((d < 0) & (d > -_FULL_TURN / 2)):
        return lst            # a small step backwards is not a wrap: not time-ordered
    drops = np.nonzero(d < 0)[0]
    if drops.size == 0:
        return lst
    if drops.size == 1:
        out = lst.copy()
        out[drops[0] + 1:] += _FULL_TURN
        if np.all(np.diff(out) > 0):
            return out
    return lst            # not time-ordered: cadence_deg will report None
