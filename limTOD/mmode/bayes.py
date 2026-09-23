"""Bayesian m-mode inference through bayesmith.

The linear solve truncates and smooths so that the modes above ``M`` do not
land on the ones below. The Bayesian route replaces both devices with a
prior: every mode up to ``mmax`` is a latent with a Gaussian prior of width
``envelope[m]``, the data are the *raw* samples with independent noise, and
the prediction is affine in the modes. bayesmith certifies that structure
and draws the posterior exactly (Wiener mean, GCR samples). When the noise
level is itself unknown it is one more latent; the compiler then moves it
with NUTS and re-draws the modes exactly at every step.

bayesmith is an optional dependency (``pip install limTOD[mmode-bayes]``);
everything else in :mod:`limTOD.mmode` works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .linear import design_matrix, unpack
from .window import LSTWindow

__all__ = ["BayesResult", "exact_posterior", "posterior_samples", "build_graph"]

_MISSING = ("bayesmith is not installed. `pip install limTOD[mmode-bayes]` "
            "(or `pip install bayesmith`) to use limTOD.mmode.bayes.")


def _require():
    try:
        import bayesmith  # noqa: F401
        import jax
    except ImportError as exc:                          # pragma: no cover - environment dependent
        raise ImportError(_MISSING) from exc
    jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True, eq=False)
class BayesResult:
    modes: np.ndarray            # posterior mean, (n_freq, mmax+1) complex
    std: Optional[np.ndarray]    # (n_freq, 2*mmax+1) posterior std of the real parameters; None without samples
    samples: Optional[np.ndarray]  # (n_draws, n_freq, mmax+1) complex, or None
    mmax: int
    prior_envelope: np.ndarray
    sigma: np.ndarray

    @property
    def d0(self) -> np.ndarray:
        return self.modes[:, 0].real

    def d0_interval(self, level: float = 0.68) -> np.ndarray:
        """``(2, n_freq)`` central credible interval on ``d_0`` from the samples."""
        if self.samples is None:
            raise ValueError("No samples were drawn; call posterior_samples.")
        lo, hi = 50 * (1 - level), 50 * (1 + level)
        return np.percentile(self.samples[:, :, 0].real, [lo, hi], axis=0)


def build_graph(y: np.ndarray, window: LSTWindow, mmax: int, envelope: Sequence[float],
                sigma: float, prior_mean: Optional[np.ndarray] = None):
    """One channel's graph: modes ~ N(mean, envelope), data ~ N(E modes, sigma / sqrt(w))."""
    _require()
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from bayesmith import det, observe, sample, trace

    e = jnp.asarray(design_matrix(window, mmax))
    env = np.asarray(envelope, dtype=float)
    if env.size != mmax + 1 or np.any(env <= 0):
        raise ValueError("envelope must give a positive width for every m = 0 .. mmax.")
    # prior widths on the real parameters: |d_m| ~ envelope[m] shared by Re and Im
    width = np.concatenate([[env[0]]] + [[env[m], env[m]] for m in range(1, mmax + 1)])
    mean = np.zeros(width.size) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    noise = sigma / np.sqrt(window.weights)

    def model(data):
        p = sample("modes", lambda: dist.Normal(jnp.asarray(mean), jnp.asarray(width)).to_event(1))
        mu = det("mu", lambda p_: e @ p_, p, linear_in=("modes",))
        observe("d", lambda m: dist.Normal(m, jnp.asarray(noise)), mu, obs=jnp.asarray(data))

    return trace(model, np.asarray(y, dtype=float))


def exact_posterior(tod: np.ndarray, window: LSTWindow, mmax: int, envelope: Sequence[float],
                    sigma: Union[float, np.ndarray] = 1.0) -> BayesResult:
    """Posterior mean of every mode, channel by channel, by the certified exact route."""
    _require()
    from bayesmith.dispatch.classify import prior_environment
    from bayesmith.exact.gaussian import precision_at
    from bayesmith.exact.linearity import linear_operator
    from bayesmith.exact.solve import wiener_solve

    y = np.asarray(tod, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    nf = y.shape[1]
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (nf,))
    means = np.zeros((nf, 2 * mmax + 1))
    for f in range(nf):
        graph = build_graph(y[:, f], window, mmax, envelope, float(sig[f]))
        block = linear_operator(graph, ("modes",), at={})
        centres = prior_environment(graph)
        m, _ = wiener_solve(block, precision=precision_at(graph, {"modes": centres["modes"]}))
        means[f] = np.asarray(m["modes"])
    env = np.asarray(envelope, dtype=float)
    return BayesResult(unpack(means), None, None, mmax, env, sig)


def posterior_samples(tod: np.ndarray, window: LSTWindow, mmax: int, envelope: Sequence[float],
                      sigma: Union[float, np.ndarray] = 1.0, n_draws: int = 200, seed: int = 0) -> BayesResult:
    """Exact posterior draws (constrained realisations) per channel; errors and intervals from them."""
    _require()
    import jax
    from bayesmith.dispatch.classify import prior_environment
    from bayesmith.exact.gaussian import precision_at
    from bayesmith.exact.linearity import linear_operator
    from bayesmith.exact.solve import gcr_sample

    y = np.asarray(tod, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    nf = y.shape[1]
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (nf,))
    draws = np.zeros((n_draws, nf, 2 * mmax + 1))
    key = jax.random.key(seed)
    for f in range(nf):
        graph = build_graph(y[:, f], window, mmax, envelope, float(sig[f]))
        block = linear_operator(graph, ("modes",), at={})
        prec = precision_at(graph, {"modes": prior_environment(graph)["modes"]})
        for i in range(n_draws):
            key, sub = jax.random.split(key)
            s, _ = gcr_sample(block, precision=prec, key=sub)
            draws[i, f] = np.asarray(s["modes"])
    env = np.asarray(envelope, dtype=float)
    return BayesResult(unpack(draws.mean(axis=0)), draws.std(axis=0), unpack(draws), mmax, env, sig)
