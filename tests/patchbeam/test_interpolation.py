"""Unit tests for limTOD.patchbeam.interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from limTOD.patchbeam import interpolation


@pytest.mark.unit
def test_on_grid_query_returns_grid_value():
    """Querying at a grid node returns that node's value exactly."""
    grid = np.linspace(-5, 5, 11)
    n_freq = 3
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(n_freq, len(grid), len(grid))).astype(np.float64)

    # Hit every grid node. Cube axis 1 is m, axis 2 is l; ravel in C order
    # iterates (i_m, i_l), so build the query order to match.
    L, M = np.meshgrid(grid, grid, indexing="xy")  # L varies along axis 1 (l), M along axis 0 (m)
    weights = interpolation.precompute_bilinear_weights(L.ravel(), M.ravel(), grid, grid)
    out = interpolation.apply_bilinear(weights, cube)  # (n_freq, n_query)

    expected = cube.reshape(n_freq, -1)
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.unit
def test_linear_function_recovered_exactly():
    """Bilinear interpolation recovers an affine function f(l, m) = a*l + b*m + c."""
    grid = np.linspace(-3, 3, 7)
    a, b, c = 1.5, -0.7, 0.3
    # Cube axes are (n_freq, n_m, n_l), so build M along axis 0 and L along axis 1.
    M, L = np.meshgrid(grid, grid, indexing="ij")
    cube = (a * L + b * M + c)[None, :, :]

    # Random off-grid queries inside the domain.
    rng = np.random.default_rng(2)
    l_q = rng.uniform(-2.5, 2.5, size=50)
    m_q = rng.uniform(-2.5, 2.5, size=50)
    weights = interpolation.precompute_bilinear_weights(l_q, m_q, grid, grid)
    out = interpolation.apply_bilinear(weights, cube)

    expected = a * l_q + b * m_q + c
    np.testing.assert_allclose(out[0], expected, atol=1e-12)


@pytest.mark.unit
def test_out_of_range_returns_zero_and_is_flagged():
    grid = np.linspace(-1, 1, 5)
    cube = np.ones((1, 5, 5))

    l_q = np.array([2.0, 0.5, -2.0])
    m_q = np.array([0.0, 0.0, 0.0])
    weights = interpolation.precompute_bilinear_weights(l_q, m_q, grid, grid)
    out = interpolation.apply_bilinear(weights, cube)

    assert weights.valid.tolist() == [False, True, False]
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0])


@pytest.mark.unit
def test_freq_subset_indexing():
    """Selecting a frequency subset returns the matching slabs."""
    grid = np.linspace(-1, 1, 5)
    cube = np.arange(3 * 5 * 5, dtype=np.float64).reshape(3, 5, 5)
    l_q = np.array([0.0])
    m_q = np.array([0.0])
    weights = interpolation.precompute_bilinear_weights(l_q, m_q, grid, grid)

    out = interpolation.apply_bilinear(weights, cube, freq_indices=np.array([0, 2]))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out[0, 0], cube[0, 2, 2])
    np.testing.assert_allclose(out[1, 0], cube[2, 2, 2])
