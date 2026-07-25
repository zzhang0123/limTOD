"""Unit tests for limTOD.patchbeam.beam."""

from __future__ import annotations

import numpy as np
import pytest

from limTOD.patchbeam.beam import MeerKLASSBeam, synthetic_gaussian_beam


@pytest.fixture
def gaussian_beam() -> MeerKLASSBeam:
    freq_MHz = np.linspace(800, 1000, 5)
    margin_deg = np.linspace(-6, 6, 121)
    return synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg, fwhm_deg=1.1)


@pytest.mark.unit
def test_beam_extent_and_freq_range(gaussian_beam: MeerKLASSBeam):
    assert gaussian_beam.beam_extent_deg == (-6.0, 6.0)
    assert gaussian_beam.freq_range_MHz == (800.0, 1000.0)
    assert gaussian_beam.polarizations == ("HH", "VV")


@pytest.mark.unit
def test_beam_peak_is_one_at_centre(gaussian_beam: MeerKLASSBeam):
    cube = gaussian_beam.power_cube("HH")
    centre_idx = len(gaussian_beam.margin_deg) // 2
    assert cube[0, centre_idx, centre_idx] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_beam_solid_angle_matches_analytic_gaussian(gaussian_beam: MeerKLASSBeam):
    """For a circular Gaussian power beam exp(-r^2 / sigma^2), Omega = pi * sigma^2."""
    # Power-beam sigma matching synthetic_gaussian_beam's convention.
    sigma_deg = 1.1 / (2.0 * np.sqrt(np.log(2.0)))
    sigma_rad = np.deg2rad(sigma_deg)
    expected = np.pi * sigma_rad**2

    omega = gaussian_beam.beam_solid_angle("HH")
    # Numerical integration on a +/-6 deg patch with the chosen FWHM is
    # accurate to a few parts in 10^3 (truncation outside the grid).
    np.testing.assert_allclose(omega, expected, rtol=5e-3)


@pytest.mark.unit
def test_synthetic_gaussian_beam_has_correct_fwhm():
    """A scan along a single axis hits half-power exactly at r = FWHM/2."""
    freq_MHz = np.array([1000.0])
    margin_deg = np.linspace(-6, 6, 2401)  # 0.005 deg sampling
    target_fwhm = 1.5
    beam = synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg, fwhm_deg=target_fwhm)
    cube = beam.power_cube("HH")[0]  # (n_m, n_l)

    # Slice along m=0: profile(l) -> linearly interpolate the radius at half-max.
    m_centre = cube.shape[0] // 2
    centre_idx = len(margin_deg) // 2
    profile = cube[m_centre, centre_idx:]  # take the l >= 0 half
    half = profile[0] / 2.0  # B(0) = 1
    cross = np.searchsorted(-profile, -half)  # profile is monotonically decreasing
    # Linear interpolation between (cross-1) and cross.
    y0, y1 = profile[cross - 1], profile[cross]
    x0, x1 = margin_deg[centre_idx + cross - 1], margin_deg[centre_idx + cross]
    r_half = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    np.testing.assert_allclose(2.0 * r_half, target_fwhm, rtol=1e-3)


@pytest.mark.unit
def test_freq_indices_aligned():
    freq_MHz = np.linspace(800, 1000, 5)
    margin_deg = np.linspace(-1, 1, 5)
    beam = synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg)

    idx = beam.freq_indices([800.0, 1000.0])
    np.testing.assert_array_equal(idx, [0, 4])


@pytest.mark.unit
def test_freq_indices_misaligned_raises():
    freq_MHz = np.array([800.0, 900.0, 1000.0])
    margin_deg = np.linspace(-1, 1, 3)
    beam = synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg)

    with pytest.raises(ValueError):
        beam.freq_indices([801.0], tol=0.1)


@pytest.mark.unit
def test_evaluate_centre_returns_one(gaussian_beam: MeerKLASSBeam):
    val = gaussian_beam.evaluate(800.0, np.array([0.0]), np.array([0.0]), polarization="HH")
    np.testing.assert_allclose(val[0, 0], 1.0, atol=1e-6)


@pytest.mark.unit
def test_evaluate_vv_pol_matches_hh_for_symmetric_beam(gaussian_beam: MeerKLASSBeam):
    """Synthetic Gaussian uses identical HH/VV cubes; both pols agree."""
    val_hh = gaussian_beam.evaluate(900.0, np.array([0.3]), np.array([-0.2]), polarization="HH")
    val_vv = gaussian_beam.evaluate(900.0, np.array([0.3]), np.array([-0.2]), polarization="VV")
    np.testing.assert_allclose(val_hh, val_vv, atol=1e-7)


@pytest.mark.unit
def test_beam_solid_angle_is_cached(gaussian_beam: MeerKLASSBeam):
    """Second call returns the exact same object (cached)."""
    first = gaussian_beam.beam_solid_angle("HH")
    second = gaussian_beam.beam_solid_angle("HH")
    assert first is second


@pytest.mark.unit
def test_from_arrays_rejects_non_uniform_grid():
    """Non-uniformly spaced margin_deg raises ValueError."""
    margin = np.array([-1.0, -0.5, 0.0, 0.6, 1.0])  # last step is wider
    freq_MHz = np.array([1000.0])
    cube = np.ones((1, len(margin), len(margin)), dtype=np.float32)
    with pytest.raises(ValueError, match="uniformly spaced"):
        MeerKLASSBeam.from_arrays(freq_MHz=freq_MHz, margin_deg=margin, power={"HH": cube})


@pytest.mark.unit
def test_unknown_polarization_raises():
    with pytest.raises(ValueError, match="not supported"):
        MeerKLASSBeam.from_arrays(
            freq_MHz=np.array([1000.0]),
            margin_deg=np.linspace(-1, 1, 3),
            power={"XX": np.ones((1, 3, 3), dtype=np.float32)},
        )


@pytest.mark.unit
def test_npz_file_loader_roundtrip(tmp_path):
    """The real NPZ constructor path (previously only from_arrays was
    covered): antenna selection, pol materialization, power = |Jones|^2."""
    rng = np.random.default_rng(0)
    n_freq, n_grid, n_ant = 2, 9, 3
    margin = np.linspace(-2.0, 2.0, n_grid)
    freq = np.array([950.0, 1050.0])
    jones = rng.standard_normal((4, n_ant, n_freq, n_grid, n_grid)) + 1j * rng.standard_normal(
        (4, n_ant, n_freq, n_grid, n_grid)
    )
    path = tmp_path / "beam.npz"
    np.savez(
        path,
        beam=jones,
        pols=np.array([b"HH", b"HV", b"VH", b"VV"]),
        antnames=np.array([b"m000", b"array_average", b"m001"]),
        freq_MHz=freq,
        margin_deg=margin,
    )

    beam = MeerKLASSBeam(path, antenna="array_average", polarizations=("HH",))
    np.testing.assert_allclose(np.asarray(beam.freq_MHz), freq)
    np.testing.assert_allclose(np.asarray(beam.margin_deg), margin)
    expected = np.abs(jones[0, 1]) ** 2  # HH is Jones index 0; ant index 1
    np.testing.assert_allclose(
        beam.power_cube("HH"), expected.astype(np.float32), rtol=1e-6
    )
    with pytest.raises(KeyError):
        beam.power_cube("VV")  # not materialised
    with pytest.raises(ValueError, match="not found"):
        MeerKLASSBeam(path, antenna="nonexistent")
