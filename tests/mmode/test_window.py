import numpy as np
import pytest

from limTOD.mmode import LSTWindow


def test_closed_scan_has_no_duplicate_lst_and_full_coverage(closed):
    assert closed.closed
    assert closed.n == 718
    assert closed.coverage == 1.0
    assert np.isclose(closed.span_deg, 360.0)
    assert np.isclose(closed.rayleigh, 1.0)
    assert len(np.unique(np.round(closed.lst_deg % 360, 6))) == closed.n


def test_open_arc_geometry(arc7h):
    assert not arc7h.closed
    assert arc7h.n == 211
    assert np.isclose(arc7h.span_deg, 105.288)
    assert np.isclose(arc7h.rayleigh, 360 / 105.288)
    assert 0.29 < arc7h.coverage < 0.30
    assert np.isclose(arc7h.shannon_number(6), 13 * arc7h.coverage)


@pytest.mark.parametrize("span", [359.0, 359.4, 359.6, 359.9])
@pytest.mark.parametrize("cadence", [120.0, 600.0])
def test_almost_closed_arcs_are_open(span, cadence):
    """A seam that is not exactly one cadence must not be treated as periodic."""
    w = LSTWindow.uniform_arc(0.0, span, cadence_s=cadence)
    assert not w.closed
    assert w.span_deg < 360.0


@pytest.mark.parametrize("cadence", [120.0, 300.0, 600.0, 1200.0])
def test_coverage_is_one_for_a_closed_scan_at_any_cadence(cadence):
    assert LSTWindow.uniform_arc(0.0, 360.0, cadence_s=cadence).coverage == 1.0


def test_coverage_does_not_credit_gaps():
    w = LSTWindow(np.array([0.0, 1.0, 2.0, 180.0, 181.0, 182.0]))
    assert w.coverage == pytest.approx(6 / 360)


def test_wrapped_and_unwrapped_scans_agree():
    a = LSTWindow.uniform_arc(350.0, 20.0)
    b = LSTWindow(np.mod(a.lst_deg, 360.0))
    assert b.cadence_deg == pytest.approx(a.cadence_deg)
    assert b.span_deg == pytest.approx(a.span_deg)
    assert not b.closed


def test_phase_is_measured_from_t_ref():
    w = LSTWindow(np.array([10.0, 20.0]), t_ref_deg=10.0)
    assert np.allclose(w.phase, np.deg2rad([0.0, 10.0]))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least two"):
        LSTWindow(np.array([1.0]))
    with pytest.raises(ValueError, match="weights"):
        LSTWindow(np.array([1.0, 2.0]), weights=np.array([1.0]))
    with pytest.raises(ValueError, match="positive"):
        LSTWindow(np.array([1.0, 2.0]), weights=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="positive"):
        LSTWindow.uniform_arc(0.0, -5.0)
    with pytest.raises(ValueError, match="nights"):
        LSTWindow.stacked_nights(0, 0.0, 10.0)


def test_non_uniform_or_unordered_window_reports_no_cadence():
    assert LSTWindow(np.array([0.0, 1.0, 3.0, 6.0])).cadence_deg is None
    shuffled = LSTWindow(np.array([3.0, 1.0, 0.0, 2.0]))
    assert shuffled.cadence_deg is None and not shuffled.closed


def test_windows_compare_by_identity():
    a = LSTWindow.uniform_arc(0.0, 10.0)
    b = LSTWindow.uniform_arc(0.0, 10.0)
    assert a != b and a == a
    assert hash(a) != hash(b)


def test_stacked_nights_close_the_circle_in_about_260_nights():
    one = LSTWindow.stacked_nights(1, 338.816, 105.288)
    assert one.n == 107 and np.all(one.weights > 0)      # 338.8 .. 444.1 deg -> bins 338..359, 0..84
    assert LSTWindow.stacked_nights(240, 338.816, 105.288).coverage < 1.0
    full = LSTWindow.stacked_nights(260, 338.816, 105.288)
    assert full.coverage == 1.0 and full.closed
    assert full.weights.sum() == pytest.approx(260 * 211, rel=1e-6)


def test_revisits_of_one_lst_do_not_reduce_coverage():
    one = LSTWindow(np.arange(0.0, 360.0, 2.5))
    two = LSTWindow(np.r_[np.arange(0.0, 360.0, 2.5)] * 1, weights=None)
    assert one.coverage == 1.0
    assert LSTWindow(np.tile(np.arange(0.0, 360.0, 2.5), 3)).coverage == 1.0
    night = np.arange(0.0, 100.0, 0.5)
    assert LSTWindow(np.tile(night, 3)).coverage == pytest.approx(LSTWindow(night).coverage)


def test_a_small_backward_step_is_not_a_wrap_and_span_stays_positive():
    w = LSTWindow(np.array([10.0, 20.0, 5.0, 15.0]))
    assert w.cadence_deg is None and not w.closed
    assert w.span_deg == pytest.approx(15.0) and w.rayleigh > 0
    assert LSTWindow(np.array([20.0, 15.0, 10.0, 5.0])).span_deg == pytest.approx(15.0)
    assert LSTWindow(np.array([350.0, 355.0, 0.0, 5.0, 350.0, 355.0, 0.0])).span_deg > 0
