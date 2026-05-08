import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from limTOD.HPW_filter import HPW_mapmaking


def _identity_mapmaker(n_samples):
    mapper = HPW_mapmaking.__new__(HPW_mapmaking)
    mapper.num_tods = 1
    mapper.npol = 1
    mapper.num_pixels = n_samples
    mapper.nsky_params = n_samples
    mapper.Tsys_others = False
    mapper.Tsys_operators = np.eye(n_samples)
    return mapper


def test_mapmaker_defaults_to_no_high_pass_without_cutoff():
    tod = np.array([1.0, 2.0, 3.0])
    mapper = _identity_mapmaker(len(tod))

    sky_est, sky_unc = mapper(
        TOD_group=tod,
        dtime=1.0,
        cutoff_freq_group=None,
        noise_variance=1.0,
        regularization=0.0,
    )

    np.testing.assert_allclose(sky_est, tod)
    np.testing.assert_allclose(sky_unc, np.ones_like(tod))


def test_mapmaker_requires_cutoff_when_high_pass_enabled():
    tod = np.array([1.0, 2.0, 3.0])
    mapper = _identity_mapmaker(len(tod))

    with pytest.raises(ValueError, match="cutoff_freq_group must be provided"):
        mapper(
            TOD_group=tod,
            dtime=1.0,
            cutoff_freq_group=None,
            use_high_pass=True,
            noise_variance=1.0,
        )
