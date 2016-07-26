from nose.tools import raises
from numpy.testing import assert_almost_equal
from pymot.base import calculate_mota, calculate_motp


def test_calculate_mota_value():
    assert_almost_equal(calculate_mota(0, 0, 0, 77), 1.0)
    assert_almost_equal(calculate_mota(0, 21, 67, 103), 0.14563106796)
    assert_almost_equal(calculate_mota(34, 0, 67, 103), 0.01941747572)
    assert_almost_equal(calculate_mota(34, 21, 0, 103), 0.46601941747)
    assert_almost_equal(calculate_mota(34, 21, 67, 103), -0.18446601941)

@raises(ValueError)
def test_calculate_mota_zero_gts():
    calculate_mota(0, 0, 0, 0)


@raises(ValueError)
def test_calculate_mota_negative_gts():
    calculate_mota(0, 0, 0, -3)


@raises(ValueError)
def test_calculate_mota_negative_fn():
    calculate_mota(-34, 21, 67, 103)


@raises(ValueError)
def test_calculate_mota_negative_fp():
    calculate_mota(34, -21, 67, 103)


@raises(ValueError)
def test_calculate_mota_negative_is():
    calculate_mota(34, 21, -67, 103)


def test_calculate_motp_value():
    assert_almost_equal(calculate_motp(34.5, 100), 0.345)
    assert_almost_equal(calculate_motp(0, 100), 0)


@raises(ValueError)
def test_calculate_motp_zero_correspondences():
    calculate_motp(0, -3)


@raises(ValueError)
def test_calculate_motp_negative_correspondences():
    calculate_motp(0, -3)


@raises(ValueError)
def test_calculate_motp_negative_overlap():
    calculate_motp(-34.5, 100)
