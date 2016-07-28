import warnings
from nose.tools import raises
from numpy.testing import assert_almost_equal, assert_equal
from pymot.boundingbox import BoundingBox


def test_bb_constructor_int_id():
    bb = BoundingBox(3, 5, 10, 15, 7)
    assert_equal(bb.x, 3)
    assert_equal(bb.y, 5)
    assert_equal(bb.width, 10)
    assert_equal(bb.height, 15)
    assert bb.id == '7'


def test_bb_constructor_string_id():
    bb = BoundingBox(3, 5, 10, 15, 'referee')
    assert_equal(bb.x, 3)
    assert_equal(bb.y, 5)
    assert_equal(bb.width, 10)
    assert_equal(bb.height, 15)
    assert bb.id == 'referee'


def test_bb_constructor_default_id():
    bb = BoundingBox(3, 5, 10, 15)
    assert_equal(bb.x, 3)
    assert_equal(bb.y, 5)
    assert_equal(bb.width, 10)
    assert_equal(bb.height, 15)
    assert bb.id == ''


@raises(ValueError)
def test_bb_constructor_negative_width():
    BoundingBox(3, 5, -10, 15)


@raises(ValueError)
def test_bb_constructor_negative_height():
    BoundingBox(3, 5, 10, -15)
    
    
def test_bb_init_from_dic_int_id():
    bb_dic = {'x': 3,
              'y': 5,
              'width': 10,
              'height': 15,
              'id': 7}
    bb = BoundingBox.init_from_dic(bb_dic)
    assert_equal(bb.x, 3)
    assert_equal(bb.y, 5)
    assert_equal(bb.width, 10)
    assert_equal(bb.height, 15)
    assert bb.id == '7'


def test_bb_init_from_dic_string_id():
    bb_dic = {'x': 3,
              'y': 5,
              'width': 10,
              'height': 15,
              'id': 'referee'}
    bb = BoundingBox.init_from_dic(bb_dic)
    assert_equal(bb.x, 3)
    assert_equal(bb.y, 5)
    assert_equal(bb.width, 10)
    assert_equal(bb.height, 15)
    assert bb.id == 'referee'


def test_bb_init_from_dic_default_id():
    bb_dic = {'x': 3,
              'y': 5,
              'width': 10,
              'height': 15}
    bb = BoundingBox.init_from_dic(bb_dic)
    assert_equal(bb.x, 3)
    assert_equal(bb.y, 5)
    assert_equal(bb.width, 10)
    assert_equal(bb.height, 15)
    assert bb.id == ''


@raises(ValueError)
def test_bb_init_from_dic_negative_width():
    bb_dic = {'x': 3,
              'y': 5,
              'width': -10,
              'height': 15}
    BoundingBox.init_from_dic(bb_dic)


@raises(ValueError)
def test_bb_init_from_dic_negative_height():
    bb_dic = {'x': 3,
              'y': 5,
              'width': 10,
              'height': -15}
    BoundingBox.init_from_dic(bb_dic)


def test_bb_area():
    bb = BoundingBox(3, 5, 10, 15)
    assert_equal(bb.area(), 150)


def test_bb_no_intersect():
    with warnings.catch_warnings(record=True) as w:
        bb1 = BoundingBox(3, 5, 10, 15)
        bb2 = BoundingBox(1, 2, 1, 1)
        intersect = bb1.intersect(bb2)
        assert_equal(intersect.x, 3)
        assert_equal(intersect.y, 5)
        assert_equal(intersect.width, 0)
        assert_equal(intersect.height, 0)
        assert intersect.id == 'intersect'
        assert str(w[0].message.args[0]) == 'Bounding boxes do not intersect'


def test_bb_some_intersect():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 4, 15)
    intersect = bb1.intersect(bb2)
    assert_equal(intersect.x, 3)
    assert_equal(intersect.y, 5)
    assert_equal(intersect.width, 2)
    assert_equal(intersect.height, 12)
    assert intersect.id == 'intersect'


def test_bb_all_intersect():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(3, 5, 10, 15)
    intersect = bb1.intersect(bb2)
    assert_equal(intersect.x, bb1.x)
    assert_equal(intersect.x, bb2.x)
    assert_equal(intersect.y, bb1.y)
    assert_equal(intersect.y, bb2.y)
    assert_equal(intersect.width, bb1.width)
    assert_equal(intersect.width, bb2.width)
    assert_equal(intersect.height, bb1.height)
    assert_equal(intersect.height, bb2.height)
    assert intersect.id == 'intersect'


def test_bb_no_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 1, 1)
    iou = bb1.intersection_over_union(bb2)
    assert_equal(iou, 0)


def test_bb_some_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 4, 15)
    iou = bb1.intersection_over_union(bb2)
    assert_equal(iou, 4 / 31)


def test_bb_half_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(8, 5, 10, 15)
    iou = bb1.intersection_over_union(bb2)
    assert_equal(iou, 1 / 3)


def test_bb_all_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(3, 5, 10, 15)
    iou = bb1.intersection_over_union(bb2)
    assert_equal(iou, 1)


def test_bb_zero_center_distance():
    bb1 = BoundingBox(0, 0, 20, 20)
    bb2 = BoundingBox(5, 5, 10, 10)
    d = bb1.center_distance(bb2)
    assert_equal(d, 0)


def test_bb_center_distance():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 2, 1)
    d = bb1.center_distance(bb2)
    assert_almost_equal(d, 11.6619037897)
    