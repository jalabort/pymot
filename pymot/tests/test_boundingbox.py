from nose.tools import raises
import warnings

from pymot.boundingbox import BoundingBox


def test_bb_constructor_int_id():
    bb = BoundingBox(3, 5, 10, 15, 7)
    assert bb.x == 3
    assert bb.y == 5
    assert bb.width == 10
    assert bb.height == 15
    assert bb.id == '7'


def test_bb_constructor_string_id():
    bb = BoundingBox(3, 5, 10, 15, 'referee')
    assert bb.x == 3
    assert bb.y == 5
    assert bb.width == 10
    assert bb.height == 15
    assert bb.id == 'referee'


def test_bb_constructor_default_id():
    bb = BoundingBox(3, 5, 10, 15)
    assert bb.x == 3
    assert bb.y == 5
    assert bb.width == 10
    assert bb.height == 15
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
    assert bb.x == 3
    assert bb.y == 5
    assert bb.width == 10
    assert bb.height == 15
    assert bb.id == '7'


def test_bb_init_from_dic_string_id():
    bb_dic = {'x': 3,
              'y': 5,
              'width': 10,
              'height': 15,
              'id': 'referee'}
    bb = BoundingBox.init_from_dic(bb_dic)
    assert bb.x == 3
    assert bb.y == 5
    assert bb.width == 10
    assert bb.height == 15
    assert bb.id == 'referee'


def test_bb_init_from_dic_default_id():
    bb_dic = {'x': 3,
              'y': 5,
              'width': 10,
              'height': 15}
    bb = BoundingBox.init_from_dic(bb_dic)
    assert bb.x == 3
    assert bb.y == 5
    assert bb.width == 10
    assert bb.height == 15
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
    assert bb.area() == 150


def test_bb_no_intersect():
    with warnings.catch_warnings(record=True) as w:
        bb1 = BoundingBox(3, 5, 10, 15)
        bb2 = BoundingBox(1, 2, 1, 1)
        intersect = bb1.intersect(bb2)
        assert intersect.x == 3
        assert intersect.y == 5
        assert intersect.width == 0
        assert intersect.height == 0
        assert intersect.id == 'intersect'
        assert str(w[0].message.args[0]) == 'Bounding boxes do not intersect'


def test_bb_some_intersect():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 4, 15)
    intersect = bb1.intersect(bb2)
    assert intersect.x == 3
    assert intersect.y == 5
    assert intersect.width == 2
    assert intersect.height == 12
    assert intersect.id == 'intersect'


def test_bb_all_intersect():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(3, 5, 10, 15)
    intersect = bb1.intersect(bb2)
    assert intersect.x == bb1.x == bb2.x
    assert intersect.y == bb1.y == bb2.y
    assert intersect.width == bb1.width == bb2.width
    assert intersect.height == bb1.height == bb2.height
    assert intersect.id == 'intersect'


def test_bb_no_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 1, 1)
    iou = bb1.iou(bb2)
    assert iou == 0


def test_bb_some_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(1, 2, 4, 15)
    iou = bb1.iou(bb2)
    assert iou == 4 / 31


def test_bb_half_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(8, 5, 10, 15)
    iou = bb1.iou(bb2)
    print(iou)
    assert iou == 1 / 3


def test_bb_all_iou():
    bb1 = BoundingBox(3, 5, 10, 15)
    bb2 = BoundingBox(3, 5, 10, 15)
    iou = bb1.iou(bb2)
    assert iou == 1
