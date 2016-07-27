import warnings
import numpy as np


class BoundingBox:
    r"""
    Bounding box class that is used for both hypothesis and ground truth
    objects.

    Parameters
    ----------
    x : `float`
        The image horizontal coordinate of the top left vertex of the
        bounding box.
    y : `float`
        The image vertical coordinate of the top left vertex of the
        bounding box.
    width : `float`
        The horizontal size of the bounding box
    height : `float`
        The vertical size of the bounding box.
    id : `str` or `int`, optional
        The id of the object that the bounding box represents.
    """
    def __init__(self, x, y, width, height, id=''):
        if width < 0 or height < 0:
            raise ValueError('Width and height must be bigger or equal than 0')
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._id = str(id)

    @classmethod
    def init_from_dic(cls, bb_dic):
        r"""
        Create a bounding box from a dictionary.

        Parameters
        ----------
        bb_dic : `dict`
            A dictionary representing a bounding box. Must have the following
            format:
                ``{'x': `float`,
                   'y': `float`,
                   'width': `float`,
                   'height': `float`,
                   'id': `str` or `int`}``
        Returns
        -------
        bb : :map:`BoundingBox`
            A new bounding box.
        """
        if 'id' in bb_dic:
            return cls(bb_dic['x'], bb_dic['y'],
                       bb_dic['width'], bb_dic['height'],
                       id=bb_dic['id'])
        else:
            return cls(bb_dic['x'], bb_dic['y'],
                       bb_dic['width'], bb_dic['height'])

    @property
    def x(self):
        r"""
        The image horizontal coordinate of the top left vertex of the
        bounding box.

        :type: `float`
        """
        return self._x

    @property
    def y(self):
        r"""
        The image vertical coordinate of the top left vertex of the
        bounding box.

        :type: `float`
        """
        return self._y

    @property
    def width(self):
        r"""
        The horizontal size of the bounding box

        :type: `float`
        """
        return self._width

    @property
    def height(self):
        r"""
        The vertical size of the bounding box

        :type: `float`
        """
        return self._height

    @property
    def id(self):
        r"""
        The id of the object that the bounding box represents.

        :type: `str`
        """
        return self._id

    @property
    def top_left(self):
        return np.asarray([self._x, self._y])

    @property
    def bottom_right(self):
        return np.asarray([self._x + self._width, self._y + self._height])

    @property
    def center(self):
        return np.asarray([self._x + self._width / 2,
                           self._y + self._height / 2])

    def area(self):
        r"""
        Compute and return the area of the bounding box.

        Returns
        -------
        area : `float`
            Area of the bounding box.
        """
        return self._width * self._height
    
    def intersect(self, bb):
        r"""
        Return a bounding box that is the intersection between this bounding
        box and another bounding box.

        Parameters
        ----------
        bb : :map:`BoundingBox`
            Another bounding box object.

        Returns
        -------
        intersection : :map:`BoundingBox`
            A new bounding box representing the intersection between the
            bounding boxes.
        """
        x = max(self._x, bb.x)
        y = max(self._y, bb.y)
        width = max(0, min(self._x + self._width, bb.x + bb.width) - x)
        height = max(0, min(self._y + self._height, bb.y + bb.height) - y)

        if width == 0 or height ==0:
            warnings.warn('Bounding boxes do not intersect')

        return BoundingBox(x, y, width, height, id='intersect')

    def intersection(self, bb):
        r"""
        Calculate the area of the intersection between this bounding box and
        another bounding box.

        Parameters
        ----------
        bb : :map:`BoundingBox`
            Another bounding box object.

        Returns
        -------
        intersection : `float`
            Area of the intersection between the bounding boxes.
        """
        return self.intersect(bb).area()

    def union(self, bb):
        r"""
        Calculate the area of the union between this bounding box and another
        bounding box.

        Parameters
        ----------
        bb : :map:`BoundingBox`
            Another bounding box object.

        Returns
        -------
        union : `float`
            Area of the union between the bounding boxes.
        """
        return self.area() + bb.area() - self.intersect(bb).area()

    def intersection_over_union(self, bb):
        r"""
        Calculate the Intersection over Union (IoU) between this bounding
        box and another bounding box.

        Parameters
        ----------
        bb : :map:`BoundingBox`
            Another bounding box object.

        Returns
        -------
        iou : `float`
            IoU between the bounding boxes.
        """
        return intersection_over_union(self, bb)

    def center_distance(self, bb):
        r"""
        Calculate the Euclidean distance between the center of this bounding
        box and the center of another bounding box.

        Parameters
        ----------
        bb : :map:`BoundingBox`
            Another bounding box object.

        Returns
        -------
        center_distance : `float`
            Euclidean distance between the centers of the bounding boxes.
        """
        return center_distance(self, bb)

    def __str__(self):
        return '(id, x, y, w, h) = (%s, %.1f, %.1f, %.1f, %.1f)' % (
            self._id, self._x, self._y, self._width, self._height)


def intersection_over_union(bb1, bb2):
    r"""
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    bb1 : :map:`BoundingBox`
        The first bounding box.
    bb2 : :map:`BoundingBox`
        The second bounding box.

    Returns
    -------
    iou : `float`
        IoU between bounding boxes.
    """
    intersection = bb1.intersection(bb2)
    union = bb1.area() + bb2.area() - intersection
    return intersection / union


def center_distance(bb1, bb2):
    r"""
    Calculate the Euclidean distance between the center of two bounding boxes.

    Parameters
    ----------
    bb1 : :map:`BoundingBox`
        The first bounding box.
    bb2 : :map:`BoundingBox`
        The second bounding box.

    Returns
    -------
    center_distance : `float`
        Euclidean distance between the centers of the bounding boxes.
    """
    return np.sqrt((bb1.center - bb2.center) ** 2)