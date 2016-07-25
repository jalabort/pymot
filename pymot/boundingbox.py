import warnings


class BoundingBox:
    r"""
    Bounding box class that is used for both hypothesis and ground truth
    objects.

    :param x : `float`
        The image horizontal coordinate of the top left vertex of the
        bounding box.
    :param y : `float`
        The image vertical coordinate of the top left vertex of the
        bounding box.
    :param width : `float`
        The horizontal size of the bounding box
    :param height : `float`
        The vertical size of the bounding box.
    :param id : `str` or `int`, optional
        The id of the object that the bounding box represents.
    """
    def __init__(self, x, y, width, height, id=''):
        if width < 0 or height < 0:
            raise ValueError('Width and height must be bigger or equal than 0')
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.id = str(id)

    @classmethod
    def init_from_dic(cls, bb_dic):
        r"""
        Create a bounding box from a dictionary.

       :param bb_dic : `dict`
            A dictionary representing a bounding box. Must have the following
            format:
            ``{'x': `float`,
            'y': `float`,
            'width': `float`,
            'height': `float`,
            'id': `str` or `int`}``

       :return: bb : :map:`BoundingBox`
            A new bounding box.
        """
        if 'id' in bb_dic:
            return cls(bb_dic['x'], bb_dic['y'],
                       bb_dic['width'], bb_dic['height'],
                       id=bb_dic['id'])
        else:
            return cls(bb_dic['x'], bb_dic['y'],
                       bb_dic['width'], bb_dic['height'])

    def area(self):
        r"""
        Compute and return the area of the bounding box.

       :return: area : `float`
            Area of the bounding box.
        """
        return self.width * self.height
    
    def intersect(self, bb):
        r"""
        Compute and return a bounding box object that is the intersection
        between self and another bounding box.

        :param bb : `:map:`BoundingBox``
            Another bounding box object.

        :return: intersection : `:map:`BoundingBox``
            A new bounding box object representing the intersection between
            self and bb.
        """
        x = max(self.x, bb.x)
        y = max(self.y, bb.y)
        width = max(0, min(self.x + self.width, bb.x + bb.width) - x)
        height = max(0, min(self.y + self.height, bb.y + bb.height) - y)

        if width == 0 or height ==0:
            warnings.warn('Bounding boxes do not intersect')

        return BoundingBox(x, y, width, height, id='intersect')

    def iou(self, bb):
        r"""
        Compute and return the intersection over union (IoU) between self and
        another bounding box.

        :param bb : `:map:`BoundingBox``
            Another bounding box object.

        :return: iou : `float`
            IoU between self and bb.
        """
        intersection = self.intersect(bb).area()
        union = self.area() + bb.area() - intersection
        return intersection / union

    def __str__(self):
        return '(id, x, y, w, h) = (%s, %.1f, %.1f, %.1f, %.1f)' % (
            self.id, self.x, self.y, self.width, self.height)
