from numbers import Number
from collections import namedtuple

__all__ = ["Rectangle", "Point"]


def _cvt2float(number) -> float:
    if not isinstance(number, Number):
        return float(number)
    else:
        return number


class Point(namedtuple("Point", ["y", "x"])):

    def __new__(cls, y: int, x: int):
        yloc = int(y)
        xloc = int(x)
        return super().__new__(cls, y=yloc, x=xloc)

    def inside(self, image) -> bool:
        """Check Point locate inside given image or not"""
        h, w, _ = image.shape
        return 0 <= self.y < h and 0 <= self.x < w

    def __add__(self, other):
        return Point(y=self.y + other.y, x=self.x + other.x)

    def __sub__(self, other):
        return Point(y=self.y - other.y, x=self.x - other.x)

    def __neg__(self):
        return Point(y=-self.y, x=-self.x)


class Rectangle(namedtuple("Rectangle", ["ymin", "xmin", "ymax", "xmax"])):

    def __new__(cls, ymin, xmin, ymax, xmax):
        ymin = _cvt2float(ymin)
        xmin = _cvt2float(xmin)
        ymax = _cvt2float(ymax)
        xmax = _cvt2float(xmax)

        if not ymin < ymax:
            raise ValueError("ymin must less than ymax")
        elif not xmin < xmax:
            raise ValueError("xmin must less than xmax")

        return super().__new__(cls, ymin, xmin, ymax, xmax)

    @property
    def area(self):
        """the area of the rectangle"""
        return (self.ymax - self.ymin) * (self.xmax - self.xmin)

    @staticmethod
    def overlap(rect1, rect2) -> bool:
        """if two rectangles overlap"""
        if rect1.ymax <= rect2.ymin or rect2.ymax <= rect1.ymin:
            return False
        if rect1.xmax <= rect2.xmin or rect2.xmax <= rect1.xmin:
            return False
        return True

    def overlap_with(self, other) -> bool:
        """if the rectange is overlapped with another rectangle"""
        return self.overlap(self, other)
