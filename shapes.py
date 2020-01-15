from collections import namedtuple


class Rectangle(namedtuple("Rectangle", ["ymin", "xmin", "ymax", "xmax"])):

    @property
    def area(self):
        return (self.ymax - self.ymin) * (self.xmax - self.xmin)
