import pytest
import numpy as np

from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle, Point


class TestPoint:

    def test_construct(self):
        """Point.x Point.y should return right value"""
        pt = Point(x=5, y=8)
        assert pt.y == 8
        assert pt.x == 5

        pat = Point(8, 5)
        assert pt.y == 8
        assert pt.x == 5

    def test_inside_image(self):
        """Point.inside should return right result"""
        sample_img = Image(np.zeros((100, 200, 3), dtype=np.uint8))

        # test 4 corners
        top_left = Point(y=0, x=0)
        bottom_left = Point(y=99, x=0)
        top_right = Point(y=0, x=199)
        bottom_right = Point(y=99, x=199)

        assert top_left.inside(sample_img)
        assert bottom_left.inside(sample_img)
        assert top_right.inside(sample_img)
        assert bottom_right.inside(sample_img)

        # test out-side point
        pt1 = Point(y=-1, x=50)
        pt2 = Point(y=100, x=50)
        pt3 = Point(y=50, x=-1)
        pt4 = Point(y=50, x=200)

        assert not pt1.inside(sample_img)
        assert not pt2.inside(sample_img)
        assert not pt3.inside(sample_img)
        assert not pt4.inside(sample_img)

    def test_add_substract_points(self):
        """+ or - on two points return a point whose location are summed"""
        pt1 = Point(x=1, y=2)
        pt2 = Point(x=3, y=40)
        summed = pt1 + pt2
        substracted = pt1 - pt2

        assert summed.x == 4
        assert summed.y == 42
        assert substracted.x == -2
        assert substracted.y == -38

    def test_unary(self):
        pt = Point(x=-5, y=200)
        pt = -pt
        assert pt.x == 5
        assert pt.y == -200


class TestRectangle:

    def test_construct(self):
        """Create Rectangle should works properly"""
        rect = Rectangle(1, 2, 3, 4)
        assert rect.ymin == 1
        assert rect.xmin == 2
        assert rect.ymax == 3
        assert rect.xmax == 4

        rect = Rectangle(-100, -50, -60, -25)
        assert rect.ymin == -100
        assert rect.xmin == -50
        assert rect.ymax == -60
        assert rect.xmax == -25

        rect = Rectangle(50.5, 70.2, 80.2, 100.3)
        assert rect.ymin == 50.5
        assert rect.xmin == 70.2
        assert rect.ymax == 80.2
        assert rect.xmax == 100.3

        # should able to auto convert string
        rect = Rectangle("-123", "-456.7", "111", "222.3")
        assert rect.ymin == -123
        assert rect.xmin == -456.7
        assert rect.ymax == 111
        assert rect.xmax == 222.3

    def test_construct_with_invalid(self):
        """Create Rectanle with invalid argumants should raise ValueError"""
        with pytest.raises(ValueError):
            rect = Rectangle(300, 100, 500, 50)  # xmin > xmax

        with pytest.raises(ValueError):
            rect = Rectangle(300, 100, 200, 500)  # ymin > ymax

        with pytest.raises(ValueError):
            rect = Rectangle(300, 100, 300, 500)  # ymin == ymax

        with pytest.raises(ValueError):
            rect = Rectangle(300, 500, 400, 500)  # ymin == ymax

        # unconvertalbe string
        with pytest.raises(ValueError):
            rect = Rectangle("GG", 500, 400, 500)

    def test_area(self):
        """.area should return right values"""
        rect = Rectangle(30, 50, 130, 60)
        assert rect.area == 100 * 10

        rect = Rectangle(10.5, 20.7, 11.2, 50.6)
        assert abs(rect.area - 20.93) < 1e-10

        rect = Rectangle(-10, -20, 10, 60)
        assert rect.area == 20 * 80

    def test_overlap(self):
        """Rectangle.overlap & .overlap_with should return correct results"""
        rect1 = Rectangle(10, 20, 30, 40)
        rect2 = Rectangle(50, 60, 70, 80)

        # overlap should be commutative
        assert not rect1.overlap_with(rect2)
        assert not rect2.overlap_with(rect1)
        assert not Rectangle.overlap(rect1, rect2)
        assert not Rectangle.overlap(rect2, rect1)

        rect1 = Rectangle(-10, -20, 10, 60)
        rect2 = Rectangle(0, 50, 100, 200)
        assert rect1.overlap_with(rect2)
        assert rect2.overlap_with(rect1)
        assert Rectangle.overlap(rect1, rect2)
        assert Rectangle.overlap(rect2, rect1)

        # rectangles with only same boarder are not considered overlapped
        rect1 = Rectangle(-30, -10, -20, 0)
        rect2 = Rectangle(-20, -5, 30, 20)
        rect3 = Rectangle(-40, 0, 30, 20)
        assert not rect1.overlap_with(rect2)
        assert not rect1.overlap_with(rect3)
        assert not Rectangle.overlap(rect2, rect1)
        assert not Rectangle.overlap(rect3, rect1)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
