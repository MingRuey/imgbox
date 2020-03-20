from typing import Tuple, List

import cv2
import numpy as np

from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle, Point


def draw_rectangle(
        img: Image, rectangle: Rectangle,
        color: Tuple[int, int, int], line_width: int
        ):
    """Draw rectangle onto image by given color and line_width"""
    cv2.rectangle(
        img._array,
        (rectangle.xmin, rectangle.ymin),
        (rectangle.xmax, rectangle.ymax),
        color, line_width
    )


def draw_points(img, points: List[Point], color: Tuple[int, int, int]):
    """Draw points onto image by given color
    """
    if isinstance(points, np.ndarray):
        points = [points]

    if any(not pt.inside(img) for pt in points):
        msg = "Points {} not all inside image with shape {}"
        raise ValueError(msg.format(points, img))

    # stack points into indices
    indices = np.array([(pt.y, pt.x) for pt in points], dtype=np.int)
    img._array[indices[:, 0], indices[:, 1], ...] = color
