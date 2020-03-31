from typing import Tuple, List

import cv2
import numpy as np

from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle, Point

__all__ = ["draw_rectangle", "draw_points"]


def draw_rectangle(
        img: Image, rectangle: Rectangle,
        color: Tuple[int, int, int], line_width: int
        ):
    """Draw rectangle onto image by given color and line_width"""
    cv2.rectangle(
        img,
        (rectangle.xmin, rectangle.ymin),
        (rectangle.xmax, rectangle.ymax),
        color, line_width
    )


def draw_points(img, points: List[Point], color: Tuple[int, int, int]):
    """Draw points onto image by given color

    Args:
        img: the Image object to draw on
        points: points to draw on images, can be either
            a) list of Point object
            b) a numpy array of dim (N, 2)
        color: the drawing color of image
    """
    if isinstance(points, np.ndarray):
        if not (points.ndim == 2 and points.shape[-1] == 2):
            msg = "Invalid dimension for points, must be (N, 2), got {}"
            raise ValueError(msg.format(points.shape))
        indices = points.astype(np.int)
    else:
        if any(not pt.inside(img) for pt in points):
            msg = "Points {} not all inside image with shape {}"
            raise ValueError(msg.format(points, img))
        # stack points into indices
        indices = np.array([(pt.y, pt.x) for pt in points], dtype=np.int)

    img[indices[:, 0], indices[:, 1], ...] = color
