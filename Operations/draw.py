from typing import Tuple
from collections import namedtuple

import cv2

from IMGBOX.core import Image


Rectangle = namedtuple(
    "Rectangle",
    ["ymin", "xmin", "ymax", "xmax"]
)


def draw_rectangle(
        img: Image, color: Tuple[int, int, int],
        rectangle: Rectangle, line_width: int
        ):
    cv2.rectangle(
        img._array,
        (rectangle.xmin, rectangle.ymin),
        (rectangle.xmax, rectangle.ymax),
        color, line_width
    )
