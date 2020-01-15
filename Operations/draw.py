from typing import Tuple

import cv2

from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle


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
