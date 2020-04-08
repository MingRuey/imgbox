from typing import Tuple

import cv2
import numpy as np

from IMGBOX.core import Image
from IMGBOX.Operations.base import BinaryOperation

__all__ = ["Overlap", "Mask"]


def _check_color(color: Tuple[int, int, int], argname: str):
    """Check input color valid"""
    color = tuple(int(val) for val in color)

    msg = "Value of {} must specify in (B, G, R), got {}"
    if len(color) != 3:
        raise ValueError(msg.format(argname, color))

    msg = "Value of {} shold lies in 0 <= val <= 255, got {}"
    if not all(0 <= val <= 255 for val in color):
        raise ValueError(msg.format(argname, color))

    return color


class Mask:
    """For create image from overlap a mask with another"""

    def __init__(self, color: Tuple[int, int, int] = (0, 0, 255)):
        self._color = _check_color(color, argname="color")

    def on(self, image: Image, mask: Image):
        color_mask = mask.to_color()
        color_mask[mask.to_gray() > 0, :] = self._color
        color_mask[mask.to_gray() <= 0, :] = [0, 0, 0]
        result = cv2.addWeighted(
            src1=image.to_color(), alpha=0.9,
            src2=color_mask, beta=0.1,
            gamma=0
        )
        return Image(result)


class Overlap(BinaryOperation):
    """For create image from overlap one image with another"""

    _color = "gray"

    def __init__(
            self,
            color1: Tuple[int, int, int] = (255, 0, 0),
            color2: Tuple[int, int, int] = (0, 0, 255)
            ):
        """
        Args:
            color1, color2 (Tuple[int, int, int]):
                The overlap color for img1 and img2.
                Overlap will first convert img1 and img2 into gray level.
                and assign the gray level to the result image with given color.
        """
        self._color1 = _check_color(color1, "color1")
        self._color2 = _check_color(color2, "color2")

    def _operate(self, array1: Image, array2: Image) -> np.ndarray:
        if not array1.shape[:2] == array2.shape[:2]:
            msg = "Height, Width of two images must be equal, get: {} and {}"
            raise ValueError(msg.format(array1.shape[:2], array2.shape[:2]))

        array1 = np.tile(array1[..., None], (1, 1, 3))
        array2 = np.tile(array2[..., None], (1, 1, 3))

        result = (array1 / 255) * self._color1 + (array2/255) * self._color2
        result = np.clip(result, a_min=0, a_max=255).astype(np.uint8)
        return result
