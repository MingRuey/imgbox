import numpy as np
import cv2

from IMGBOX.core import Image
from IMGBOX.Operations.base import SingularOperation


class Canny(SingularOperation):
    """Simple Canny operation"""

    def __init__(self, threshold1: float = 50.0, threshold2: float =200.0):
        self._thres1 = threshold1
        self._thres2 = threshold2

    def _operate(self, img: np.array) -> np.array:
        result = cv2.Canny(
            img.astype(np.uint8, copy=False),
            self._thres1, self._thres2
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
