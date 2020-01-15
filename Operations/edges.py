import numpy as np
import cv2

from IMGBOX.core import Image
from IMGBOX.Operations.base import SingularOperation


class Canny(SingularOperation):
    """Simple Canny operation"""

    def _operate(self, img: np.array) -> np.array:
        result = cv2.Canny(img.astype(np.uint8, copy=False), 50, 200)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
