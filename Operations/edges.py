import numpy as np
import cv2

from IMGBOX.core import Image
from IMGBOX.Operations.base import SingularOperation


class Canny(SingularOperation):
    """Simple Canny operation"""

    def __init__(self, threshold1: float = 50.0, threshold2: float = 200.0):
        """
        Args:
            threshold1 (float): Low threshold for Canny. Defaults to 50.0.
            threshold2 (float): High threshold for Canny. Defaults to 200.0.
        """
        self._thres1 = threshold1
        self._thres2 = threshold2

    def _operate(self, img: np.array) -> np.array:
        result = cv2.Canny(
            img.astype(np.uint8, copy=False),
            self._thres1, self._thres2
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result


class Laplacian(SingularOperation):
    """Simple Laplacian edge detector"""

    def __init__(self, kernel_size: int = 3):
        """
        Args:
            kernel_size (int): The size of Laplace kernel. Defaults to 3.
        """
        self._kern = kernel_size

    def _operate(self, img: np.array) -> np.array:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Laplacian(
            img.astype(np.uint8, copy=False),
            ddepth=cv2.CV_8U, ksize=self._kern
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
