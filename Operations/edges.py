from typing import List

import cv2
import numpy as np
from skimage.segmentation import active_contour

from IMGBOX.core import Image
from IMGBOX.Operations.base import SingularOperation


__all__ = ["ActiveContour", "Canny", "Laplacian"]


class ActiveContour(SingularOperation):

    def __init__(self, snakes: List[np.array], **kwargs):
        """Initialize active contour (snake) by specifying initial snakes

        The **kwargs follows parameters of scikit-image active_contour
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

        Args:
            snakes (List[np.array]): initial snake coordinates.
            **kwargs: checkout out scikit-image active_contour
        """
        if not snakes:
            msg = "Empty snakes."
            raise ValueError(msg)
        if isinstance(snakes, np.ndarray):
            snakes = [snakes]
        if len(snakes) > 1:
            msg = "Currently only support single snakes"
            raise ValueError(msg)

        self._snakes = [array.copy() for array in snakes]
        self._kwargs = kwargs

    def _operate(self, img: np.array) -> np.array:
        return active_contour(img, self._snakes[0], **self._kwargs)


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
