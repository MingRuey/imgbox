from typing import List

import cv2
import numpy as np
from skimage.segmentation import active_contour, chan_vese
from skimage.segmentation import morphological_chan_vese as morph_chan_vese
from skimage.segmentation import morphological_geodesic_active_contour as morph_gac

from IMGBOX.core import Image
from IMGBOX.Operations.base import SingularOperation


__all__ = [
    "Canny", "Laplacian",
    "ActiveContour", "ChanVese",
    "MorphChanVese", "MorphGAC"
]


class _FromSK:
    """base class of operations simply used from skicit-image built-ins"""

    def __init__(self, **kwargs):
        """
        The **kwargs follows parameters of scikit-image
        """
        self._kwargs = kwargs

    @property
    def op_func(self):
        raise NotImplementedError()

    def on(self, img: Image) -> np.ndarray:
        gray = img.to_gray()
        return self.op_func(gray, **self._kwargs)


MorphGAC = type("MorphGAC", (_FromSK,), {"op_func": staticmethod(morph_gac)})
MorphChanVese = type("MorphChanVese", (_FromSK,), {"op_func": staticmethod(morph_chan_vese)})
ChanVese = type("ChanVese", (_FromSK,), {"op_func": staticmethod(chan_vese)})
ActiveContour = type("ActiveContour", (_FromSK,), {"op_func": staticmethod(active_contour)})


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

    def _operate(self, img: np.ndarray) -> np.ndarray:
        result = cv2.Canny(
            img.astype(np.uint8, copy=False),
            self._thres1, self._thres2
        )
        return result


class Laplacian(SingularOperation):
    """Simple Laplacian edge detector"""

    def __init__(self, kernel_size: int = 3):
        """
        Args:
            kernel_size (int): The size of Laplace kernel. Defaults to 3.
        """
        self._kern = kernel_size

    def _operate(self, img: np.ndarray) -> np.ndarray:
        result = cv2.Laplacian(
            img.astype(np.uint8, copy=False),
            ddepth=cv2.CV_8U, ksize=self._kern
        )
        return result
