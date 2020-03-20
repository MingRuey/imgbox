import numpy as np
from scipy.signal import fftconvolve as fftconv

from IMGBOX.core import Image
from IMGBOX.Operations.base import BinaryOperation


__all__ = ["CrossCorrelate2D"]


class CrossCorrelate2D(BinaryOperation):
    """Calculate cross correlation between two images

    It will first convert image to gray-scale, then conduct fftconvolve
    """

    def _operate(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        img1 = np.sum(array1, axis=2)
        img1 -= np.mean(img1)

        img2 = np.sum(array2, axis=2)
        img2 -= np.mean(img2)

        result = fftconv(img1, img2[::-1, ::-1], mode="same")
        result = (result - np.min(result))
        result = (result / np.max(result)) * 255
        return result
