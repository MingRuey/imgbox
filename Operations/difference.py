import numpy as np

from IMGBOX.core import Image
from IMGBOX.Operations.base import BinaryOperation

__all__ = ["AbsDiff"]


class AbsDiff(BinaryOperation):
    """Pixel wise difference on image"""

    _cvt_to_f32 = True

    def _operate(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        return np.abs(array1 - array2).astype(np.uint8)
