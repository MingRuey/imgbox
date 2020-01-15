import numpy as np

from IMGBOX.core import Image
from IMGBOX.Operations.base import BinaryOperation


class AbsDiff(BinaryOperation):
    """Pixel wise difference on image"""

    def _operate(self, array1: np.array, array2: np.array) -> np.array:
        return np.abs(array1 - array2)
