import numpy as np

from IMGBOX.core import Image
from IMGBOX.Operations.base import BinaryOperation


class AbsDiff(BinaryOperation):

    def _operate(self, array1: np.array, array2: np.array) -> np.array:
        """Pixel wise difference on image"""
        return np.abs(array1 - array2)
