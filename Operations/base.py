from abc import ABC, abstractmethod

import numpy as np

from IMGBOX.core import Image


class SingularOperation(ABC):
    """Operation for single image"""

    @abstractmethod
    def _operate(self, img1: np.array) -> np.array:
        """Acutal operation on numpy array that subclass must implement"""
        pass

    def on(self, img: Image) -> Image:
        """Operate on single image"""
        result_array = self._operate(np.array(img))
        return Image.from_array(result_array.astype(np.uint8, copy=False))


class BinaryOperation(ABC):
    """Operation for two images"""

    @abstractmethod
    def _operate(self, img1: np.array, img2: np.array) -> np.array:
        """Acutal operation on numpy array that subclass must implement"""
        pass

    def on(self, img1: Image, img2: Image) -> Image:
        """Operate on two images"""
        result_array = self._operate(np.array(img1), np.array(img2))
        return Image.from_array(result_array.astype(np.uint8, copy=False))
