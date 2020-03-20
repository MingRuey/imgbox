from abc import ABC, abstractmethod

import numpy as np

from IMGBOX.core import Image


class SingularOperation(ABC):
    """Operation for single image"""

    @abstractmethod
    def _operate(self, img1: np.ndarray) -> np.ndarray:
        """Acutal operation on numpy array that subclass must implement"""
        pass

    def on(self, img: Image) -> Image:
        """Operate on single image"""
        result_array = self._operate(np.array(img))
        name = "{} on ".format(self.__class__.__name__) + img.name
        return Image.from_array(
            result_array.astype(np.uint8, copy=False),
            name=name
        )


class BinaryOperation(ABC):
    """Operation for two images"""

    @abstractmethod
    def _operate(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Acutal operation on numpy array that subclass must implement"""
        pass

    def on(self, img1: Image, img2: Image) -> Image:
        """Operate on two images"""
        result_array = self._operate(np.array(img1), np.array(img2))
        name = "{} on ({}, {})".format(
            self.__class__.__name__, img1.name, img2.name
        )
        return Image.from_array(
            result_array.astype(np.uint8, copy=False), name=name
        )
