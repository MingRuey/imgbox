from abc import ABC, abstractmethod

import numpy as np

from IMGBOX.core import Image


class SingularOperation(ABC):
    """Operation for single image"""

    _cvt_to_f32 = False
    _color = "unchanged"  # options: "unchanged", "color", "gray"

    @abstractmethod
    def _operate(self, img1: np.ndarray) -> np.ndarray:
        """Acutal operation on numpy array that subclass must implement

        For operations accepts only float32:
            overwrite and set _cvt_to_f32 to be True
        For operations accepts only gray/color image:
            overwrite and set _color to "gray"/"color",
            otherwise it will remains unchanged as user inputs
        """
        pass

    def on(self, img: Image) -> Image:
        """Operate on single image"""
        if self._color == "color":
            img = img.to_color()
        elif self._color == "gray":
            img = img.to_gray()
        elif self._color != "unchanged":
            msg = "Unrecognized color option: {}".format(self._color)
            raise ValueError(msg)

        if self._cvt_to_f32:
            img = img.astype(np.float32, copy=False)
        result_array = self._operate(img)
        name = "{} on ".format(self.__class__.__name__) + img.name
        return Image(result_array, name=name)


class BinaryOperation(ABC):
    """Operation for two images"""

    _cvt_to_f32 = False
    _color = "unchanged"  # options: "unchanged", "color", "gray"

    @abstractmethod
    def _operate(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Acutal operation on numpy array that subclass must implement

        For operations accepts only float32:
            overwrite and set _cvt_to_f32 to be True
        For operations accepts only gray/color image:
            overwrite and set _color to "gray"/"color",
            otherwise it will remains unchanged as user inputs
        """
        pass

    def on(self, img1: Image, img2: Image) -> Image:
        """Operate on two images"""
        if self._color == "color":
            img1 = img1.to_color()
            img2 = img2.to_color()
        elif self._color == "gray":
            img1 = img1.to_gray()
            img2 = img2.to_gray()
        elif self._color != "unchanged":
            msg = "Unrecognized color option: {}".format(self._color)
            raise ValueError(msg)

        if self._cvt_to_f32:
            img1 = img1.astype(np.float32, copy=False)
            img2 = img2.astype(np.float32, copy=False)

        result_array = self._operate(img1, img2)
        name = "{} on ({}, {})".format(
            self.__class__.__name__, img1.name, img2.name
        )
        return Image(result_array, name=name)
