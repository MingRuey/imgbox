import os
import pathlib
import operator
from functools import partial
from typing import Tuple
from numbers import Number

import cv2
import numpy as np

from IMGBOX.shapes import Rectangle

__all__ = ["Image"]


def _safe_imread(file: str) -> np.ndarray:
    """Read an image file and detect corropyt"""
    with open(file, "rb") as f:
        content = f.read()
        array = np.frombuffer(content, np.uint8)
        array = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if array is None:
            msg = "Decode image {} failed"
            raise ValueError(msg.format(file))
        return array


class Image(np.ndarray):

    def __new__(
            cls, array: np.ndarray, name: str = "",
            to_color: bool = False, dtype=np.uint8
            ):
        """
        checkout numpy tutorial:
        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

        Args:
            array (np.ndarray): the internal array represents of Image
            name (str): the name of the Image
            to_color (bool): if auto cast the image into BGR
            dtype: data type for the image array
        """
        input_array_info = "array of shape {} and dtype {}"
        input_array_info = input_array_info.format(array.shape, array.dtype)
        if array.dtype != np.uint8:
            msg = "dtype of array must be uint8, get {}"
            raise ValueError(msg.format(input_array_info))

        if array.ndim == 2 and to_color:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[-1] != 3:
            msg = "Color image array must be (h, w, 3), get {}"
            raise ValueError(msg.format(input_array_info))
        elif array.ndim != 3 and array.ndim != 2:
            msg = "Array must be (h, w, 3) for color; (h, w) for gray, got {}"
            raise ValueError(msg.format(input_array_info))

        instance = np.array(array, dtype=dtype, copy=True).view(cls)
        instance.name = name if name else "array_" + str(id(array))
        return instance

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)

    @classmethod
    def from_file(cls, file: str):
        """Construct an Image object from image file

        Note: it currently use cv2 to read image

        Args:
            file (str): the target image file
        """
        array = _safe_imread(file)
        return cls.__new__(
            cls, array=array, name=pathlib.Path(file).stem
        )

    def save(self, out_file: str, overwrite: bool = True):
        """Output image to file

        Args:
            out_file (str): the target output file
            overwrite (bool): if out_file exists, over-write it or not.
        """
        if not overwrite:
            target = pathlib.Path(out_file)
            if pathlib.Path(out_file).exists():
                msg = "Can not write image to {}, it alreay exists."
                raise ValueError(msg.format(target))
        cv2.imwrite(out_file, self)

    @property
    def h(self) -> int:
        return self.shape[0]

    @property
    def w(self) -> int:
        return self.shape[1]

    @property
    def c(self) -> int:
        return 0 if self.ndim == 2 else self.shape[-1]

    @property
    def is_color(self) -> bool:
        return self.c == 3

    def to_color(self):
        if self.is_color:
            return self
        else:
            return Image(
                cv2.cvtColor(self, cv2.COLOR_GRAY2BGR),
                name=self.name
            )

    def to_gray(self):
        if self.is_color:
            return Image(
                cv2.cvtColor(self, cv2.COLOR_BGR2GRAY),
                name=self.name
            )
        else:
            return self

    def resize(
            self, shape: Tuple[int, int],
            interpolation: str = "INTER_AREA"
            ):
        """Resize image into given shape

        Args:
            shape: tuple of (h, w)
            interpolation:
                one of - "INTER_AREA": default,
                "INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC",
                "INTER_LANCZOS4"
        Return:
            a new Image object with shape resized
        """
        if not hasattr(cv2, interpolation):
            msg = "Not supported interpolation method: {}"
            raise ValueError(msg.format(interpolation))

        if len(shape) != 2 or \
                any(not isinstance(dim, int) for dim in shape) or \
                any(dim <= 0 for dim in shape):
            msg = "Invalid target shape: {}"
            raise ValueError(msg.format(shape))

        resize = cv2.resize(
            self, shape[::-1],
            interpolation=getattr(cv2, interpolation)
        )
        return Image(resize, name=self.name)
