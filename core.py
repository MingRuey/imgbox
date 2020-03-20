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
        array = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if array is None:
            msg = "Decode image {} failed"
            raise ValueError(msg.format(file))
        return array


def _math_operator(op, order=None):
    """make image suuport numpy(with dtype uint8) like math operations

    Args:
        other: the value for conducting math operations
        op: function defined in standard library 'operator'
        order:
            be either None or "r",
            indicating use __radd__, __rsub__, ... or not
    Return:
        a funcition, which return either op(self, other) or op(other, self),
        depending on the 'order' parameter
    """
    def math_op(self, other):
        if isinstance(other, Image):
            target = other._array
        else:
            target = other

        if order is None:
            result = op(self._array, target)
        elif order == "r":
            result = op(target, self._array)
        else:
            raise RuntimeError("order {} not recognized".format(order))
        return Image.from_array(result.astype(np.uint8))

    return math_op


class Image:

    def __new__(cls):
        """Create an Image object from numpy array"""
        bare = object.__new__(cls)
        return bare

    def __init__(self):
        raise RuntimeError("Do not construct directly from __init__")

    @classmethod
    def from_file(cls, file: str):
        """Construct an Image object from image file

        Note: it currently use cv2 to read image

        Args:
            file (str): the target image file
        """
        instance = cls.__new__(cls)
        array = _safe_imread(file)
        array = array.astype(np.float32, copy=False)
        instance._array = array
        instance._name = pathlib.Path(file).stem
        return instance

    @classmethod
    def from_array(
            cls, array: np.ndarray, name: str = "", to_color: bool = True
            ):
        """Construct an Image from numpy array

        Args:
            array:
                the target numpy array, dtype must be uint8.
                If to_color is False, the dimension must be (h, w, 3)
            name: the image name, default using array id.
            to_color: whether to convert image from gray to RGB
        """
        input_array_info = "array of shape {} and dtype {}"
        input_array_info = input_array_info.format(array.shape, array.dtype)
        if array.dtype != np.uint8:
            msg = "dtype of array must be uint8, get {}"
            raise ValueError(msg.format(input_array_info))

        if array.ndim != 3 and to_color:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

        elif array.ndim != 3 or array.shape[-1] != 3:
            msg = "shape of array must be (h, w, 3), get {}"
            raise ValueError(msg.format(input_array_info))

        instance = cls.__new__(cls)
        instance._array = array.astype(np.float32)
        instance._name = name if name else "array_" + str(id(array))
        return instance

    @classmethod
    def copy(cls, image):
        """Create a copy of another image"""
        instance = cls.__new__(cls)
        instance._array = np.array(image)
        instance._name = image.name
        return instance

    @property
    def name(self):
        return self._name

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
        cv2.imwrite(out_file, self._array)

    def __array__(self) -> np.ndarray:
        return self._array

    def numpy(self, dtype=np.uint8) -> np.ndarray:
        return self._array.astype(dtype)

    @property
    def h(self) -> int:
        return self._array.shape[0]

    @property
    def w(self) -> int:
        return self._array.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image tuple of (height, width, channel)"""
        return self._array.shape

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

        self._array = cv2.resize(
            self._array, shape,
            interpolation=getattr(cv2, interpolation)
        )

    __add__ = _math_operator(op=operator.add, order=None)
    __radd__ = _math_operator(op=operator.add, order="r")
    __sub__ = _math_operator(op=operator.sub, order=None)
    __rsub__ = _math_operator(op=operator.sub, order="r")
    __mul__ = _math_operator(op=operator.mul, order=None)
    __rmul__ = _math_operator(op=operator.mul, order="r")
    __truediv__ = _math_operator(op=operator.truediv, order=None)
    __rtruediv__ = _math_operator(op=operator.truediv, order="r")
    __floordiv__ = _math_operator(op=operator.floordiv, order=None)
    __rfloordiv__ = _math_operator(op=operator.floordiv, order="r")

    __iadd__ = _math_operator(op=operator.iadd, order=None)
    __isub__ = _math_operator(op=operator.isub, order=None)
    __imul__ = _math_operator(op=operator.imul, order=None)
    __itruediv__ = _math_operator(op=operator.itruediv, order=None)
    __ifloordiv__ = _math_operator(op=operator.ifloordiv, order=None)
