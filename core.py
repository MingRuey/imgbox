import os
import pathlib
from typing import Tuple

import cv2
import numpy as np


def _safe_imread(file: str) -> np.array:
    """Read an image file and detect corropyt"""
    with open(file, "rb") as f:
        content = f.read()
        array = np.frombuffer(content, np.uint8)
        array = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if array is None:
            msg = "Decode image {} failed"
            raise ValueError(msg.format(file))
        return array


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
        # img = cv2.imread(file)
        # if img is None:
        #     msg = "Unable to load image file: {}"
        array = _safe_imread(file)
        instance._array = array
        instance._name = pathlib.Path(file).stem
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

    def __array__(self) -> np.array:
        return np.copy(self._array)

    def numpy(self) -> np.array:
        return np.copy(self._array)

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
