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
        return instance

    def __array__(self) -> np.array:
        return np.copy(self._array)

    def numpy(self) -> np.array:
        return np.copy(self._array)
