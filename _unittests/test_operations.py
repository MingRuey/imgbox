import pathlib

import cv2
import numpy as np
import pytest

from IMGBOX.core import Image
from IMGBOX.Operations.difference import AbsDiff


class TestAbsDiff:

    def test_case1(self):
        """Sample case for validate AbsDiff"""
        array1 = np.random.randint(0, 255, size=(2, 2, 3), dtype=np.uint8)
        array2 = np.random.randint(0, 255, size=(2, 2, 3), dtype=np.uint8)

        img1 = Image.from_array(array1)
        img2 = Image.from_array(array2)
        zero = Image.from_array(np.zeros((2, 2, 3), dtype=np.uint8))
        operation = AbsDiff()

        result = operation.on(img1, zero)
        assert np.all(result.numpy() == array1)
        assert np.all(img1.numpy() == array1)
        assert np.all(zero.numpy() == 0)
        result = operation.on(zero, img1)
        assert np.all(result.numpy() == array1)
        assert np.all(img1.numpy() == array1)
        assert np.all(zero.numpy() == 0)

        result1 = operation.on(img2, img1)
        result2 = operation.on(img1, img2)
        assert np.all(result1.numpy() == result2.numpy())


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
