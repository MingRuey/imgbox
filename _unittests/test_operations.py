import pathlib

import cv2
import numpy as np
import pytest

from IMGBOX.core import Image
from IMGBOX.Operations.difference import AbsDiff
from IMGBOX.Operations.correlation import CrossCorrelate2D
from IMGBOX.Operations.edges import Canny, Laplacian

from IMGBOX._unittests.configs import SAMPLE_IMAGES, IMAGE_BW
from IMGBOX.Visualization.plot import display


class TestAbsDiff:

    @pytest.mark.parametrize(
        "array1,array2", [
            (
                np.random.randint(0, 255, size=(2, 2, 3), dtype=np.uint8),
                np.random.randint(0, 255, size=(2, 2, 3), dtype=np.uint8)
            ),
            (
                np.random.randint(0, 255, size=(3, 3), dtype=np.uint8),
                np.random.randint(0, 255, size=(3, 3), dtype=np.uint8)
            )
        ],
        ids=["color", "gray"]
    )
    def test_simple_cases(self, array1, array2):
        """Sample cases for validate AbsDiff"""
        img1 = Image(array1)
        img2 = Image(array2)
        zero = Image(np.zeros(img2.shape, dtype=np.uint8))
        operation = AbsDiff()

        result = operation.on(img1, zero)
        assert np.all(result == array1)
        assert np.all(img1 == array1)
        assert np.all(zero == 0)

        result = operation.on(zero, img1)
        assert np.all(result == array1)
        assert np.all(img1 == array1)
        assert np.all(zero == 0)

        result1 = operation.on(img2, img1)
        result2 = operation.on(img1, img2)
        assert np.all(result1 == result2)


class TestCrossCorrelation:

    def test_case(self):
        array1 = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        array2 = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        img1 = Image(array1)
        img2 = Image(array2)
        op = CrossCorrelate2D()
        op.on(img1, img2)


@pytest.mark.parametrize(
    "file", [SAMPLE_IMAGES[0], IMAGE_BW], ids=["color", "gray"]
)
class TestEdges:

    def test_canny(self, file):
        """Canny should work on both color and gray image"""
        img = Image.from_file(file)
        op = Canny()
        canny = op.on(img)
        assert not canny.is_color

    def test_laplacian(self, file):
        """Laplacian should work on both color and gray image"""
        img = Image.from_file(file)
        op = Laplacian()
        laplace = op.on(img)
        assert laplace.is_color == img.is_color


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
