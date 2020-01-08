import numpy as np
import pytest

from IMGBOX.core import Image
from IMGBOX._unittests.configs import INVALID_IMAGES, SAMPLE_IMAGES
from IMGBOX._unittests.configs import UNDETECTED_IMAGES


class TestImageObject:

    def test_can_not_init(self):
        """Init Image object is prohibited"""
        with pytest.raises(RuntimeError):
            Image()

    def test_from_file(self):
        """Image object can and should be loaded from file"""
        for file in SAMPLE_IMAGES:
            img = Image.from_file(str(file))
            assert isinstance(img.numpy(), np.ndarray)

    @pytest.mark.parametrize(
        "sample", [str(file) for file in INVALID_IMAGES]
    )
    def test_from_invalid_file(self, sample):
        """Create Image object from invalid files should raise ValueError"""
        with pytest.raises(ValueError):
            img = Image.from_file(sample)

    @pytest.mark.xfail(reason="Image object can not detect these image defects")
    @pytest.mark.parametrize(
        "sample", [str(file) for file in UNDETECTED_IMAGES]
    )
    def test_from_invalid_file_uncaught(self, sample):
        """There images are broken, but Image object won't detect"""
        with pytest.raises(ValueError):
            img = Image.from_file(sample)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
