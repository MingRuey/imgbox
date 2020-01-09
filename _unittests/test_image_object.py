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

    def test_numpy_array(self):
        """.numpy() should give np.ndarray and to np.ndarray give the same"""
        file = str(SAMPLE_IMAGES[0])

        img = Image.from_file(file)
        numpy_method = img.numpy()
        assert isinstance(img.numpy(), np.ndarray)

        to_numpy = np.array(img)
        assert np.all(numpy_method == to_numpy)

        # check the reference of the numpy
        img2 = img.numpy()
        img2[0, ...] = 10
        to_numpy2 = np.array(img)
        to_numpy2[0, ...] = 100
        assert not np.allclose(img, img2)
        assert not np.allclose(to_numpy, img2)
        assert not np.allclose(to_numpy, to_numpy2)
        assert not np.allclose(img, to_numpy2)

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
