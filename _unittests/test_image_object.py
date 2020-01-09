import pathlib

import pytest
import cv2
import numpy as np

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

    def test_properties(self):
        """Image name, h, w and shape properties should match that of .numpy()"""
        file = SAMPLE_IMAGES[0]
        img = Image.from_file(str(file))
        assert img.name == file.stem

        array = img.numpy()
        assert img.h == array.shape[0]
        assert img.w == array.shape[1]
        assert len(img.shape) == 3
        assert img.shape == array.shape

    @pytest.mark.parametrize(
        "sample", [str(file) for file in INVALID_IMAGES]
    )
    def test_from_invalid_file(self, sample):
        """Create Image object from invalid files should raise ValueError"""
        with pytest.raises(ValueError):
            img = Image.from_file(sample)

    def test_save_to_file(self, tmp_path: pathlib.Path):
        file = str(SAMPLE_IMAGES[0])
        img = Image.from_file(file)
        ori_arr = img.numpy()

        file_name = "test_save_file.png"
        file = tmp_path.joinpath(file_name)
        img.save(str(file))

        assert file.is_file()
        assert np.allclose(cv2.imread(str(file)), ori_arr)

        # save to same file should return error
        file2 = str(SAMPLE_IMAGES[1])
        img = Image.from_file(file2)

        with pytest.raises(ValueError):
            img.save(str(file), overwrite=False)

        img.save(str(file), overwrite=True)

    @pytest.mark.xfail(reason="Image object can not detect these image defects")
    @pytest.mark.parametrize(
        "sample", [str(file) for file in UNDETECTED_IMAGES]
    )
    def test_from_invalid_file_uncaught(self, sample):
        """There images are broken, but Image object won't detect"""
        with pytest.raises(ValueError):
            img = Image.from_file(sample)

    def test_resize(self):
        """Test image resize to target shape"""
        file = str(SAMPLE_IMAGES[0])
        img = Image.from_file(file)

        img.resize((224, 224))
        assert img.shape == img.numpy().shape == (224, 224, 3)

        img.resize((448, 448), interpolation="INTER_LINEAR")
        assert img.shape == img.numpy().shape == (448, 448, 3)

        img.resize((224, 224), interpolation="INTER_CUBIC")
        assert img.shape == img.numpy().shape == (224, 224, 3)

        img.resize((448, 448), interpolation="INTER_LANCZOS4")
        assert img.shape == img.numpy().shape == (448, 448, 3)

    def test_resize_invalid_shape(self):
        """.resize() should raise ValueError when shape is invalid"""
        file = str(SAMPLE_IMAGES[0])
        img = Image.from_file(file)

        with pytest.raises(ValueError):
            img.resize((224, ))

        with pytest.raises(ValueError):
            img.resize((224, -1))

        with pytest.raises(ValueError):
            img.resize((113.6, 3.1415))

    def test_resize_invalid_interpolation(self):
        """.resize() should raise ValueError when interpolation is invalid"""
        file = str(SAMPLE_IMAGES[0])
        img = Image.from_file(file)

        with pytest.raises(ValueError):
            img.resize((224, 224), interpolation="NOT_EXIST")


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
