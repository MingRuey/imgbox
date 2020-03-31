import operator
import pathlib

import pytest
import cv2
import numpy as np

from IMGBOX.core import Image
from IMGBOX._unittests.configs import INVALID_IMAGES, SAMPLE_IMAGES
from IMGBOX._unittests.configs import UNDETECTED_IMAGES, IMAGE_BW


def _is_ref_unequal(array1, array2):
    """Return if altering one array would affect another"""
    if not np.all(array1 == array2):
        return True

    array1[0, 0] = 100
    flag1 = np.all(array1 == array2)

    array2[1, 1] = 100
    flag2 = np.all(array1 == array2)
    return (not flag1) and (not flag2)


class TestImageObject:

    def test_from_file_color(self):
        """Image object can and should be loaded from color image file"""
        for file in SAMPLE_IMAGES:
            img = Image.from_file(str(file))
            assert img.name == file.stem
            assert img.is_color

    def test_from_file_blackwhite(self):
        """Image object can and should be loaded from black white image"""
        img = Image.from_file(IMAGE_BW)
        assert img.name == IMAGE_BW.stem
        assert not img.is_color

    def test_from_array(self):
        """Image object can created from uint8 numpy array"""
        array = np.random.randint(0, 255, size=(100, 500, 3), dtype=np.uint8)
        img = Image(array)
        assert img.name == "array_" + str(id(array))
        assert img.is_color
        assert _is_ref_unequal(img, array)

        array = np.random.randint(0, 255, size=(100, 500), dtype=np.uint8)
        img = Image(array)
        assert img.name == "array_" + str(id(array))
        assert not img.is_color
        assert _is_ref_unequal(img, array)

    def test_from_invalid_array(self):
        """Image object should refuse invalid array"""
        wrong_dimension = np.random.randint(
            0, 255, size=(100, 500, 1), dtype=np.uint8
        )
        with pytest.raises(ValueError):
            img = Image(wrong_dimension, to_color=False)

        wrong_dtype = np.random.randint(0, 255, size=(100, 500, 3))
        wrong_dtype = wrong_dtype.astype(np.float32)
        with pytest.raises(ValueError):
            img = Image(wrong_dtype)

    @pytest.mark.parametrize(
        "img", [
            Image(np.random.randint(0, 255, size=(100, 500, 3), dtype=np.uint8)),
            Image(np.random.randint(0, 255, size=(100, 500), dtype=np.uint8))
        ],
        ids=["color", "bw"]
    )
    def test_change_type(self, img):
        """.astype(dtype) should use copy of original array, and keep name"""
        assert img.dtype == np.uint8

        change_type = img.astype(np.float32)
        assert np.allclose(change_type, img)
        assert change_type.name == img.name
        assert _is_ref_unequal(change_type, img)

        # check the reference of the numpy
        unchange_type = img.astype(np.uint8)
        assert np.allclose(unchange_type, img)
        assert unchange_type.name == img.name
        assert _is_ref_unequal(unchange_type, img)

    def test_slice(self):
        """Slice on image should use original array, and keep name"""
        img = Image(np.random.randint(0, 255, size=(100, 500, 3), dtype=np.uint8))

        sliced = img[...]
        assert np.allclose(sliced, img)
        assert sliced.name == img.name
        assert not _is_ref_unequal(sliced, img)

    @pytest.mark.parametrize(
        "img", [
            Image(np.random.randint(0, 255, size=(100, 500, 3), dtype=np.uint8)),
            Image(np.random.randint(0, 255, size=(100, 500), dtype=np.uint8))
        ],
        ids=["color", "bw"]
    )
    def test_recreate(self, img):
        """Recreate Image from given Image should use copy of original array"""
        recreate = Image(img)
        assert np.allclose(recreate, img)
        assert _is_ref_unequal(recreate, img)

    def test_properties(self):
        """Image name, h, w and shape properties should match that of .numpy()"""
        file = SAMPLE_IMAGES[0]
        img = Image.from_file(str(file))
        assert img.name == file.stem

        assert img.h == 260
        assert img.w == 260
        assert len(img.shape) == 3
        assert img.ndim == 3
        assert img.c == 3

    def test_color_conversion(self):
        """Convert to same color space return original ref, otherwise return the converted"""
        color = Image.from_file(SAMPLE_IMAGES[0])
        assert color.is_color

        color_to_color = color.to_color()
        assert color_to_color.is_color
        assert id(color_to_color) == id(color)

        color_to_gray = color.to_gray()
        assert not color_to_gray.is_color
        assert id(color_to_gray) != id(color)

        gray = Image.from_file(IMAGE_BW)
        gray_to_color = gray.to_color()
        assert gray_to_color.is_color
        assert id(gray_to_color) != id(gray)

        gray_to_gray = gray.to_gray()
        assert not gray_to_gray.is_color
        assert id(gray_to_gray) == id(gray)

    @pytest.mark.parametrize(
        "sample", [str(file) for file in INVALID_IMAGES]
    )
    def test_from_invalid_file(self, sample):
        """Create Image object from invalid files should raise ValueError"""
        with pytest.raises(ValueError):
            img = Image.from_file(sample)

    @pytest.mark.parametrize(
        "sample", [SAMPLE_IMAGES[0], IMAGE_BW], ids=["color", "bw"]
    )
    def test_save_to_file(self, tmp_path: pathlib.Path, sample):
        file = str(sample)
        img = Image.from_file(file)

        file_name = "test_save_file.png"
        file = tmp_path.joinpath(file_name)
        img.save(str(file))

        assert file.is_file()
        cv2_read = cv2.imread(str(file), flags=cv2.IMREAD_UNCHANGED)
        assert np.allclose(cv2_read, img)

        # save to same file should return error
        file2 = str(sample)
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

    def test_resize_color(self):
        """Test color image resize, it should keep original name"""
        file = str(SAMPLE_IMAGES[0])
        img = Image.from_file(file)
        assert img.is_color

        resize = img.resize((100, 400))
        assert img.name == resize.name
        assert resize.shape == (100, 400, 3)

        resize = img.resize((100, 400), interpolation="INTER_LINEAR")
        assert img.name == resize.name
        assert resize.shape == (100, 400, 3)

        resize = img.resize((400, 200), interpolation="INTER_CUBIC")
        assert img.name == resize.name
        assert resize.shape == (400, 200, 3)

    def test_resize_bw(self):
        """Test bw image resize, it should keep original name"""
        file = str(IMAGE_BW)
        img = Image.from_file(file)
        assert not img.is_color
        assert img.shape != (100, 400)

        resize = img.resize((100, 400))
        assert img.name == resize.name
        assert resize.shape == (100, 400)

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


_Operands = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


class TestMathOperation:

    @pytest.mark.parametrize(
        "operand", [k for k in _Operands.keys()]
    )
    def test_math_operators_with_constant(self, operand):
        """Test image basic math operation on constants"""
        operator = _Operands[operand]
        array = np.arange(1, 251).reshape(50, 5).astype(np.uint8)
        constant = 10

        image = Image(array)
        op = operator(image, constant)
        r_op = operator(constant, image)

        result = operator(array, constant)
        r_result = operator(constant, array)
        assert np.all(op == result)
        assert np.all(r_op == r_result)

        # commutative operator
        if operand in ["+", "*"]:
            assert np.all(op == r_op)

    @pytest.mark.parametrize(
        "operand", [k for k in _Operands.keys()]
    )
    def test_math_operators_with_images(self, operand):
        """Test image basic math operation with another image"""
        operator = _Operands[operand]

        array1 = np.arange(1, 251).reshape(10, 25).astype(np.uint8)
        array2 = np.arange(250, 0, -1).reshape(10, 25).astype(np.uint8)

        image1 = Image(array1)
        image2 = Image(array2)

        op = operator(image1, image2)
        r_op = operator(image2, image1)

        result = operator(array1, array2)
        r_result = operator(array2, array1)

        assert np.all(op == result)
        assert np.all(r_op == r_result)

        # commutative operator
        if operand in ["+", "*"]:
            assert np.all(op == r_op)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
