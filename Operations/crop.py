import numpy as np

from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle
from IMGBOX.Operations.base import SingularOperation


class Crop(SingularOperation):
    """Simple Canny operation"""

    def __init__(self, cropped_region: Rectangle):
        self._cropped_region = cropped_region
        self.ymin, self.xmin, self.ymax, self.xmax = \
            cropped_region.ymin, cropped_region.xmin, \
            cropped_region.ymax, cropped_region.xmax

    def _operate(self, img: np.array) -> np.array:
        if self.ymin < 0 or self.ymax > img.shape[0] or \
                self.xmin < 0 or self.xmax > img.shape[1]:
            msg = "Invalid cropped region {}".format(self._cropped_region)
            raise ValueError(msg)

        return img[self.ymin:self.ymax, self.xmin:self.xmax, ...]
