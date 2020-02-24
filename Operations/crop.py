import numpy as np

from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle
from IMGBOX.Operations.base import SingularOperation


class Crop(SingularOperation):
    """Simple Canny operation"""

    def __init__(self, cropped_region: Rectangle):
        """Sepcify the cropped region in Rectangle

        cropped_region specify region in pixel positions,
        i.e. [0, height-1] and [0, width -1].
        So it does not allow negative and float values.
        """
        if cropped_region.ymin < 0 or cropped_region.xmin < 0:
            raise ValueError("crop region must >= 0")

        self._cropped_region = cropped_region
        self.ymin, self.xmin, self.ymax, self.xmax = \
            int(cropped_region.ymin), int(cropped_region.xmin), \
            int(cropped_region.ymax), int(cropped_region.xmax)

    def _operate(self, img: np.array) -> np.array:
        if self.ymax > img.shape[0] or self.xmax > img.shape[1]:
            msg = "Cropped region out of range: cropped {} from image shape {}"
            raise ValueError(msg.format(self._cropped_region, img.shape))

        return img[self.ymin:self.ymax, self.xmin:self.xmax, ...]
