from IMGBOX.core import Image
from IMGBOX.shapes import Rectangle
from IMGBOX.Operations.edges import Canny
from IMGBOX.Operations.crop import Crop
from IMGBOX.Operations.correlation import CrossCorrelate2D
from IMGBOX.Visualization.plot import display


def list_ops():
    """Get a list of availiable operations"""
    return [op.__name__ for op in (Canny, Crop, CrossCorrelate2D)]
