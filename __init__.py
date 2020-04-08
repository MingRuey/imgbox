from IMGBOX.core import *
from IMGBOX.shapes import *
from IMGBOX.Operations.edges import *
from IMGBOX.Operations.crop import *
from IMGBOX.Operations.draw import *
from IMGBOX.Operations.overlap import *
from IMGBOX.Operations.difference import *
from IMGBOX.Operations.correlation import *
from IMGBOX.Visualization.plot import *


def list_ops():
    """Get a list of availiable operations"""
    return [op for op in globals().keys() if not op.startswith("__")]
