from typing import Tuple

import cv2
import numpy as np
import tkinter as tk

from IMGBOX.core import Image


__all__ = ["display"]


def _get_curr_montior_geometry():
    """Get current monitor resolution, even in multiple monitors setup

    Note: copy-paste from: https://stackoverflow.com/questions/3129322/

    Returns:
        tuple of int: (windows height, windows width)
    """
    root = tk.Tk()
    root.withdraw()
    h = root.winfo_screenheight()
    w = root.winfo_screenwidth()
    return (h, w)


def _get_keep_aspect_ratio_shape(
        target_shape: tuple, dst_shape: tuple
        ) -> Tuple[int, int]:
    """Get the resize shape that fits into dst_shape with the
       aspect ratio (almost) unchanged.

    Args:
        target_shape: the shape of (h, w) to be resized
        dst_shape: the shape of (h, w) for target_shape to fits into

    Return
        tuple of (h, w), the new shape for target shape to resize
    """
    msg = "Shape must be tuple of (height, width)"
    assert len(target_shape) == len(dst_shape) == 2, msg

    target_h, target_w = target_shape
    dst_h, dst_w = dst_shape

    if target_h > dst_h or target_w > dst_w:
        ratio_h = target_h / dst_h
        ratio_w = target_w / dst_w
        if ratio_h > ratio_w:
            new_h = dst_h
            new_w = int(target_w / ratio_h)
        else:
            new_h = int(target_h / ratio_w)
            new_w = dst_w
        return (new_h, new_w)
    else:
        return target_h, target_w


def display(img: Image, title: str = None):
    """Display image content via pop-up windows"""
    dst_shape = _get_keep_aspect_ratio_shape(
        target_shape=img.shape[:2],
        dst_shape=_get_curr_montior_geometry()
    )

    if dst_shape != img.shape:
        img = img.resize(dst_shape)

    if title is None:
        title = img.name

    cv2.imshow(title, img.astype(np.uint8))
    cv2.waitKey()
