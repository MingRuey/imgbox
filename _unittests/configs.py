import pathlib


SAMPLE_DIR = pathlib.Path(__file__).parent.joinpath("samples")
assert SAMPLE_DIR.is_dir()

SAMPLE_IMAGES = list(SAMPLE_DIR.glob("sample*.*"))
assert SAMPLE_IMAGES

INVALID_IMAGES = list(SAMPLE_DIR.glob("Invalid*.*"))
assert INVALID_IMAGES

UNDETECTED_IMAGES = list(SAMPLE_DIR.glob("Undetected*.*"))
assert UNDETECTED_IMAGES
