import numpy as np
import pytest
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils import download_weights
from sam2.utils.misc import variant_to_config_mapping


@pytest.fixture
def load_image() -> np.ndarray:
    image = Image.open("tests/assets/images/00.jpeg")
    image = np.array(image.convert("RGB"))
    return image


@pytest.fixture
def image_predictor(download_weights) -> "SAM2ImagePredictor":
    model = build_sam2(
        variant_to_config_mapping["tiny"],
        "./artifacts/sam2_hiera_tiny.pt",
    )
    image_predictor = SAM2ImagePredictor(model)
    return image_predictor


@pytest.fixture
def mask_generator(download_weights) -> "SAM2AutomaticMaskGenerator":
    model = build_sam2(
        variant_to_config_mapping["tiny"],
        "./artifacts/sam2_hiera_tiny.pt",
    )
    mask_generator = SAM2AutomaticMaskGenerator(model)
    return mask_generator


@pytest.fixture
def video_predictor(download_weights):
    return build_sam2_video_predictor(
        variant_to_config_mapping["tiny"],
        "./artifacts/sam2_hiera_tiny.pt",
    )
