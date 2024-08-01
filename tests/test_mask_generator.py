import numpy as np
import pytest
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.utils import download_weights
from sam2.utils.misc import variant_to_config_mapping


@pytest.mark.full
def test_mask_generator(download_weights) -> None:
    image = Image.open("tests/assets/images/00.jpeg")
    image = np.array(image.convert("RGB"))
    model = build_sam2(
        variant_to_config_mapping["tiny"],
        "./artifacts/sam2_hiera_tiny.pt",
    )
    mask_generator = SAM2AutomaticMaskGenerator(model)
    masks = mask_generator.generate(image)
    assert len(masks) > 0
