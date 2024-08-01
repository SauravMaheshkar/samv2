import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils import download_weights
from sam2.utils.misc import variant_to_config_mapping


def test_image_predictor_with_single_point(download_weights) -> None:
    image = Image.open("tests/assets/images/00.jpeg")
    image = np.array(image.convert("RGB"))
    model = build_sam2(
        variant_to_config_mapping["tiny"],
        "./artifacts/sam2_hiera_tiny.pt",
    )
    image_predictor = SAM2ImagePredictor(model)
    image_predictor.set_image(image)

    assert isinstance(image_predictor._features["image_embed"], torch.Tensor)

    point = np.array([[150, 150]])
    label = np.array([1])

    masks, scores, logits = image_predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    assert isinstance(masks, np.ndarray)
    assert masks.shape[0] == 3
    assert isinstance(scores, np.ndarray)
    assert isinstance(logits, np.ndarray)
