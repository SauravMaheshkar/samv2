import numpy as np
import pytest
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils import download_weights
from sam2.utils.misc import variant_to_config_mapping


def test_video_predictor(download_weights) -> None:
    video_predictor = build_sam2_video_predictor(
        variant_to_config_mapping["tiny"],
        "./artifacts/sam2_hiera_tiny.pt",
    )
    inference_state = video_predictor.init_state(
        video_path="tests/assets/images/",
    )
    point = np.array([[150, 150]])
    label = np.array([1])
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,
    )

    assert isinstance(out_mask_logits, torch.Tensor)
