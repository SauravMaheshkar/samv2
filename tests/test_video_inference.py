import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "point, label",
    [
        pytest.param(np.array([[150, 150]]), np.array([1]), id="single_point"),
    ],
)
def test_video_predictor(video_predictor, point, label) -> None:
    inference_state = video_predictor.init_state(
        video_path="tests/assets/images/",
    )
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,
    )

    assert isinstance(out_mask_logits, torch.Tensor)
    # TODO: Add better tests
