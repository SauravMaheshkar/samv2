import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "point, label, box, num_masks",
    [
        pytest.param(np.array([[150, 150]]), np.array([1]), None, 1, id="single_point"),
        pytest.param(
            np.array([[150, 150], [120, 120]]),
            np.array([1, 0]),
            None,
            1,
            id="multiple_points",
        ),
        pytest.param(None, None, np.array([50, 150, 50, 150]), 1, id="single_box"),
        pytest.param(
            None,
            None,
            np.array([[50, 150, 50, 150], [75, 175, 75, 175]]),
            2,
            id="multiple_boxes",
        ),
        pytest.param(
            np.array([[120, 120]]),
            np.array([0]),
            np.array([50, 150, 50, 150]),
            1,
            id="box_and_points",
        ),
    ],
)
def test_prompt_mode(load_image, image_predictor, point, label, box, num_masks) -> None:
    image_predictor.set_image(load_image)

    assert isinstance(image_predictor._features["image_embed"], torch.Tensor)

    masks, _, _ = image_predictor.predict(
        point_coords=point,  # type: ignore
        point_labels=label,  # type: ignore
        box=box,  # type: ignore
        multimask_output=False,
    )

    assert masks.shape[0] == num_masks


@pytest.mark.full
def test_mask_generation(load_image, mask_generator) -> None:
    masks = mask_generator.generate(load_image)
    assert len(masks) > 0
