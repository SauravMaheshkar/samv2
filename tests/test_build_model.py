import pytest
import torch

from sam2.build_sam import build_sam2
from sam2.utils import download_weights
from sam2.utils.misc import variant_to_config_mapping


@pytest.mark.full
@pytest.mark.parametrize(
    "variant",
    ["tiny", "small", "base_plus", "large"],
)
def test_build_sam(download_weights, variant: str):
    model = build_sam2(
        variant_to_config_mapping[variant],
        f"./artifacts/sam2_hiera_{variant}.pt",
    )

    assert isinstance(model, torch.nn.Module)
