import pytest
from utils.grid_utils import get_grid, apply_split
from utils.config_dataclasses import Config
from pathlib import Path

@pytest.mark.parametrize(("split", "resulting_size"), [
        ((0.7, 0.2, 0.1), (23_800, 6800, 3400)),
        ((0.5, 0.35, 0.15), (17000, 11900, 5100)),
    #        "test_group_training_config.ini",
])
def test_apply_split(split: tuple, resulting_size: tuple):
    train_split, test_split, val_split = split

    dataset = get_grid() # len == 34,000
    config = Config(model="",
                    model_type="",
                    model_path=Path(""),
                    output_path=Path(""),
                    dataset_path=Path("datasets/grid/"),
                    train_split=train_split,
                    test_split=test_split,
                    val_split=val_split)

    dataset_dict = apply_split(dataset, config)

    assert len(dataset_dict["train"]) == resulting_size[0]
    assert len(dataset_dict["test"]) == resulting_size[1]
    assert len(dataset_dict["val"]) == resulting_size[2]
