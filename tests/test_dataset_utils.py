from tests.test_grid_utils import real_grid_folder
from utils.config_dataclasses import Config
from utils.dataset_utils import get_dataset, apply_split
from datasets import Dataset, DatasetDict
from pathlib import Path
import pytest

from utils.grid_utils import get_grid

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()
grid_folder = test_folder / "grid"
grid_bc_folder = test_folder / "grid_bc"

@pytest.mark.parametrize(("dataset_type", "dataset_path", "l"), [
    ("grid", grid_folder, 2000),
    ("grid_bc", grid_bc_folder, 24)
])
def test_get_dataset(dataset_type, dataset_path, l):
    dataset: Dataset = get_dataset(dataset_type=dataset_type, dataset_path=dataset_path)
    assert isinstance(dataset, Dataset)
    assert l == len(dataset)

def test_get_dataset_exception():
    try:
        dataset: Dataset = get_dataset(dataset_type="asdf")
    except KeyError:
        assert True


@pytest.mark.parametrize(("split", "resulting_size"), [
        ((0.7, 0.2, 0.1, 1), (23_800, 6800, 3400)),
        ((0.5, 0.35, 0.15, 1), (17000, 11900, 5100)),
        ((0.5, 0.1, 0.1, 1), (17000, 3400, 3400)),
        ((0.8, 0.2, 0, 1), (27200, 6800, 0)),
        ((1, 2, 3, 1), (1, 2, 3)),
        ((0.5, 0.1, 0.1, 0.5), (8500, 1700, 1700)),
        ((100, 101, 102, 1), (100, 101, 102)),
        ((1., 0, 0, 1), (34000, 0, 0)),
        ((0, 1., 0, 1), (0, 34000, 0)),
        ((0, 0, 1., 1), (0, 0, 34000)),
        ((10, 10, 0, 1), (10, 10, 0)),
        ((10, 0, 10, 1), (10, 0, 10)),
        ((0, 10, 10, 1), (0, 10, 10)),
])
def test_apply_split(split: tuple, resulting_size: tuple):
    dataset = get_grid(real_grid_folder) # len == 34,000

    dataset_dict = apply_split(dataset, *split)

    for s, n in zip(["train", "test", "val"], [0,1,2]):
        if resulting_size[n] != 0:
            assert len(dataset_dict[s]) == resulting_size[n]
        else:
            try:
                assert len(dataset_dict[s]) == resulting_size[n]
                assert False
            except KeyError:
                assert True


@pytest.mark.parametrize(("scale", "resulting_size"), [
    (1, 34000),
    (0.5, 17000)
])
def test_apply_split_for_full_val_split(scale: int | float, resulting_size: int):
    dataset = get_grid(real_grid_folder) # len == 34,000
    config = Config(model="",
                    model_type="",
                    model_path=Path(""),
                    output_path=Path("tests/inference_test"),
                    dataset_type="grid",
                    dataset_path=Path("datasets/grid/"),
                    train_split=0,
                    test_split=0,
                    val_split=1.,
                    dataset_scaling=scale)

    dataset_dict = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    assert len(dataset_dict["val"]) == resulting_size


@pytest.mark.parametrize(("dataset_type", "dataset_path"), [
    ("grid", grid_folder),
    ("grid_bc", grid_bc_folder),
])
def test_get_dataset_and_add_noise(dataset_type, dataset_path):
    try:
        dataset  = get_dataset(dataset_type=dataset_type, dataset_path=dataset_path, add_noise=True)
        assert dataset_type=="grid"
    except ValueError:
        assert dataset_type=="grid_bc"