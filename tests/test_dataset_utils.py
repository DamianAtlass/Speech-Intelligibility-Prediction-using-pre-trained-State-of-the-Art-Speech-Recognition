
from utils.new_config_dataclass import DataSplitConfig, DatasetConfig
from utils.dataset_utils import apply_split, get_dataset_dict, get_dataset
from datasets import Dataset, DatasetDict
import pytest

from utils.grid_utils import get_grid

from utils.paths import TEST_GRID_FOLDER, TEST_BC_FOLDER, GRID_FOLDER, BC_FOLDER

@pytest.mark.parametrize(("dataset_type", "dataset_path", "l"), [
    ("grid", TEST_GRID_FOLDER, 2000),
    ("grid_bc", TEST_BC_FOLDER, 24)
])
def test__get_dataset(dataset_type, dataset_path, l):
    dataset: Dataset = get_dataset(DataSplitConfig(dataset_type=dataset_type, path=dataset_path, start=0, end=l, noise=False))
    assert isinstance(dataset, Dataset)
    assert l == len(dataset)

def test_get_dataset_exception():
    try:
        dataset: Dataset = get_dataset(DataSplitConfig(dataset_type="sdfs", path="dataset_path", start=0, end=10, noise=False))
    except NotImplementedError:
        assert True


@pytest.mark.parametrize(("start", "end", "scaling", "expected_size"), [
    (0, 34_000, 1, 34_000),
    (0, 20_000, 1, 20_000),
    (0, 20_000, 0.5, 10_000),
    (0, 5_000, 1, 5_000),
    (5_000, 10_000, 1, 5_000),
])
def test_apply_split(start, end, scaling, expected_size: int):
    dataset = get_grid(GRID_FOLDER) # len == 34,000
    dataset = apply_split(dataset, start=start, end=end, scaling=scaling)
    assert len(dataset) == expected_size


@pytest.mark.parametrize(("dataset_type", "dataset_path"), [
    ("grid", GRID_FOLDER),
    ("grid_bc", BC_FOLDER),
])
def test_add_noise(dataset_type, dataset_path):
    dataset = get_dataset(DataSplitConfig(dataset_type=dataset_type, path=dataset_path, start=0, end=1, noise=True))

    assert isinstance(dataset, Dataset)


def test_get_dataset():
    config = DataSplitConfig(dataset_type="grid", path=None, start=0, end=10, scaling=1, noise=True)
    dataset: Dataset = get_dataset(config)
    assert isinstance(dataset, Dataset)

def test_get_dataset_dict():
    dataset_config = DatasetConfig(
        train_split= DataSplitConfig(dataset_type="grid", path=None, start=0, end=1., scaling=1, noise=True),
        test_split=DataSplitConfig(dataset_type="grid_bc", path=None, start=0, end=.2, scaling=1, noise=False),
        val_split=DataSplitConfig(dataset_type="grid_bc", path=None, start=.2, end=1., scaling=1, noise=False),
    )
    data_dict = get_dataset_dict(dataset_config)
    assert isinstance(data_dict, DatasetDict)