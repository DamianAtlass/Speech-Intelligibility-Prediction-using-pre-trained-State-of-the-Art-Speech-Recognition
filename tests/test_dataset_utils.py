from utils.dataset_utils import get_dataset
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
import pytest

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