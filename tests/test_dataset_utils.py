from utils.dataset_utils import get_dataset
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()
grid_folder = test_folder / "grid"
def test_get_dataset():
    dataset: Dataset = get_dataset(dataset_type="grid", dataset_path=grid_folder)
    assert isinstance(dataset, Dataset)

def test_get_dataset_exception():
    try:
        dataset: Dataset = get_dataset(dataset_type="asdf")
    except KeyError:
        assert True