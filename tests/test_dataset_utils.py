from utils.dataset_utils import get_dataset
from datasets import Dataset, DatasetDict, load_from_disk


def test_get_dataset():
    dataset: Dataset = get_dataset(dataset_type="grid")
    assert isinstance(dataset, Dataset)

def test_get_dataset_exception():
    try:
        dataset: Dataset = get_dataset(dataset_type="asdf")
    except KeyError:
        assert True