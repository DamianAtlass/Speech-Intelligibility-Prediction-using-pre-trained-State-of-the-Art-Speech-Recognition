from utils.grid_utils import get_grid
from utils.grid_bc_utils import get_grid_bc

import subprocess
import tarfile
from shutil import rmtree
import librosa
from datasets import Dataset, DatasetDict, load_from_disk
import wave
from datasets import Audio
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import cast

WANTED_SAMPLE_RATE = 16_000




default_dataset_paths = {
    "grid": Path.cwd()/"datasets/grid",
    "grid_bc": Path.cwd()/"datasets/GridIntelligibilityDatabase"
}

def get_dataset(dataset_type: str, dataset_path: Path = None):

    if not dataset_path:
        dataset_path = default_dataset_paths[dataset_type]

    if dataset_type=="grid":
        return get_grid(dataset_path if dataset_path else default_dataset_paths[dataset_type])
    elif dataset_type=="grid_bc":
        return get_grid_bc(dataset_path if dataset_path else default_dataset_paths[dataset_type])
    raise NotImplementedError(f"Dataset type {dataset_type} not implemented")


def apply_split(dataset : Dataset,
                train_split: int | float,
                test_split: int | float,
                val_split: int | float,
                dataset_scaling: int | float ) -> DatasetDict:
    """
    Split the dataset depending on the given parameters.

    Returns:
        DatasetDict
    """
    def calculate_size(len_:int, n: float | int) -> int:
        return cast(int, int(n * len_) if isinstance(n, float) else n)

    l = len(dataset)
    train_size = calculate_size(l, train_split)
    test_size = calculate_size(l, test_split)
    val_size = calculate_size(l, val_split)

    d = {}
    for label, split_value in zip(["train", "test", "val"], [train_size, test_size, val_size]):
        if split_value == 0:
            continue

        elif split_value == l:
            d[label] = dataset
            l = 0

        else:
            split = dataset.train_test_split(
                train_size=split_value,
                shuffle=True,
                seed=0,
            )
            d[label] = split["train"]
            dataset = split["test"]
            l = len(dataset)

    dataset_dict = DatasetDict(d)

    if dataset_scaling != 1:
        for split in ["train", "test", "val"]:
            try:
                dataset_dict[split] = dataset_dict[split].select(range(
                    int(len(dataset_dict[split]) * dataset_scaling)
                ))
            except KeyError as e:
                pass

    return dataset_dict
