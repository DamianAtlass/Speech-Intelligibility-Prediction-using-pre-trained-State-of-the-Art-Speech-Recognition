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
import numpy as np
from utils.manipulate_audio import add_noise_transformation
WANTED_SAMPLE_RATE = 16_000
SNRS = [-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, None]




default_dataset_paths = {
    "grid": Path.cwd()/"datasets/grid",
    "grid_bc": Path.cwd()/"datasets/GridIntelligibilityDatabase"
}

def _get_dataset(dataset_type: str, dataset_path: Path = None) -> Dataset:

    if not dataset_path:
        dataset_path = default_dataset_paths[dataset_type]

    if dataset_type=="grid":
        return get_grid(dataset_path if dataset_path else default_dataset_paths[dataset_type])
    elif dataset_type=="grid_bc":
        return get_grid_bc(dataset_path if dataset_path else default_dataset_paths[dataset_type])
    raise NotImplementedError(f"Dataset type {dataset_type} not implemented")

def get_dataset(dataset_type: str, dataset_path: Path = None, add_noise: bool = False) -> Dataset:
    dataset = _get_dataset(dataset_type, dataset_path)
    
    if add_noise:
        if dataset_type=="grid_bc":
            raise ValueError("Do you really wanna do this?")
        dataset = add_noise_to_dataset(dataset)
    return dataset

def add_noise_to_dataset(dataset: Dataset):
    logger.info(f"Add noise to all {len(dataset)} samples.")
    rng = np.random.default_rng(0)
    dataset = dataset.add_column("snr", rng.choice(SNRS, size=len(dataset)))
    dataset.set_transform(add_noise_transformation)
    return dataset

def apply_split(dataset : Dataset,
                train_split: int | float,
                test_split: int | float,
                val_split: int | float,
                dataset_scaling: int | float = 1) -> DatasetDict:
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
