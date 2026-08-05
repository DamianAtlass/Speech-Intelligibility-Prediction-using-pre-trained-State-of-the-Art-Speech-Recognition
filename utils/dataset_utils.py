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
from utils.paths import GRID_FOLDER, BC_FOLDER
from utils.new_config_dataclass import DatasetConfig, DataSplitConfig



default_dataset_paths = {
    "grid": GRID_FOLDER,
    "grid_bc": BC_FOLDER
}

def _get_dataset(dataset_type: str,
                 path: Path|None) -> Dataset:
    if dataset_type not in default_dataset_paths.keys():
        raise NotImplementedError(f"Dataset type {dataset_type} not implemented")

    if not path:
        dataset_path = default_dataset_paths[dataset_type]
    else:
        dataset_path = path

    return get_grid(dataset_path)

def get_dataset(split: DataSplitConfig) -> Dataset:

    dataset = _get_dataset(split.dataset_type, split.path)
    dataset = dataset.shuffle(seed=0)
    dataset = apply_split(dataset, split.start, split.end, split.scaling)
    if split.noise:
        dataset = add_noise_to_dataset(dataset)

    return dataset

def get_dataset_dict(config: DatasetConfig) -> DatasetDict:
    dataset_dict = {}
    for split, label in zip([config.train_split, config.test_split, config.val_split], ["train", "test", "val"]):
        if split is None:
            continue
        dataset = get_dataset(split)
        dataset_dict[label] = dataset

    dataset_dict = DatasetDict(dataset_dict)
    return dataset_dict

def add_noise_to_dataset(dataset: Dataset) -> Dataset:
    logger.info(f"Add noise to all {len(dataset)} samples.")
    rng = np.random.default_rng(0)
    dataset = dataset.add_column("snr", rng.choice(SNRS, size=len(dataset)))
    dataset.set_transform(add_noise_transformation)
    return dataset

def apply_split(dataset : Dataset,
                start: int | float = 0,
                end: int | float = 1.,
                scaling: int | float = 1) -> Dataset:
    """
    Split the dataset depending on the given parameters.

    Returns:
        DatasetDict
    """
    def calculate_size(len_:int, n: float | int) -> int:
        return cast(int, int(n * len_) if isinstance(n, float) else n)

    l = len(dataset)
    start = calculate_size(l, start)
    end = calculate_size(l, end)

    start = int(start * scaling)
    end = int(end * scaling)

    if not (start == 0 and end == len(dataset)):
        dataset =  dataset.select(range(start, end))


    return dataset
