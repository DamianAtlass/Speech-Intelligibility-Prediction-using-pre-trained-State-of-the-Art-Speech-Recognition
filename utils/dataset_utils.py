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
    "grid": Path("datasets/grid"),
    "grid_bc": Path("datasets/GridIntelligibilityDatabase")
}

def get_dataset(dataset_type: str, dataset_path: Path = None):

    if not dataset_path:
        dataset_path = default_dataset_paths[dataset_type]

    if dataset_type=="grid":
        return get_grid(dataset_path if dataset_path else default_dataset_paths[dataset_type])
    elif dataset_type=="grid_bc":
        get_grid_bc()
    raise NotImplementedError(f"Dataset type {dataset_type} not implemented")