from datasets import Dataset, DatasetDict, load_dataset
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import cast
import numpy as np
import os
from tqdm import tqdm
import json
import soundfile as sf
import shutil

from utils.grid_utils import get_grid
from utils.grid_bc_utils import get_grid_bc
from utils.manipulate_audio import add_noise_transformation
from utils.paths import GRID_FOLDER, BC_FOLDER, PROJECT_ROOT
from utils.new_config_dataclass import DatasetConfig, DataSplitConfig
WANTED_SAMPLE_RATE = 16_000
SNRS = [-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, None]


default_dataset_paths = {
    "grid": GRID_FOLDER,
    "grid_bc": BC_FOLDER
}

def _get_dataset(dataset_type: str,
                 path: Path|None) -> Dataset:
    if dataset_type != "libri":
        if dataset_type not in default_dataset_paths.keys():
            pass
            raise NotImplementedError(f"Dataset type {dataset_type} not implemented")

        if not path:
            dataset_path = default_dataset_paths[dataset_type]
        else:
            dataset_path = path

    match dataset_type:
        case "grid":
            return get_grid(dataset_path)
        case "grid_bc":
            return get_grid_bc(dataset_path)
        case "libri":
            return get_libri()
    raise RuntimeError(f"Dataset type {dataset_type} not implemented")

def get_libri() -> Dataset:
    ds = load_dataset("openslr/librispeech_asr", token=os.getenv("HF_TOKEN"))
    return ds["validation.clean"]

def get_dataset(split: DataSplitConfig) -> Dataset:

    dataset = _get_dataset(split.dataset_type, split.path)
    dataset = dataset.shuffle(seed=0)
    dataset = apply_split(dataset, split.start, split.end, split.scaling)
    if split.noise:
        dataset = add_noise_to_dataset(dataset)

    return dataset

def apply_filter(dataset: Dataset, filter_items: dict) -> Dataset:
    indices = []
    for k,v in filter_items.items():
        for i, sentence in tqdm(enumerate(dataset[k])):
            if sentence in v:
                indices.append(i)

        dataset = dataset.select(indices)
        indices = []

    return dataset

def get_dataset_dict(config: DatasetConfig) -> DatasetDict:
    dataset_dict = DatasetDict({})

    for split, label in zip([config.train_split, config.val_split, config.test_split], ["train", "val", "test"]):
        if split is None:
            continue
        dataset = get_dataset(split)
        dataset_dict[label] = dataset

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

def create_manifest(manifest_path: Path, dataset: Dataset) -> Path:
    """
    Create manifest file.
    Args:
        dataset:
        manifest_path: ends with .jsonl

    Returns:
    """
    assert str(manifest_path).endswith(".jsonl")
    logger.info(f"Create manifest for {manifest_path.name}...")
    records = []
    for sample in tqdm(dataset):

        record = {
            "audio_filepath": str(PROJECT_ROOT/sample["audio_path"]),
            "text": sample["sentence"],
            "duration": len(sample["audio"]["array"]) / 16_000
        }
        records.append(record)

    with open(str(manifest_path), 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    manifest_path.exists()

    return manifest_path


def override_noised_dataset_as_files(target_folder="datasets/grid_with_noise") -> None:
    """
    Use this function when noise cant be applied using a transformation, bc saved files are needed (in a manifest for example).
    Returns:

    Steps:
    # 1: copy the original directory ("datasets/grid")
    # 2: rm the saved_dataset directory
    # 3: uncomment the double-noise prevention in DataSplitConfigs !!!!
    # 4: run this function

    Example:
    shutil.copytree(src="datasets/grid", dst=grid_with_noise)
    shutil.rmtree("datasets/grid_with_noise/saved_dataset")
    override_noised_dataset_as_files(target_folder=grid_with_noise)

    """
    split_config = DataSplitConfig(dataset_type='grid', path=target_folder, start=0, end=1., noise=True,
                                   scaling=1.0)
    if not split_config.noise:
        raise ValueError("Sure this makes sense?")

    saved_dataset_path = PROJECT_ROOT / target_folder/ "saved_dataset"
    if saved_dataset_path.exists():
        raise RuntimeError(f"Did you already delete that: f{saved_dataset_path}")


    print("Read the new dataset...") #important bc of the paths
    dataset: Dataset = get_dataset(split_config)

    def validate_path(audio_path: Path):
        path_list: list[str] = list(audio_path.parts)
        assert path_list[0] == "datasets"
        assert path_list[1] not in ["grid", "GridIntelligibilityDatabase"], "Don't override the original data!"

    print("Apply noise and save individual files...")
    for sample in tqdm(dataset):
        audio_path = Path(sample["audio_path"])
        validate_path(audio_path)
        audio_array = sample["audio"]["array"] # noise is applied automatically with a transformation, see dataset_utils

        sf.write(str(audio_path), audio_array, 16_000)
    print("Files copied.")
    del dataset
    print("Remove unnoised dataset...")
    shutil.rmtree(str(saved_dataset_path))


def create_grid_without_bc_sentences(source=None, dest="datasets/grid_without_bc_sentences") -> None:

    # for noised: create_grid_without_bc_sentences(source="datasets/grid_with_noise", dest="datasets/grid_with_noise_without_bc_sentences")
    #       (execute override_noised_dataset_as_files before)
    # normal: create_grid_without_bc_sentences()
    grid = get_dataset(split=DataSplitConfig(dataset_type='grid', path=source, start=0, end=1.0, noise=False, scaling=1.0))
    grid_bc = get_dataset(split=DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=1.0, noise=False, scaling=1.0))

    sentences_in_grid = grid.unique("sentence")
    sentences_in_grid_bc = grid_bc.unique("sentence")
    sentences_in_not_in_bc = set(sentences_in_grid) - set(sentences_in_grid_bc)
    del sentences_in_grid, sentences_in_grid_bc

    grid = apply_filter(grid, {"sentence": sentences_in_not_in_bc})

    save_at = Path.cwd() / dest / "saved_dataset"
    if save_at.exists(): raise RuntimeError("Dataset already exists!")
    save_at.mkdir(parents=True, exist_ok=True)
    grid.save_to_disk(save_at)


def create_grid_bc_without_duplicates() -> None:
    split_config = DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=1.0, noise=False, scaling=1.0)
    dataset = get_dataset(split_config)

    seen = set()
    indices = []

    for i, sentence in tqdm(enumerate(dataset["sentence"])):
        if sentence not in seen:
            seen.add(sentence)
            indices.append(i)

    dataset = dataset.select(indices)

    from collections import Counter
    print(Counter(dataset["snr_db"]))

    save_at = Path.cwd() / "datasets" / "grid_bc_without_duplicates" / "saved_dataset"
    save_at.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_at)
