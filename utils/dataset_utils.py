import shutil

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
from collections import Counter

from utils.grid_utils import get_grid
from utils.grid_bc_utils import get_grid_bc
from utils.avsec_utils import get_avsec2_enhanced
from utils.manipulate_audio import add_noise_transformation
from utils.paths import GRID_FOLDER, BC_FOLDER, PROJECT_ROOT, AVSEC2_LTR
from utils.new_config_dataclass import DatasetConfig, DataSplitConfig
WANTED_SAMPLE_RATE = 16_000
SNRS = [-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, None]


default_dataset_paths = {
    "grid": GRID_FOLDER,
    "grid_bc": BC_FOLDER,
    "avsec2_enhanced": AVSEC2_LTR,
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
        case "avsec2_enhanced":
            return get_avsec2_enhanced()
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


def override_noised_dataset_as_files(target_folder: str = "datasets/grid_with_noise") -> None:
    """
    Use this function when noise cant be applied using a transformation, bc saved files are needed
    (in a manifest for training parakeet for example).
    Returns:

    Steps:
    # 1: uncomment the double-noise prevention in DataSplitConfigs !!!!
    # 2: run this function

    Example:
    shutil.copytree(src="datasets/grid", dst=grid_with_noise)
    shutil.rmtree("datasets/grid_with_noise/saved_dataset")
    override_noised_dataset_as_files(target_folder=grid_with_noise)

    """
    target_folder = Path(target_folder)
    if str("datasets/grid") in str(PROJECT_ROOT/target_folder):
        raise ValueError("Dont execute on original Grid folder, files would be overwritten!")

    split_config = DataSplitConfig(dataset_type='grid', path=target_folder, start=0, end=1.0, noise=True,
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

    print("Noisy files saved. Save dataset...")
    dataset.reset_format() #dataset cant be saved if not reset
    #dataset = dataset.rename_column("snr", "snr_db")
    dataset.save_to_disk(PROJECT_ROOT/target_folder/"saved_dataset")


def remove_bc_samples_from_grid_dataset(source: str=None, dest="datasets/grid_without_bc_sentences") -> None:
    """
    Use this function to remove samples from the original Grid corpus whose reference occurs in the BC dataset.
    Use specifically on a noised Grid to get a training set. It doesnt need to be a noised Grid, though.

    Args:
        source: str,  noised (or not) Grid source path
        dest: str, new dataset path

    Returns: None

    """
    save_at = PROJECT_ROOT / dest / "saved_dataset"
    if save_at.exists(): raise RuntimeError("Dataset already exists!")

    noisy_grid = get_dataset(split=DataSplitConfig(dataset_type='grid', path=Path(source), start=0, end=1.0, noise=False, scaling=1.0))
    # noisy_grid should still have the snr column, IF it has been created with override_noised_dataset_as_files

    grid_bc = get_dataset(split=DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=1.0, noise=False, scaling=1.0))

    sentences_in_grid = noisy_grid.unique("sentence")
    sentences_in_grid_bc = grid_bc.unique("sentence")
    sentences_not_in_bc = set(sentences_in_grid) - set(sentences_in_grid_bc)
    del sentences_in_grid, sentences_in_grid_bc

    noisy_grid_without_bc_samples = apply_filter(noisy_grid, {"sentence": sentences_not_in_bc})

    save_at.mkdir(parents=True, exist_ok=True)
    noisy_grid_without_bc_samples.save_to_disk(save_at)


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

def has_even_snrs(dataset: Dataset) -> bool:
    """
    Check if a dataset has an even number of SNRs.
    """
    return len(set(Counter(dataset["snr_db"]).values())) == 1

def even_out_snrs(dataset: Dataset):
    """
    Randomly choose samples so that the maximum possible samper per SNR exist.
    """

    if has_even_snrs(dataset):
        return dataset
    #needs testing
    snr_name = "snr" if "snr" in dataset.column_names else "snr_db"
    snr_counter = Counter(dataset[snr_name])

    n = min(snr_counter.values())
    snrs = np.asarray(dataset[snr_name])  # reads only that one column
    rng = np.random.default_rng(0)

    dataset_list = [
        rng.choice(np.flatnonzero(snrs == snr), n, replace=False)
        for snr in snr_counter
    ]
    indices = np.concatenate(dataset_list)
    balanced_snrs = dataset.select(np.sort(indices))
    assert has_even_snrs(balanced_snrs)
    return balanced_snrs.shuffle(seed=0)


def grid_bc_get_dataset_with_unique_audio_samples(dest: str = "datasets/grid_bc_unique_audios"):
    save_at = PROJECT_ROOT / dest / "saved_dataset"
    grid_bc = get_dataset(
        DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=1.,
                        noise=False, scaling=1.0))

    seen_audios = set()  # unique audio is identified by speaker & snr & reference sentence (NOT by listener, obviously)

    def filter_fn(sample) -> bool:
        speaker = sample["speaker"]
        snr = sample["snr_db"]
        reference = sample["sentence"]
        identifying_tuple = tuple([speaker, snr, reference])

        if identifying_tuple not in seen_audios:
            seen_audios.add(identifying_tuple)
            return True
        else:
            return False

    grid_bc_unique_samples = grid_bc.filter(function=filter_fn)

    # check if audios are really unique
    audio_tuples = [tuple([o["speaker"], o["snr_db"], o["sentence"]]) for o in tqdm(grid_bc_unique_samples)]
    assert len(audio_tuples) == len(set(audio_tuples)), "audios appear multiple times!"

    grid_bc_unique_samples = even_out_snrs(grid_bc_unique_samples)
    assert has_even_snrs(grid_bc_unique_samples)

    if save_at.exists():
        shutil.rmtree(save_at)
    save_at.mkdir(exist_ok=True, parents=True)

    grid_bc_unique_samples.save_to_disk(save_at)