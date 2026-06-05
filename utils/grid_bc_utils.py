
import librosa
from datasets import Dataset, DatasetDict, load_from_disk

SAMPLE_RATE_DOWNLOADED_FILES = 25_000
WANTED_SAMPLE_RATE = 16_000

from tqdm import tqdm
from datasets import Audio
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import cast

def parse_and_save_grid_bc(grid_bc_folder: Path = Path.cwd() / "datasets" / "GridIntelligibilityDatabase",
                           save_at: Path = None,
                           #for debugging
                           max_noise_folders: int | None = None,
                           max_listener: int | None = None,
                           max_files_per_listener: int | None = None) -> Dataset:

    noise_folder = grid_bc_folder / "BC2007wavs" / "BC2007"
    if save_at is None:
        save_at = grid_bc_folder / "saved_dataset"

    data = {
        "audio": [],
        #"sample_rate": [],
        #"sentence": [],
        #"alignment": [],
        #"audio_path": [],
        #"align_path": [],
    }
    counter_noise = 0
    for noise_level_folder in tqdm(noise_folder.iterdir()):
        listener_counter = 0
        if not noise_level_folder.is_dir():
            continue

        for listener_folder in noise_level_folder.iterdir():
            if not listener_folder.is_dir():
                continue

            counter_audio_file =  0
            for audio_file in listener_folder.iterdir():
                print(audio_file)
                data["audio"].append(str(audio_file))  # will be converted to Audio() later, see below

                counter_audio_file += 1
                if counter_audio_file == max_files_per_listener:
                    break

            listener_counter += 1
            if listener_counter == max_files_per_listener:
                break

        counter_noise += 1
        if counter_noise == max_files_per_listener:
            break

    dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=WANTED_SAMPLE_RATE))

    #n = dataset[0]["audio"]["array"]
    #import soundfile as sf
    #sf.write("file.wav", n, WANTED_SAMPLE_RATE)


    assert len(dataset) == (max_files_per_listener or 120) * (max_listener or 20) * (max_noise_folders or 12)

    if save_at is None:
        save_at = grid_bc_folder / "saved_dataset"

    dataset.save_to_disk(save_at)

    return dataset



def get_grid_bc(dataset_directory: Path = Path("datasets/GridIntelligibilityDatabase")) -> Dataset:
    try:
        return cast(Dataset, load_from_disk(dataset_directory/"saved_dataset"))

    except FileNotFoundError:
        dataset: Dataset = parse_and_save_grid_bc(dataset_directory)
        return dataset
