import librosa
from datasets import Dataset, DatasetDict, load_from_disk
from utils.grid_utils import get_sentence_and_alignments, convert_short_name_to_ref
SAMPLE_RATE_DOWNLOADED_FILES = 25_000
WANTED_SAMPLE_RATE = 16_000

from tqdm import tqdm
from datasets import Audio
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import cast
import json

def convert_noise_level(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -int(s[1:])

def parse_and_save_grid_bc(grid_bc_folder: Path = Path.cwd() / "datasets" / "GridIntelligibilityDatabase",
                           save_at: Path = None,
                           #for debugging
                           max_noise_folders: int | None = None,
                           max_listener: int | None = None,
                           max_files_per_listener: int | None = None) -> Dataset:

    save_at = grid_bc_folder if save_at is None else save_at
    save_at = save_at / "saved_dataset"

    BC2007 = grid_bc_folder / "BC2007wavs" / "BC2007"
    listenerData_folder = grid_bc_folder/"listener_data"

    data = {
        "audio": [],
        "sample_rate": [],
        "sentence": [],
        "alignment": [],
        "audio_path": [],
        "align_path": [],
        "snr_db": [],
        "human_recognized_words": [],
    }
    project_root = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()

    counter_noise = 0
    for BC2007_noiseLevel in tqdm(BC2007.iterdir()):
        if counter_noise == max_noise_folders:
            break
        listener_counter = 0
        if not BC2007_noiseLevel.is_dir():
            continue

        for BC2007_noiseLevel_listener in BC2007_noiseLevel.iterdir():
            if BC2007_noiseLevel_listener.name =="18":
                continue # skip bc data for that listener at SNR 2 was missing
            if listener_counter == max_listener:
                break
            if not BC2007_noiseLevel_listener.is_dir():
                continue
            listenerData_json_path = listenerData_folder / BC2007_noiseLevel.name / f"{BC2007_noiseLevel_listener.name}.json"

            with open(listenerData_json_path) as f:
                l_data = json.load(f)
                tested_files = l_data["results"]["sent"]
                results = l_data["results"]["heard"]

            counter_audio_file =  0

            for audio_file in BC2007_noiseLevel_listener.iterdir():
                if counter_audio_file == max_files_per_listener:
                    break
                if not audio_file.name in tested_files:
                    continue

                speaker = audio_file.stem.split("_")[0]

                data["audio"].append(str(audio_file))  # will be resampled and converted to Audio() later, see below
                data["sample_rate"].append(WANTED_SAMPLE_RATE)
                data["audio_path"].append(str(audio_file.relative_to(project_root)))
                data["snr_db"].append(str(convert_noise_level(BC2007_noiseLevel.name)))

                alignment_file_name = audio_file.stem.split("_")[1] + ".align"
                alignment_file_path = grid_bc_folder / "word16kHz" / speaker / alignment_file_name
                data["align_path"].append(str(alignment_file_path.relative_to(project_root)))

                reference, alignments = get_sentence_and_alignments(alignment_file_path,
                                                                    original_sr=WANTED_SAMPLE_RATE,
                                                                    new_sr=WANTED_SAMPLE_RATE)
                assert len(reference.split(" ")) == 6, f"A GRID sentence has to be 6 words-long! ({reference})"
                data["sentence"].append(reference)
                data["alignment"].append(alignments)


                index = tested_files.index(audio_file.name)
                recognized_words = results[index]
                tmp = audio_file.stem.split("_")[1]
                assert reference == convert_short_name_to_ref(tmp, input_is_only_keywords=False)
                data["human_recognized_words"].append(convert_short_name_to_ref(recognized_words))

                counter_audio_file += 1
            listener_counter += 1
        counter_noise += 1


    dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=WANTED_SAMPLE_RATE))

    #n = dataset[0]["audio"]["array"]
    #import soundfile as sf
    #sf.write("file.wav", n, WANTED_SAMPLE_RATE)

    dataset.save_to_disk(save_at)

    return dataset



def get_grid_bc(dataset_directory: Path = Path.cwd()/"datasets"/"GridIntelligibilityDatabase") -> Dataset:
    try:
        return cast(Dataset, load_from_disk(dataset_directory/"saved_dataset"))

    except FileNotFoundError:
        dataset: Dataset = parse_and_save_grid_bc(dataset_directory)
        return dataset
