import subprocess
import tarfile
from shutil import rmtree

import librosa
from datasets import Dataset, DatasetDict, load_from_disk

SAMPLE_RATE_DOWNLOADED_FILES = 25_000
WANTED_SAMPLE_RATE = 16_000

import wave
import tqdm
from datasets import Audio
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import cast

def get_grid(dataset_directory: Path = Path.cwd()/"datasets"/"grid") -> Dataset:
    try:
        return cast(Dataset, load_from_disk(dataset_directory/"saved_dataset"))

    except FileNotFoundError:
        try:
            return parse_and_save_grid(dataset_directory)
        except FileNotFoundError:
            download_grid(dataset_directory)
            return parse_and_save_grid(dataset_directory)


def download_grid(dataset_directory: Path = Path.cwd()/"datasets"/"grid",
                  #for testing
                  max_speaker: int = 34) -> None:
    # Create directories
    download_folder = Path(dataset_directory, "downloaded_grid_files")

    #os.makedirs(Path(*parent, "/raw/audio"), exist_ok=True)
    (download_folder/"raw"/"audio").mkdir(exist_ok=True, parents=True)
    (download_folder/"raw"/"align").mkdir(exist_ok=True, parents=True)

    (download_folder/"audio").mkdir(exist_ok=True, parents=True)
    (download_folder/"align").mkdir(exist_ok=True, parents=True)

    extract_files = "y"

    for i in range(1, max_speaker + 1):
        logger.info(f"\n\n------------------------- Downloading {i}th speaker -------------------------\n\n")

        # Download audio files
        subprocess.run(["curl", f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{i}/audio/s{i}.tar", "-o",
                        (download_folder/"raw"/"audio"/f"s{i}.tar")])
        subprocess.run(["curl", f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{i}/align/s{i}.tar", "-o",
                        (download_folder/"raw"/"align"/f"s{i}.tar")])

        # Extract files if requested
        if extract_files.lower() == "y":
            with tarfile.open((download_folder/"raw"/"audio"/f"s{i}.tar"), 'r') as tar_ref:
                tar_ref.extractall((download_folder/"audio"/f"s{i}"))
            with tarfile.open((download_folder/"raw"/"align"/f"s{i}.tar"), 'r') as tar_ref:
                tar_ref.extractall((download_folder/"align"/f"s{i}"))

    rmtree((download_folder/"raw"))

    logger.info("Download completed.")

def get_sentence_and_alignments(align_file_path: Path) -> tuple[str, list[tuple[int, int, str]]]:
    with open(align_file_path, "r") as f:
        start = []
        end = []
        keyword_or_utterance = []

        for line in f.readlines():
            line = line.strip().split(" ")
            start.append(line[0])
            end.append(line[1])
            keyword_or_utterance.append(line[2])

    start = [str(int(s)/(SAMPLE_RATE_DOWNLOADED_FILES/WANTED_SAMPLE_RATE)) for s in start]
    end = [str(int(s)/(SAMPLE_RATE_DOWNLOADED_FILES/WANTED_SAMPLE_RATE)) for s in end]

    alignment = [(s,e,k) for s,e,k in zip(start, end, keyword_or_utterance)]

    sentence = [w for w in keyword_or_utterance if not w in ["sil", "sp"]]
    sentence = " ".join(sentence)
    return sentence, alignment

def process_audio(audio_file_path):
    array, sample_rate = librosa.load(audio_file_path, sr=WANTED_SAMPLE_RATE)
    return array

def get_sample_rate_of_mp3(file_path : str) -> float:
    with wave.open(file_path, "rb") as wave_file:
        return wave_file.getframerate()

def parse_and_save_grid(grid_folder: Path = Path.cwd() / "datasets" / "grid",
                        save_at: Path = None,
                        #for debugging
                        max_speaker: int | None = None,
                        max_files_per_speaker: int | None = None) -> Dataset:
    logger.info("Parse and save GRID.")
    download_folder = "downloaded_grid_files"
    align_folder = grid_folder / download_folder / "align"
    audio_folder = grid_folder / download_folder / "audio"

    data = {
        "audio": [],
        "sample_rate": [],
        "sentence": [],
        "alignment": [],
        "audio_path": [],
        "align_path": [],
    }
    counter_speaker = 0
    for speaker in tqdm.tqdm(audio_folder.iterdir()):
        speaker = speaker.name
        align = align_folder/speaker/"align"
        audio = audio_folder/speaker/speaker


        assert align.is_dir()
        assert audio.is_dir()

        counter_file = 0
        for file in audio.iterdir():
            audio_file_path = audio/file.name
            align_file_path = align/f"{file.stem}.align"

            if not audio_file_path.is_file():
                raise FileNotFoundError(f"Filepath {audio_file_path} does not exist")

            if not align_file_path.is_file():
                raise FileNotFoundError(f"Filepath {align_file_path} does not exist")

            data["audio"].append(str(audio_file_path)) #will be converted to Audio() later, see below

            reference, alignments = get_sentence_and_alignments(align_file_path)

            assert len(reference.split(" ")) == 6, f"A GRID sentence has to be 6 words-long! ({reference})"

            data["sentence"].append(reference)
            data["alignment"].append(alignments)
            data["audio_path"].append(str(audio_file_path.relative_to(Path.cwd())))
            data["align_path"].append(str(align_file_path.relative_to(Path.cwd())))
            data["sample_rate"].append(WANTED_SAMPLE_RATE)

            counter_file+=1
            if counter_file==max_files_per_speaker:
                break

        counter_speaker += 1
        if counter_speaker == max_speaker:
            break

    dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=WANTED_SAMPLE_RATE))

    assert len(dataset) == max_files_per_speaker * max_speaker
    if save_at is None:
        save_at = grid_folder / "saved_dataset"

    dataset.save_to_disk(save_at)

    return dataset

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


    if not (train_split==0 and test_split==0 and val_split==1):
        l = len(dataset)
        dataset = dataset.train_test_split(train_size=calculate_size(l, train_split),
                                           shuffle=True,
                                           seed=0)

        temp = dataset["test"].train_test_split(train_size=calculate_size(l, test_split),
                                                test_size=calculate_size(l, val_split),
                                                shuffle=True,
                                                seed=0)
        dataset_dict = DatasetDict({
            "train": dataset["train"],
            "test": temp["train"],
            "val": temp["test"],
        })
    else:
        dataset_dict = DatasetDict({
            "val": dataset,
        })

    if dataset_scaling != 1:
        for split in ["train", "test", "val"]:
            try:
                dataset_dict[split] = dataset_dict[split].select(range(
                    int(len(dataset_dict[split]) * dataset_scaling)
                ))
            except KeyError as e:
                pass

    return dataset_dict

def main():
    get_grid(Path.cwd().parent/"datasets"/"grid")

if __name__ == '__main__':
    main()