import subprocess
import tarfile
from shutil import rmtree

import librosa
from datasets import Dataset, DatasetDict, load_from_disk

SAMPLE_RATE_DOWNLOADED_FILES = 25_000
WANTED_SAMPLE_RATE = 16_000

import wave
from tqdm import tqdm
from datasets import Audio
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import cast

def get_grid(grid_dir: Path = Path.cwd() / "datasets" / "grid") -> Dataset:
    try:
        return cast(Dataset, load_from_disk(grid_dir / "saved_dataset"))

    except FileNotFoundError:
        try:
            return parse_and_save_grid(grid_dir)
        except FileNotFoundError:
            download_grid(grid_dir)
            return parse_and_save_grid(grid_dir)


def download_grid(grid_folder: Path = Path.cwd() / "datasets"/"grid",
                  #for testing
                  max_speaker: int = 34) -> Path:
    download_folder = grid_folder / "downloaded_grid_files"
    #os.makedirs(Path(*parent, "/raw/audio"), exist_ok=True)
    (download_folder / "raw" / "audio").mkdir(exist_ok=True, parents=True)
    (download_folder / "raw" / "align").mkdir(exist_ok=True, parents=True)

    (download_folder / "audio").mkdir(exist_ok=True, parents=True)
    (download_folder / "align").mkdir(exist_ok=True, parents=True)

    extract_files = "y"

    for i in range(1, max_speaker + 1):
        logger.info(f"\n\n------------------------- Downloading {i}th speaker -------------------------\n\n")

        # Download audio files
        subprocess.run(["curl", f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{i}/audio/s{i}.tar", "-o",
                        (download_folder / "raw" / "audio" / f"s{i}.tar")])
        subprocess.run(["curl", f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{i}/align/s{i}.tar", "-o",
                        (download_folder / "raw" / "align" / f"s{i}.tar")])

        # Extract files if requested
        if extract_files.lower() == "y":
            with tarfile.open((download_folder / "raw" / "audio" / f"s{i}.tar"), 'r') as tar_ref:
                tar_ref.extractall((download_folder / "audio" / f"s{i}"))
            with tarfile.open((download_folder / "raw" / "align" / f"s{i}.tar"), 'r') as tar_ref:
                tar_ref.extractall((download_folder / "align" / f"s{i}"))

    rmtree((download_folder / "raw"))

    logger.info("Download completed.")
    return download_folder


def get_sentence_and_alignments(align_file_path: Path, original_sr: int, new_sr: int) -> tuple[str, list[tuple[int, int, str]]]:
    """
    align_file_path: Path, path to the alignment file
    original_sr: int, the ole sample rate of the file
    new_sr: int, the new sample rate of the file

    Returns:
        a tuple of alignments with adjusted timestamps
    """
    with open(align_file_path, "r") as f:
        start = []
        end = []
        keyword_or_utterance = []

        for line in f.readlines():
            line = line.strip().split(" ")
            start.append(line[0])
            end.append(line[1])
            keyword_or_utterance.append(line[2])

    start = [str(int(int(s)/(original_sr/new_sr))) for s in start]
    end = [str(int(int(s)/(original_sr/new_sr))) for s in end]

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
                        #for debugging
                        max_speaker: int | None = None,
                        max_files_per_speaker: int | None = None) -> Dataset:
    save_at = grid_folder / "saved_dataset"

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
    for speaker in tqdm(audio_folder.iterdir(), total=max_speaker or 34):
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

            data["audio"].append(str(audio_file_path)) #will be resampled and converted to Audio() later, see below

            reference, alignments = get_sentence_and_alignments(align_file_path,
                                                                original_sr=SAMPLE_RATE_DOWNLOADED_FILES,
                                                                new_sr=WANTED_SAMPLE_RATE)
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

    l = len(dataset)
    train_size = calculate_size(l, train_split)
    test_size = calculate_size(l, test_split)
    val_size = calculate_size(l, val_split)

    d = {}
    full_split = False
    for label, v in zip(["test", "train", "val"], [test_size, train_size, val_size]):
        if v == l:
            d[label] = dataset
            full_split = True

    if not full_split:
        if train_size > 0:
            split = dataset.train_test_split(
                train_size=train_size,
                shuffle=True,
                seed=0,
            )

            d["train"] = split["train"]
            remainder = split["test"]
        else:
            remainder = dataset

        if test_size > 0 and val_size > 0:
            temp = remainder.train_test_split(
                train_size=test_size,
                test_size=val_size,
                shuffle=True,
                seed=0,
            )
            d["test"] = temp["train"]
            d["val"] = temp["test"]

        elif test_size > 0:
            temp = remainder.train_test_split(
                test_size=test_size,
                shuffle=True,
                seed=0,
            )
            d["test"] = temp["test"]

        elif val_size > 0:
            temp = remainder.train_test_split(
                test_size=val_size,
                shuffle=True,
                seed=0,
            )
            d["val"] = temp["test"]

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


def convert_short_name_to_ref(string: str, input_is_only_keywords=True) -> str:
    """
    Converts the short names of grid files to the corresponding sentences based on their combination of letters.

    string: str, the input string. Must be either 3 or 6 letters long.
    input_is_only_keywords: bool, set this to ignore non-keywords
    """
    assert len(string)==6 or len(string)==3

    grid_vocab = [{'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'},
                  {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'},
                  {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'},
                  {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j',
                   'k': 'k', 'l': 'l', \
                   'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u', 'v': 'v',
                   'x': 'x', 'y': 'y', 'z': 'z'},
                  {'z': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                   '7': 'seven', '8': 'eight', '9': 'nine'},
                  {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}]
    keywords_index = [1, 3, 4]

    tmp = []
    if input_is_only_keywords:
        grid_vocab = [grid_vocab[i] for i in keywords_index]

    for letter, vocab in zip(string, grid_vocab):
        tmp.append(vocab[letter])

    return " ".join(tmp)


def main():
    get_grid(Path.cwd().parent/"datasets"/"grid")

if __name__ == '__main__':
    main()