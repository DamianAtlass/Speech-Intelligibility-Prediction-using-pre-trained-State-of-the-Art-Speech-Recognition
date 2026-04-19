import subprocess
import tarfile
from shutil import rmtree

import librosa
from datasets import Dataset, DatasetDict, load_from_disk

SAMPLE_RATE = 16000
import wave
import tqdm
from datasets import Audio
from pathlib import Path

SAMPLES_PER_SPEAKER = 1000 # samples a speaker recorded

def get_grid(dataset_directory: Path = Path.cwd()/"datasets"/"grid"):
    try:
        return load_from_disk(dataset_directory/"saved_datasetdict")

    except FileNotFoundError:
        try:
            return parse_and_save_grid()
        except FileNotFoundError:
            download_grid()
            return parse_and_save_grid()


def download_grid(dataset_directory: Path = Path.cwd()/"datasets"/"grid") -> None:
    # Create directories
    parent = Path(dataset_directory, "downloaded_grid_files")

    #os.makedirs(Path(*parent, "/raw/audio"), exist_ok=True)
    (parent/"raw"/"audio").mkdir(exist_ok=True, parents=True)
    (parent/"raw"/"align").mkdir(exist_ok=True, parents=True)

    (parent/"audio").mkdir(exist_ok=True, parents=True)
    (parent/"align").mkdir(exist_ok=True, parents=True)

    extract_files = "y"

    for i in range(1, 34 + 1):
        print(f"\n\n------------------------- Downloading {i}th speaker -------------------------\n\n")

        # Download audio files
        subprocess.run(["curl", f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{i}/audio/s{i}.tar", "-o",
                        (parent/"raw"/"audio"/f"s{i}.tar")])
        subprocess.run(["curl", f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{i}/align/s{i}.tar", "-o",
                        (parent/"raw"/"align"/f"s{i}.tar")])

        # Extract files if requested
        if extract_files.lower() == "y":
            with tarfile.open((parent/"raw"/"audio"/f"s{i}.tar"), 'r') as tar_ref:
                tar_ref.extractall((parent/"audio"/f"s{i}"))
            with tarfile.open((parent/"raw"/"align"/f"s{i}.tar"), 'r') as tar_ref:
                tar_ref.extractall((parent/"align"/f"s{i}"))

    rmtree((parent/"raw"))

    print("Download completed.")

def get_sentence_and_alignments(align_file_path: Path):
    with open(align_file_path, "r") as f:
        start = []
        end = []
        keyword_or_utterance = []

        for line in f.readlines():
            line = line.strip().split(" ")
            start.append(line[0])
            end.append(line[1])
            keyword_or_utterance.append(line[2])

    alignment = [(s,e,k) for s,e,k in zip(start,end,keyword_or_utterance)]

    sentence = [w for w in keyword_or_utterance if not w in ["sil", "sp"]]
    sentence = " ".join(sentence)
    return sentence, alignment

def process_audio(audio_file_path):
    array, sample_rate = librosa.load(audio_file_path, sr=SAMPLE_RATE)
    return array

def get_sample_rate_of_mp3(file_path : str) -> float:
    with wave.open(file_path, "rb") as wave_file:
        return wave_file.getframerate()

def parse_and_save_grid(grid_folder = Path.cwd() / "datasets" / "grid" ) -> DatasetDict:
    download_folder = "downloaded_grid_files"
    align_folder = grid_folder / download_folder / "align"
    audio_folder = grid_folder / download_folder / "audio"

    data = {
        "audio": [],
        "sentence": [],
        "alignment": [],
        "audio_path": [],
        "align_path": [],
    }
    for speaker in tqdm.tqdm(audio_folder.iterdir()):
        speaker = speaker.name
        align = align_folder/speaker/"align"
        audio = audio_folder/speaker/speaker

        assert align.is_dir()
        assert audio.is_dir()

        for file in audio.iterdir():
            file_name_without_ending = file.name.split(".")[0]
            audio_file_path = audio/file.name
            align_file_path = align/f"{file_name_without_ending}.align"

            if not audio_file_path.is_file():
                raise FileNotFoundError(f"Filepath {audio_file_path} does not exist")

            if not align_file_path.is_file():
                raise FileNotFoundError(f"Filepath {align_file_path} does not exist")

            data["audio"].append(str(audio_file_path)) #will be converted to Audio() later, see below

            reference, alignments = get_sentence_and_alignments(align_file_path)

            assert len(reference.split(" ")) == 6, f"A GRID sentence has to be 6 word-long! ({reference})"

            data["sentence"].append(reference)
            data["alignment"].append(alignments)
            data["audio_path"].append(str(audio_file_path))
            data["align_path"].append(str(align_file_path))

    dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    dataset = dataset.train_test_split(test_size=0.3, seed=0)
    temp = dataset["test"].train_test_split(test_size=0.5, seed=0)

    dataset_dict = DatasetDict({
        "train": dataset["train"],
        "test": temp["test"],
        "val": temp["train"],
    })

    assert len(dataset_dict["train"]) + len(dataset_dict["test"]) + len(dataset_dict["val"]) == SAMPLES_PER_SPEAKER * 34, "Unexpected number of samples" #todo different splits

    data_path = Path.cwd()/"datasets"/"grid"/"saved_datasetdict"
    dataset_dict.save_to_disk(data_path)

    return dataset_dict


def main():
    pass

if __name__ == '__main__':
    main()