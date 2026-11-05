from pathlib import Path
import csv
data_path = Path("/home/damian/Downloads/AVSEC")
from datasets import Dataset, load_from_disk, Audio

from collections import Counter
from moviepy import VideoFileClip
from utils.paths import AVSEC2_ENHANCED_SAMPLES, AVSEC2_LTR, PROJECT_ROOT
from tqdm import tqdm
from typing import cast

WANTED_SAMPLE_RATE = 16_000

def parse_and_save_avsec2_enhanced(
        save_at: Path = AVSEC2_LTR / "saved_dataset",
        #for debugging
        wav_dir=AVSEC2_LTR / "avsec2_enhanced_audio_files",
    max_samples = None,):

    with open(AVSEC2_LTR/"avse2_scenes_metadata.csv", "r") as f:
        scenes = {}
        data = csv.DictReader(f)
        for row in data:
            scenes[row["scene"]] = row

    with open(AVSEC2_LTR/"avse2_scene_to_sampleID_pairs.csv", "r") as f:
        sampleIDs_to_scenes = {}
        data = csv.DictReader(f)
        for row in data:
            sampleIDs_to_scenes[row["sample_ID"]] = row["scene"]

    wav_dir.mkdir(parents=True, exist_ok=True)
    responses = []

    with open(AVSEC2_LTR/"avse2_responses.csv", "r") as f: #
        data = csv.DictReader(f)
        for i, row in enumerate(tqdm(data)):
            if i == max_samples:
                break
            sample_id = row["sample_ID"]

            scene_id = sampleIDs_to_scenes[sample_id]
            row["scene"] = scene_id
            scene = scenes[scene_id]

            row.update({k:scene[k] for k in scene.keys()})
            alt = ["alt_1", "alt_2", "alt_3", "alt_4", "alt_5"]
            d = []
            for a in alt:
                d.append(row.pop(a))
            row["alt"] = d
            video_file_name = f"{sample_id}.mp4"
            video_file_path = AVSEC2_ENHANCED_SAMPLES / video_file_name

            if video_file_path.exists():
                audio_file_path: Path = wav_dir / (f"{sample_id}.wav")
                if not audio_file_path.exists():
                    convert_video_to_audio_moviepy(video_file_path, audio_file_path)

                row["audio_path"] = str(audio_file_path)
                row["audio"] = str(audio_file_path)  # transform into lazy row later
            else:
                raise FileNotFoundError(video_file_path.relative_to(PROJECT_ROOT))

            responses.append(row)

    dataset = {k: [o[k] for o in responses] for k in responses[0].keys()}
    print(Counter(dataset["snr"]))
    dataset = Dataset.from_dict(dataset).cast_column("audio", Audio(sampling_rate=WANTED_SAMPLE_RATE))
    dataset = dataset.rename_column("snr", "snr_db")
    dataset = dataset.rename_column("transcript", "reference")
    save_at.mkdir(parents=False, exist_ok=True)
    dataset.save_to_disk(save_at)
    return dataset


def get_avsec2_enhanced() -> Dataset:
    try:
        return cast(Dataset, load_from_disk(AVSEC2_LTR / "saved_dataset"))
    except FileNotFoundError:
        return parse_and_save_avsec2_enhanced()

def convert_video_to_audio_moviepy(video_file_path: Path,
                                   audio_file_path: Path):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood
    # https://thepythoncode.com/article/extract-audio-from-video-in-python
    """
    video_file_path = Path(video_file_path)
    clip = VideoFileClip(video_file_path)
    clip.audio.write_audiofile(audio_file_path, ffmpeg_params=["-ac", "1"], fps=WANTED_SAMPLE_RATE) #set audio channels to 1
    assert audio_file_path.exists()

