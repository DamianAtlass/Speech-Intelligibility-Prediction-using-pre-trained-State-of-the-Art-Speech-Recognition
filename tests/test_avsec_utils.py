from utils.avsec_utils import convert_video_to_audio_moviepy, parse_and_save_avsec2_enhanced
from pathlib import Path
from utils.paths import TEST_FOLDER, PROJECT_ROOT
import soundfile as sf
import numpy as np
import os
import shutil


def test_convert_video_to_audio_moviepy():
    video_file_path = TEST_FOLDER/ "ID1581.mp4"
    assert video_file_path.exists(), "File doesnt exist: {video_file_path}"
    audio_file_path = TEST_FOLDER / "ID1581.wav"
    convert_video_to_audio_moviepy(video_file_path, audio_file_path)


    data, samplerate = sf.read(audio_file_path)
    assert samplerate == 16_000
    assert isinstance(data, np.ndarray)
    os.remove(audio_file_path)


def test_parse_and_save_avsec2_enhanced():
    test_avsec2 = TEST_FOLDER /"avsec2"
    dataset_dir = test_avsec2 /"saved_dataset"
    if test_avsec2.exists():
        shutil.rmtree(test_avsec2)

    test_avsec2.mkdir(parents=True, exist_ok=True)


    dataset = parse_and_save_avsec2_enhanced(save_at=dataset_dir, wav_dir=test_avsec2, max_samples=10)
    num_files = 0
    for f in test_avsec2.iterdir():
        if f.is_file(): num_files += 1
    assert num_files == len(set(dataset["audio_path"]))
    assert num_files == len(set(dataset["sample_ID"]))

    shutil.rmtree(test_avsec2)




