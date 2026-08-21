from inference import create_filename
from utils.dataset_utils import get_dataset
from pathlib import Path
import pytest


@pytest.mark.parametrize(("dataset_type", "partial_sample_dict", "temp", "expected_result"), [
    ("grid",
    {"audio_path": "datasets/grid/downloaded_grid_files/audio/s13/s13/bbae1s.wav", "speaker": "13"},
    None,
    "s13_bbae1s"),
    ("grid_bc", {"audio_path": "datasets/GridIntelligibilityDatabase/BC2007wavs/BC2007/m8/3/s15_srba3a.wav",
        "speaker": "15",
        "listener": "3", "snr_db": -8},
    None,
    "SNRm8_l3_s15_srba3a"),
    ("grid_bc", {"audio_path": "datasets/GridIntelligibilityDatabase/BC2007wavs/BC2007/m8/3/s15_srba3a.wav",
        "speaker": "15",
        "listener": "3", "snr_db": -8},
    0.5,
    "SNRm8_l3_s15_temp0.5_srba3a"),
])
def test_create_filename(dataset_type, partial_sample_dict: dict, temp, expected_result):
    file_name = create_filename(dataset_type=dataset_type, sample=partial_sample_dict, temperature=temp)
    assert file_name == expected_result
