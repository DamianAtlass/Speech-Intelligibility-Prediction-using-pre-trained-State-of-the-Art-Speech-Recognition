from inference import create_filename
from utils.dataset_utils import get_dataset
from pathlib import Path
import pytest


@pytest.mark.parametrize(("dataset_type", "partial_sample_dict", "run", "expected_result"), [
    ("grid",
    {"audio_path": "datasets/grid/downloaded_grid_files/audio/s13/s13/bbae1s.wav", "speaker": "13"},
    None,
    "s13_bbae1s"),
    ("grid_bc", {"audio_path": "datasets/GridIntelligibilityDatabase/BC2007wavs/BC2007/m8/3/s15_srba3a.wav",
        "speaker": "15",
        "listener": "3", "snr_db": -8},
    1,
    "SNRm8_l3_s15_run1_srba3a"),
    ("grid_bc", {"audio_path": "datasets/GridIntelligibilityDatabase/BC2007wavs/BC2007/m8/3/s15_srba3a.wav",
        "speaker": "15",
        "listener": "3", "snr_db": -8},
    2,
    "SNRm8_l3_s15_run2_srba3a"),
])
def test_create_filename(dataset_type, partial_sample_dict: dict, run, expected_result):
    file_name = create_filename(dataset_type=dataset_type, sample=partial_sample_dict, run=run)
    assert file_name == expected_result
