from utils.grid_bc_utils import  parse_and_save_grid_bc, get_grid_bc
import pytest
from pathlib import Path
from datasets import Dataset
import shutil
from utils.paths import BC_FOLDER, TEST_BC_FOLDER

def test_parse_and_save_grid_bc():
    if not BC_FOLDER.exists():
        pytest.fail("Expects '/mtec/db/speech/audio/grid/extra/GridIntelligibilityDatabase' to exist as 'GridIntelligibilityDatabase' in 'datasets/'")
    if TEST_BC_FOLDER.exists():
        shutil.rmtree(TEST_BC_FOLDER)

    max_noise_folders = 3
    max_listener = 2
    max_files_per_listener = 4

    dataset = parse_and_save_grid_bc(grid_bc_folder=BC_FOLDER,
                                     save_at=TEST_BC_FOLDER,
                                     max_noise_folders=max_noise_folders,
                                     max_listener=max_listener,
                                     max_files_per_listener=max_files_per_listener)

    assert len(dataset) == max_noise_folders * max_listener  * max_files_per_listener


    assert isinstance(dataset, Dataset)
    assert TEST_BC_FOLDER.exists()

def test_get_grid_bc():

    dataset = get_grid_bc(TEST_BC_FOLDER)
    assert isinstance(dataset, Dataset)
# whole actual grid_bc should be 22800 long (12×19×100)