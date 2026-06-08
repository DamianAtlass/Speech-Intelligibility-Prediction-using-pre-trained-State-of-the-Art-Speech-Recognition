from utils.grid_bc_utils import  parse_and_save_grid_bc, get_grid_bc
import pytest
from pathlib import Path
from datasets import Dataset
import shutil
test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()
real_grid_bc_folder = test_folder.parent / "datasets" / "GridIntelligibilityDatabase"
test_grid_bc_folder = test_folder / "grid_bc"

def test_parse_and_save_grid_bc():
    if not real_grid_bc_folder.exists():
        pytest.fail("Expects '/mtec/db/speech/audio/grid/extra/GridIntelligibilityDatabase' to exist as 'GridIntelligibilityDatabase' in 'datasets/'")
    if test_grid_bc_folder.exists():
        shutil.rmtree(test_grid_bc_folder)

    max_noise_folders = 3
    max_listener = 2
    max_files_per_listener = 4

    dataset = parse_and_save_grid_bc(grid_bc_folder=real_grid_bc_folder,
                                     save_at=test_grid_bc_folder,
                                     max_noise_folders=max_noise_folders,
                                     max_listener=max_listener,
                                     max_files_per_listener=max_files_per_listener)

    assert len(dataset) == max_noise_folders * max_listener  * max_files_per_listener


    assert isinstance(dataset, Dataset)
    assert test_grid_bc_folder.exists()

def test_get_grid_bc():

    dataset = get_grid_bc(test_grid_bc_folder)
    assert isinstance(dataset, Dataset)
# whole actual grid_bc should be 22800 long (12×19×100)