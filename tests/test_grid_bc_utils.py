from utils.grid_bc_utils import  parse_and_save_grid_bc
import pytest
from pathlib import Path
from datasets import Dataset

grid_bc_path = Path("datasets/GridIntelligibilityDatabase")
def test_parse_and_save_grid_bc():
    print("ExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpectsExpects")
    if not grid_bc_path.exists():
        pytest.fail("Expects '/mtec/db/speech/audio/grid/extra/GridIntelligibilityDatabase' to exist as 'GridIntelligibilityDatabase' in 'datasets/'")
    dataset = parse_and_save_grid_bc(grid_bc_folder=grid_bc_path,
                                     save_at=None,
                                     max_noise_folders=1,
                                     max_listener=1,
                                     max_files_per_listener=1)

    assert isinstance(dataset, Dataset)