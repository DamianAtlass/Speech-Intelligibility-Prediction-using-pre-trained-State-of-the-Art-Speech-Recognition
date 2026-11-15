from utils.stoi_utils import calculate_stoi
from utils.dataset_utils import get_dataset
from pathlib import Path
from utils.new_config_dataclass import DataSplitConfig
from utils.dataset_utils import get_clean_grid_path_from_bc_path

def test_calculate_stoi():
    sample = get_dataset(DataSplitConfig(dataset_type='grid_bc', path=Path("datasets/grid_bc_unique_audios"), start=0, end=1.,noise=False, scaling=1.0))[0]
    r = calculate_stoi(clean_path=get_clean_grid_path_from_bc_path(sample["audio_path"]), noisy_path=sample["audio_path"])
    assert 0.86 == round(r,2)

