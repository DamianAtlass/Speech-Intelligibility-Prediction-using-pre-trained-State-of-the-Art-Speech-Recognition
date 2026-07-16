from inference import inference
import pytest
from pathlib import Path

from utils.config_dataclasses import InferenceConfig
from utils.grid_utils import get_grid
from utils.dataset_utils import apply_split
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
import shutil

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()

@pytest.mark.parametrize("time_stamps", [False,True])
@pytest.mark.parametrize("extract_logprobs", [False,True])
def test_batch_inference(time_stamps, extract_logprobs):

    config = InferenceConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=test_folder/"inference_test",
        dataset_type="grid",
        dataset_path=test_folder.parent/"datasets/grid/",
        train_split=0,
        test_split=0,
        val_split=1,
        extract_logprobs=extract_logprobs,
        word_timestamps=time_stamps,
        beam_size=2
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_grid(config.dataset_path)
    dataset = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset, torch.device("cpu"))

    if extract_logprobs:
        assert (test_folder/"inference_test/logprobs/s7_sgib3s.pt").exists()

