from inference import inference
import pytest
from pathlib import Path

from utils.config_dataclasses import InferenceConfig
from utils.grid_utils import get_grid, apply_split
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
import shutil


@pytest.mark.parametrize("time_stamps", [False,True])
@pytest.mark.parametrize("extract_logprobs", [False,True])
def test_batch_inference(time_stamps, extract_logprobs):

    config = InferenceConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=Path("tests/inference_test"),
        dataset_path=Path("datasets/grid/"),
        train_split=1,
        test_split=1,
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

