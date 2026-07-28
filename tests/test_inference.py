from inference import inference
import pytest
from pathlib import Path

from utils.config_dataclasses import InferenceConfig
from utils.dataset_utils import get_dataset, apply_split
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
from utils.cuda_utils import select_device
import shutil
from utils.paths import GRID_FOLDER, TEST_FOLDER

@pytest.mark.parametrize("time_stamps", [False,True])
@pytest.mark.parametrize("extract_logprobs", [False,True])
def test_batch_inference_whisper(time_stamps, extract_logprobs):

    config = InferenceConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=TEST_FOLDER/"inference_test",
        dataset_type="grid",
        dataset_path=GRID_FOLDER,
        train_split=0,
        test_split=0,
        val_split=1,
        extract_logprobs=extract_logprobs,
        word_timestamps=time_stamps,
        beam_size=2
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_dataset(config.dataset_type, config.dataset_path, config.add_noise)
    dataset = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset, device)

    assert (TEST_FOLDER / "inference_test/data/s26_pwwizs.json").exists()

    if extract_logprobs:
        assert (TEST_FOLDER/"inference_test/logprobs/s26_pwwizs.pt").exists()

def test_batch_inference_parakeet():
    config = InferenceConfig(
        model="parakeet",
        model_type="ctc-0.6b",
        model_path=None,
        output_path=TEST_FOLDER/"inference_test",
        dataset_type="grid",
        dataset_path=GRID_FOLDER,
        train_split=0,
        test_split=0,
        val_split=1,
        extract_logprobs=False,
        word_timestamps=False,
        beam_size=2
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_dataset(config.dataset_type, config.dataset_path, config.add_noise)
    dataset = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset, device)

    assert (TEST_FOLDER / "inference_test/data/s26_pwwizs.json").exists()