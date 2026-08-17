from inference import inference
import pytest
from pathlib import Path

from utils.config_dataclasses import Old_InferenceConfig
from utils.new_config_dataclass import InferenceConfig, ModelConfig, DataSplitConfig, DatasetConfig
from utils.dataset_utils import get_dataset_dict, get_dataset, apply_split
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
from utils.cuda_utils import select_device
import shutil
from utils.paths import GRID_FOLDER, TEST_FOLDER

@pytest.mark.parametrize("time_stamps", [False,True])
@pytest.mark.parametrize("extract_logprobs", [False,True])
def test_batch_inference_whisper(time_stamps, extract_logprobs):
    if time_stamps and not extract_logprobs:
        pytest.skip("unvalid constallation") #todo fix later

    config = InferenceConfig(
        output_path=TEST_FOLDER / "inference_test",
        task_type='inference',
        data=DatasetConfig(
            val_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1)),
        debug=False,
        extract_logprobs=extract_logprobs,
        word_timestamps=time_stamps,
        beam_size=2,
        model=ModelConfig(name="whisper", model_type="tiny", path=None),
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset_dict, device)

    assert (TEST_FOLDER / "inference_test/data/s26_pwwizs.json").exists()

    if extract_logprobs:
        assert (TEST_FOLDER/"inference_test/logprobs/s26_pwwizs.pt").exists()

def test_batch_inference_parakeet():
    config = InferenceConfig(
        output_path=TEST_FOLDER / "inference_test",
        task_type='inference',
        data=DatasetConfig(
            val_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1)),
        debug=False,
        extract_logprobs=False,
        word_timestamps=False, # has no effect here
        beam_size=2,
        model=ModelConfig(name="parakeet", model_type="ctc-0.6b", path=None),
    )

    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset_dict, device)

    assert (TEST_FOLDER / "inference_test/data/s26_pwwizs.json").exists()