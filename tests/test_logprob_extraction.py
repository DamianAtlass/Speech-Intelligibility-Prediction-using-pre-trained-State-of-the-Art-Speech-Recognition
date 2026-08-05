from inference import inference
import pytest

from utils.new_config_dataclass import DataSplitConfig, DatasetConfig, InferenceConfig, ModelConfig
from utils.dataset_utils import get_dataset_dict
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
import shutil
import json
from utils.paths import TEST_FOLDER

def test_logprob_extraction():
    config = InferenceConfig(
        output_path=TEST_FOLDER / "inference_test",
        task_type='inference',
        data=DatasetConfig(
            val_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1)),
        debug=False,
        extract_logprobs=True,
        word_timestamps=False, # has no effect here
        beam_size=5,
        model=ModelConfig(name="whisper", model_type="tiny", path=None),
    )

    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset_dict, torch.device("cuda"))
    print((TEST_FOLDER/"inference_test"/"data"/"s26_pwwizs.json").exists())
    with open(str(TEST_FOLDER/"inference_test"/"data"/"s26_pwwizs.json")) as f:
        data = json.load(f)
    tensor = torch.load(TEST_FOLDER/"inference_test"/"logprobs"/"s26_pwwizs.pt")
    print()

    tokens = data["prediction_result"]["decoded_tokens_with_timestamps"]