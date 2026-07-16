from inference import inference
import pytest
from pathlib import Path

from utils.config_dataclasses import InferenceConfig
from utils.dataset_utils import get_dataset, apply_split
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
import shutil
import json


test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()


def test_logprob_extraction():

    config = InferenceConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=test_folder/"logprob_extraction",
        dataset_type="grid",
        dataset_path=test_folder.parent/"datasets/grid/",
        train_split=0,
        test_split=0,
        val_split=1,
        extract_logprobs=True,
        word_timestamps=False,
        beam_size=5
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_dataset("grid")
    dataset = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset, torch.device("cuda"))
    with open(str(test_folder/"logprob_extraction"/"data"/"s9_lrakzp.json")) as f:
        data = json.load(f)
    tensor = torch.load(test_folder/"logprob_extraction"/"logprobs"/"s9_lrakzp.pt")
    print()

    tokens = data["prediction_result"]["decoded_tokens_with_timestamps"]