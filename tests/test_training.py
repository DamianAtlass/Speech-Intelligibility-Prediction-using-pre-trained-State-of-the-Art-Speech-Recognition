from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch

from pathlib import Path
import shutil
from utils.config_dataclasses import TrainingConfig
from utils.cuda_utils import select_device
from utils.grid_utils import get_grid, apply_split
from train_whisper import train_whisper


def test_whisper_training():

    config = TrainingConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=Path("tests/training_test"),
        dataset_path=Path("datasets/grid/"),
        train_split=10,
        test_split=5,
        val_split=1,
        perform_training=True,
        learning_rate=1e-5,
        num_train_epochs=1,
        warmup_steps = 1
    )

    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_grid(config.dataset_path)
    dataset = apply_split(dataset, config)
    config.output_path.mkdir(exist_ok=config.debug)
    device = select_device()

    train_whisper(config, dataset, device)

    assert (config.output_path/"model.safetensors").is_file()