from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch

import shutil
from utils.config_dataclasses import TrainingConfig
from utils.cuda_utils import select_device
from utils.dataset_utils import get_dataset, apply_split
from train_whisper import train_whisper
from utils.paths import TEST_FOLDER


def test_whisper_training():

    config = TrainingConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=TEST_FOLDER/"training_test",
        dataset_type="grid",
        dataset_path=None,
        add_noise=False,
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

    dataset = get_dataset(config.dataset_type, add_noise=config.add_noise)
    dataset = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    config.output_path.mkdir(exist_ok=config.debug)
    device = select_device()

    train_whisper(config, dataset, device)

    assert (config.output_path/"model.safetensors").is_file()