from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch

import shutil
from utils.new_config_dataclass import TrainingConfig, DatasetConfig, ModelConfig, DataSplitConfig
from utils.cuda_utils import select_device
from utils.dataset_utils import get_dataset_dict
from train_parakeet import train_parakeet
from utils.paths import TEST_FOLDER
import pytest
import os


def test_parakeet_training():
    if os.getenv("ENOUGH_RAM_FOR_PARAKEET_TRAINNIG") == "False":
        pytest.skip()

    config = TrainingConfig(
        output_path=TEST_FOLDER / "training_test",
        task_type='training',
        data=DatasetConfig(
            train_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=10,
                                        noise=True, scaling=1),
            val_split=DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=5,
                                      noise=False, scaling=1)),
        debug=False,
        perform_training=True,
        learning_rate=1234,
        num_train_epochs=1,
        batch_size=16,
        save_and_eval_steps=1234,
        warmup_steps=1234,
        model=ModelConfig(name="parakeet", model_type="ctc-0.6b", path=None)
    )

    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_dataset_dict(config.data)
    config.output_path.mkdir(exist_ok=config.debug)
    device = select_device()

    train_parakeet(config, dataset, device)

    assert (config.output_path/"checkpoint.nemo").is_file()