from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch

import shutil
from utils.new_config_dataclass import TrainingConfig, DatasetConfig, ModelConfig, DataSplitConfig
from utils.cuda_utils import select_device
from utils.dataset_utils import get_dataset_dict
from train_whisper import train_whisper
from utils.paths import TEST_FOLDER


def test_whisper_training():

    config = TrainingConfig(
        output_path=TEST_FOLDER/"training_test",
        task_type='training',
        data=DatasetConfig(
            train_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=10,
                                        noise=True, scaling=1),
            test_split=DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=5,
                                       noise=True, scaling=1)),
        debug=False,
        perform_training=True,
        learning_rate=1e-05,
        num_train_epochs=1,
        batch_size=16,
        save_and_eval_per_epoch=1,
        warmup_steps=500,
        model=ModelConfig(name="whisper", model_type="tiny", path=None)
    )

    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_dataset_dict(config.data)
    config.output_path.mkdir(exist_ok=config.debug)
    device = select_device()

    train_whisper(config, dataset, device)

    assert (config.output_path/"model.safetensors").is_file()