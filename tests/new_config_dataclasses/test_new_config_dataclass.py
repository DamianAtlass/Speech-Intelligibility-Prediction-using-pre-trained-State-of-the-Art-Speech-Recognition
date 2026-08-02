from utils.paths import TEST_FOLDER, _PROJECT_ROOT
from pathlib import Path, PosixPath
import pytest
from utils.new_config_dataclass import TrainingConfig, InferenceConfig, DatasetConfig, DataSplitConfig, BaseConfig, \
    load_config, from_dict, save_config, convert_old_config_into_new
import os
from utils.config_dataclasses import get_config as get_old_config
CONFIG_TEST_FOLDER = TEST_FOLDER/"new_config_dataclasses"


config_dict = [
{'batch_size': 16,
 'data': {
     'test_split': {
         'end': 4000, 'noise': True, 'path': 'grid/path', 'start': 100, 'type': 'grid'},
     'train_split':
         {'end': 0.3, 'noise': True, 'path': 'grid/path', 'start': 0.1, 'type': 'grid'},
     'val_split':
         {'end': 1.0, 'noise': True, 'path': 'grid_bc/path', 'start': 0.0, 'type': 'grid_bc'}},
 'dataset_scaling': 1,
 'debug': False,
 'learning_rate':1e-05,
 'num_train_epochs': 5,
 'output_path': 'output/path', # supposed to be a str
 'perform_training': True,
 'save_and_eval_per_epoch': 16,
 'task_type': 'training',
 'warmup_steps': 500}
    ,
{'beam_size': 5,
 'data':{
     'test_split': {
         'end': 4000, 'noise': True, 'path': 'grid/path', 'start': 100, 'type': 'grid'},
     'train_split':
         {'end': 0.3, 'noise': True, 'path': 'grid/path', 'start': 0.1, 'type': 'grid'},
     'val_split':
         {'end': 1.0, 'noise': True, 'path': 'grid_bc/path', 'start': 0.0, 'type': 'grid_bc'}},
 'extract_logprobs': False,
 'output_path': 'output/path', # supposed to be a str
 'task_type': 'inference',
 'word_timestamps': False,
 'dataset_scaling': 0.5,
 'debug': True,
 }
]

predefined_configs = [
TrainingConfig(
    output_path=PosixPath('output/path'),
    task_type='training',
    data=DatasetConfig(
        train_split=DataSplitConfig(type='grid', path=PosixPath('grid/path'),start=0.1, end=0.3, noise=True),
        test_split=DataSplitConfig(type='grid', path=PosixPath('grid/path'), start=100, end=4000, noise=True),
        val_split=DataSplitConfig(type='grid_bc', path=PosixPath('grid_bc/path'), start=0.0, end=1.0, noise=True)),
    debug=False,
    dataset_scaling=1,
    perform_training=True,
    learning_rate=1e-05,
    num_train_epochs=5,
    batch_size=16,
    save_and_eval_per_epoch=16,
    warmup_steps=500),
InferenceConfig(
    output_path=PosixPath('output/path'),
    task_type='inference',
    data=DatasetConfig(
        train_split=DataSplitConfig(type='grid', path=PosixPath('grid/path'),start=0.1, end=0.3, noise=True),
        test_split=DataSplitConfig(type='grid', path=PosixPath('grid/path'), start=100, end=4000, noise=True),
        val_split=DataSplitConfig(type='grid_bc', path=PosixPath('grid_bc/path'), start=0.0, end=1.0, noise=True)),
    debug=True,
    dataset_scaling=0.5,
    extract_logprobs=False,
    word_timestamps=False,
    beam_size=5)
]

@pytest.mark.parametrize(("file", "expected_class"), [
    ("training_config.yaml", TrainingConfig),
    ("inference_config.yaml", InferenceConfig)
])
def test_load_config_file(file, expected_class):
    path = CONFIG_TEST_FOLDER/file
    config = load_config(path)
    assert isinstance(config, expected_class)

@pytest.mark.parametrize(("config_dict", "config"), [
    (config_dict[0], predefined_configs[0]),
    (config_dict[1], predefined_configs[1])
])
def test_from_dict(config_dict, config: dict):
    config_from_dict: TrainingConfig|InferenceConfig = from_dict(BaseConfig, config_dict)
    # correct for shorter paths:
    #config_from_dict.output_path = _PROJECT_ROOT/config_from_dict.output_path

    assert config_from_dict == config


@pytest.mark.parametrize(("predefined_config"), [
    (predefined_configs[0]),
    (predefined_configs[1])
])
def test_correct_writing_of_file(predefined_config):
    path = CONFIG_TEST_FOLDER/"tmp_config.yaml"
    if path.exists():
        os.remove(path)
    save_config(predefined_config, path)

    loaded_config = load_config(path)
    assert loaded_config == predefined_config
    if path.exists():
        os.remove(path)

@pytest.mark.parametrize(("path_old_config", "expected_class"), [
    (TEST_FOLDER/"config_dataclasses/test_inference_config.ini", InferenceConfig),
    (TEST_FOLDER/"config_dataclasses/test_training_config.ini", TrainingConfig)
])
def test_convert_old_config_into_new(path_old_config: Path, expected_class):
    old_config = get_old_config(path_old_config)
    new_config = convert_old_config_into_new(old_config)
    assert isinstance(new_config, expected_class)

