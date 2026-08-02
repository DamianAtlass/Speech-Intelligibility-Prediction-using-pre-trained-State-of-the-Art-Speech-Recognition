from dataclasses import dataclass, fields
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass, asdict
import yaml
from typing import Any
from utils.config_dataclasses import InferenceConfig as OldInferenceConfig, TrainingConfig as OldTrainingConfig

from utils.paths import _PROJECT_ROOT


#from utils.config_dataclasses import TrainingConfig, InferenceConfig, get_config


def from_dict(cls, data: dict) -> Any:
    if cls is BaseConfig and "task_type" in data.keys():
        cls = InferenceConfig if data["task_type"] == "inference" else TrainingConfig
    if not is_dataclass(cls):
        return data

    kwargs = {}
    for field in fields(cls):
        value = data.get(field.name)

        if is_dataclass(field.type):
            value = from_dict(field.type, value)

        kwargs[field.name] = value
    return cls(**kwargs)

def config_to_dict(obj) -> dict | Any:
    if is_dataclass(obj):
        return {k: config_to_dict(v) for k, v in asdict(obj).items()}
    elif issubclass(type(obj), Path):
        obj = obj.relative_to(_PROJECT_ROOT)
        return str(obj)
    elif isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [config_to_dict(v) for v in obj]
    else:
        return obj

@dataclass(kw_only=True)
class DataSplitConfig:
    type: str
    path: Path
    start: float|int
    end: float|int
    noise: bool

    def __post_init__(self):
        self.path = _PROJECT_ROOT/self.path

@dataclass(kw_only=True)
class DatasetConfig:
    train_split: DataSplitConfig
    test_split: DataSplitConfig
    val_split: DataSplitConfig

@dataclass(kw_only=True)
class BaseConfig:
    output_path: Path
    task_type: str
    data: DatasetConfig

    #for debugging
    debug: bool = False
    dataset_scaling: float = 1

    def __post_init__(self):
        self.output_path = _PROJECT_ROOT/self.output_path

@dataclass(kw_only=True)
class InferenceConfig(BaseConfig):
    extract_logprobs: bool = True
    word_timestamps: bool = True
    beam_size: int = 5

    def __post_init__(self):
        super().__post_init__()

@dataclass(kw_only=True)
class TrainingConfig(BaseConfig):
    perform_training: bool
    learning_rate: float
    num_train_epochs: int
    batch_size: int = 16
    save_and_eval_per_epoch: int = 6
    warmup_steps: int

    def __post_init__(self):
        super().__post_init__()
        self.learning_rate = float(self.learning_rate)

#def convert_old_to_new_config(old_config_path) -> BaseConfig:
 #   config: TrainingConfig | InferenceConfig = get_config(Path(old_config_path))

def load_config(path: str|Path) -> TrainingConfig|InferenceConfig:
    path = Path.cwd() / path

    with open(path, 'r') as f:
        data: dict = yaml.load(f, Loader=yaml.SafeLoader)

    config = from_dict(BaseConfig, data)
    return config

def save_config(config: TrainingConfig | InferenceConfig, path: Path) -> None:
    if path.exists(): raise FileExistsError()
    config_dict = config_to_dict(config)

    with open(str(path), 'w') as file:
        yaml.dump(config_dict, file, indent=4, sort_keys=False)

def convert_old_config_into_new(config: OldTrainingConfig | OldInferenceConfig) -> TrainingConfig|InferenceConfig:

    if isinstance(config, OldTrainingConfig):
        config_dict = {
             'data': {
                 'test_split':
                     {'end': config.test_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'type': config.dataset_type},
                 'train_split':
                     {'end': config.train_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'type': config.dataset_type},
                 'val_split':
                     {'end': config.val_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'type': config.dataset_type}},
            'batch_size': config.batch_size,
             'dataset_scaling': config.dataset_scaling,
             'debug': config.debug,
             'learning_rate': config.learning_rate,
             'num_train_epochs': config.num_train_epochs,
             'output_path': config.output_path,
             'perform_training': config.perform_training,
             'save_and_eval_per_epoch': config.save_and_eval_per_epoch,
             'task_type': 'training',
             'warmup_steps': config.warmup_steps
        }

    elif isinstance(config, OldInferenceConfig):
        config_dict = {
            'data':{
                'test_split':
                    {'end': config.test_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'type': config.dataset_type},
                'train_split':
                    {'end': config.train_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'type': config.dataset_type},
                'val_split':
                    {'end': config.val_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'type': config.dataset_type}},
            'beam_size': config.beam_size,
            'extract_logprobs': config.extract_logprobs,
            'output_path': config.output_path,
            'task_type': 'inference',
            'word_timestamps': config.word_timestamps,
            'dataset_scaling': config.dataset_scaling,
            'debug': config.debug,
            }
    else:
        raise RuntimeError

    return from_dict(TrainingConfig if config_dict["task_type"]=="training" else InferenceConfig, config_dict)

if __name__ == '__main__':
     # split_config = SplitConfig(**{"start": 0.0, "end": 1.0})
     # dataset_config = DatasetConfig(type="foo", path="poo")
     # print(split_config, dataset_config)
     #
     # split = Split(dataset=dataset_config, noise=True, split=split_config)
     #
     # print(split)
     # dict_for_nested_dataclasses = {"noise": True,
     #                "dataset":{"type": "foo", "path":"doo"},
     #                "split": {"start": 0.0, "end": 1.0}
     #                }
     # print(from_dict(Split, dict_for_nested_dataclasses))
     with open('read.yaml', 'r') as f:
         data = yaml.load(f, Loader=yaml.SafeLoader)
     config_dataclass = from_dict(BaseConfig, data)

     d: dict = config_to_dict(config_dataclass)

     with open('write.yaml', 'w') as file:
         yaml.dump(d, file, indent=4, sort_keys=False)



