from dataclasses import dataclass, fields
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass, asdict
import yaml
from typing import Any
from utils.config_dataclasses import Old_InferenceConfig, Old_TrainingConfig, old_get_config

from utils.paths import _PROJECT_ROOT


def from_dict(cls, data: dict) -> Any:
    if cls is BaseConfig and "task_type" in data.keys():
        cls = InferenceConfig if data["task_type"] == "inference" else TrainingConfig
    if not is_dataclass(cls):
        return data
    if cls is DataSplitConfig and data is None:
        return None # allow for missing splits

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
    dataset_type: str
    path: Path | None
    start: float|int
    end: float|int
    noise: bool
    scaling: float = 1

    def __post_init__(self):
        self.path = _PROJECT_ROOT/self.path if self.path is not None else None

@dataclass(kw_only=True)
class DatasetConfig:
    train_split: DataSplitConfig = None
    test_split: DataSplitConfig = None
    val_split: DataSplitConfig = None

@dataclass(kw_only=True)
class ModelConfig:
    name: str
    model_type: str
    path: Path | None

    def __post_init__(self):
        self.path = _PROJECT_ROOT/self.path if self.path is not None else None

@dataclass(kw_only=True)
class BaseConfig:
    output_path: Path
    task_type: str
    data: DatasetConfig
    model: ModelConfig

    #for debugging
    debug: bool = False

    def __post_init__(self):
        self.output_path = _PROJECT_ROOT/self.output_path

@dataclass(kw_only=True)
class InferenceConfig(BaseConfig):
    extract_logprobs: bool = True
    word_timestamps: bool = True
    beam_size: int = 5
    subword_timestamps: bool = False

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

def load_config(config_file_path: str | Path) -> TrainingConfig | InferenceConfig:
    config_file_path = Path.cwd() / config_file_path

    backup_ini_file = config_file_path.parent / f"{config_file_path.stem}.ini"

    if config_file_path.exists():
        with open(config_file_path, 'r') as f:
            data: dict = yaml.load(f, Loader=yaml.SafeLoader)
        return from_dict(BaseConfig, data)
    else:
        # no .yaml config path, check for .ini
        if backup_ini_file.exists():
            old_config: Old_InferenceConfig = old_get_config(backup_ini_file)

            new_config = convert_old_config_into_new(old_config)
            save_config(new_config, config_file_path)
            return new_config
        else:
            raise FileNotFoundError("No config file present!")




def save_config(config: TrainingConfig | InferenceConfig, path: Path) -> None:
    if path.exists(): raise FileExistsError()
    config_dict = config_to_dict(config)

    with open(str(path), 'w') as file:
        yaml.dump(config_dict, file, indent=4, sort_keys=False)

def convert_old_config_into_new(config: Old_TrainingConfig | Old_InferenceConfig) -> TrainingConfig|InferenceConfig:

    if isinstance(config, Old_TrainingConfig):
        config_dict = {
             'data': {
                 'test_split':
                     {'end': config.test_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'dataset_type': config.dataset_type, "scaling": config.dataset_scaling},
                 'train_split':
                     {'end': config.train_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'dataset_type': config.dataset_type, "scaling": config.dataset_scaling},
                 'val_split':
                     {'end': config.val_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'dataset_type': config.dataset_type, "scaling": config.dataset_scaling}},
            'batch_size': config.batch_size,
             'debug': config.debug,
             'learning_rate': config.learning_rate,
             'num_train_epochs': config.num_train_epochs,
             'output_path': config.output_path,
             'perform_training': config.perform_training,
             'save_and_eval_per_epoch': config.save_and_eval_per_epoch,
             'task_type': 'training',
             'warmup_steps': config.warmup_steps,
             'model': {'name': config.model, 'model_type': config.model_type, 'path': config.model_path}
        }

    elif isinstance(config, Old_InferenceConfig):
        config_dict = {
            'data':{
                'test_split':
                    {'end': config.test_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'dataset_type': config.dataset_type, "scaling": config.dataset_scaling},
                'train_split':
                    {'end': config.train_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'dataset_type': config.dataset_type, "scaling": config.dataset_scaling},
                'val_split':
                    {'end': config.val_split, 'noise': config.add_noise, 'path': config.dataset_path, 'start': 0, 'dataset_type': config.dataset_type, "scaling": config.dataset_scaling}},
            'beam_size': config.beam_size,
            'extract_logprobs': config.extract_logprobs,
            'output_path': config.output_path,
            'task_type': 'inference',
            'word_timestamps': config.word_timestamps,
            'debug': config.debug,
            'model': {'name': config.model, 'model_type': config.model_type, 'path': config.model_path}
        }
    else:
        raise RuntimeError

    return from_dict(TrainingConfig if config_dict["task_type"]=="training" else InferenceConfig, config_dict)

if __name__ == '__main__':
     # split_config = SplitConfig(**{"start": 0.0, "end": 1.0})
     # dataset_config = DatasetConfig(dataset_type="foo", path="poo")
     # print(split_config, dataset_config)
     #
     # split = Split(dataset=dataset_config, noise=True, split=split_config)
     #
     # print(split)
     # dict_for_nested_dataclasses = {"noise": True,
     #                "dataset":{"dataset_type": "foo", "path":"doo"},
     #                "split": {"start": 0.0, "end": 1.0}
     #                }
     # print(from_dict(Split, dict_for_nested_dataclasses))
     with open('read.yaml', 'r') as f:
         data = yaml.load(f, Loader=yaml.SafeLoader)
     config_dataclass = from_dict(BaseConfig, data)

     d: dict = config_to_dict(config_dataclass)

     with open('write.yaml', 'w') as file:
         yaml.dump(d, file, indent=4, sort_keys=False)



