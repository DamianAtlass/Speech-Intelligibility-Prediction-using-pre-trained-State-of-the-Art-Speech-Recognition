from dataclasses import dataclass, field
import configparser
from typing import Union
from pathlib import Path

@dataclass(frozen=True, kw_only=True)
class Config:
    dataset_path: Path
    train_split: float
    test_split: float
    val_split: float = None

    #for debugging
    dataset_scaling: int = 1

    def __post_init__(self):
        object.__setattr__(self, "dataset_path", Path.cwd()/self.dataset_path)
        object.__setattr__(self, "train_split", float(self.train_split))
        object.__setattr__(self, "test_split", float(self.test_split))
        object.__setattr__(self, "val_split", float(self.val_split) if self.val_split else None)
        object.__setattr__(self, "dataset_scaling", float(self.dataset_scaling))


@dataclass(frozen=True, kw_only=True)
class TrainingConfig(Config):
    model_type: str
    output_dir: Path
    perform_training: bool
    learning_rate: float
    num_train_epochs: int
    warmup_steps: int

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "output_dir", Path.cwd()/self.output_dir)
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "num_train_epochs", int(self.num_train_epochs))
        object.__setattr__(self, "warmup_steps", int(self.warmup_steps))



def get_config(path: str = None) -> Union[TrainingConfig]:
    config_parser = configparser.RawConfigParser()
    # configParser.optionxform = str  # preserve original case
    config_parser.read(path)

    tmp = {}
    for section in config_parser.sections():
        for k, v in config_parser[section].items():
            tmp[k] = v
    if "TrainingConfig" in config_parser.sections():
        return TrainingConfig(**tmp)
    else:
        raise NotImplementedError("No inference dataclass yet!")


if __name__ == '__main__':
    pass