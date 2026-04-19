from dataclasses import dataclass, field
import configparser
from typing import Union

@dataclass(frozen=True, kw_only=True)
class Config:
    dataset_path: str
    train_split: float
    test_split: float
    val_split: float = None

@dataclass(frozen=True, kw_only=True)
class TrainingConfig(Config):
    epochs: int

    def __post_init__(self):
        object.__setattr__(self, "epochs", int(self.epochs))
        object.__setattr__(self, "train_split", float(self.train_split))
        object.__setattr__(self, "test_split", float(self.test_split))
        object.__setattr__(self, "val_split", float(self.val_split) if self.val_split else None)

def get_config(path: str = None) -> Union[TrainingConfig, ...]:
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