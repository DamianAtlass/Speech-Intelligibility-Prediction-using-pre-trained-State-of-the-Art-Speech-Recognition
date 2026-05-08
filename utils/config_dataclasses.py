from dataclasses import dataclass, fields
import configparser
from typing import Union
from pathlib import Path

@dataclass(kw_only=True)
class Config:
    model: str
    model_type: str
    model_path: Path = None

    output_dir: Path

    dataset_path: Path
    train_split: float
    test_split: float
    val_split: float = None

    #for debugging
    dataset_scaling: float = 1
    debug: bool = False

    def __post_init__(self):
        self.dataset_path = Path.cwd()/self.dataset_path
        self.train_split = float(self.train_split)
        self.test_split = float(self.test_split)
        self.val_split = float(self.val_split) if self.val_split else None
        self.dataset_scaling = float(self.dataset_scaling)
        self.output_dir = Path.cwd()/self.output_dir
        self.model_path = Path.cwd()/self.model_path if self.model_path else None
        self.debug = self.debug == "True"

    def save_to_file(self, path: Path) -> None:
        printing_template = [
            '[Config]',
            'model',
            'model_type',
            'model_path',
            'output_dir',
            '',
            'dataset_path',
            'train_split',
            'test_split',
            'val_split',
            '',
            '#for debugging',
            'dataset_scaling',
            'debug',
            '',
        ]

        save_to_file(self, path, printing_template)

@dataclass(kw_only=True)
class TrainingConfig(Config):
    perform_training: bool
    learning_rate: float
    num_train_epochs: int
    warmup_steps: int

    def __post_init__(self):
        super().__post_init__()
        self.learning_rate = float(self.learning_rate)
        self.num_train_epochs = int(self.num_train_epochs)
        self.warmup_steps = int(self.warmup_steps)
        self.perform_training = self.perform_training=="True"

@dataclass(kw_only=True)
class InferenceConfig(Config):
    extract_logits: bool = True
    word_timestamps: bool = True
    beam_size: int | list[int] = 5
    value: int | list[int] = 5


    def __post_init__(self):
        super().__post_init__()
        self.extract_logits = self.extract_logits == "True"
        self.word_timestamps = self.word_timestamps == "True"

        if "," in self.beam_size:
            self.beam_size = list(map(int, self.beam_size.split(",")))
        else:
            self.beam_size = int(self.beam_size)

        if "," in self.value:
            self.value = list(map(int, self.value.split(",")))
        else:
            self.value = int(self.value)

    def save_to_file(self, path: Path) -> None:
        super().save_to_file(path)

        printing_template = [
            '[InferenceConfig]',
            'extract_logits',
            'word_timestamps',
        ]

        save_to_file(self, path, printing_template)

def get_config(path: str) -> TrainingConfig | InferenceConfig:
    config_parser = configparser.RawConfigParser()
    # configParser.optionxform = str  # preserve original case
    config_parser.read(path)

    tmp = {}
    for section in config_parser.sections():
        for k, v in config_parser[section].items():
            tmp[k] = v
    if "TrainingConfig" in config_parser.sections():
        return TrainingConfig(**tmp)
    elif "InferenceConfig" in config_parser.sections():
        return InferenceConfig(**tmp)

    raise ValueError("Not a valid config file!")

def save_to_file(config: Config | TrainingConfig | InferenceConfig, path: Path, printing_template: list[str]) -> None:

    #if not set(f.name for f in fields(type(config))).issubset(set(printing_template)):
    #    raise NotImplementedError("You probably forgot to add some values here!")

    with open(path, "a") as f:
        for line in printing_template:
            if line == "":
                f.write("\n")
            elif any(word in line for word in ["[", "#"]):
                f.write(line + "\n")
            else:
                f.write(f"{line} = {getattr(config, line)}" + "\n")

if __name__ == '__main__':
    pass