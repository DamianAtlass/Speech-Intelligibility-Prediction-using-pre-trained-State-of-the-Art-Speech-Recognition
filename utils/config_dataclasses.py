import os
from dataclasses import dataclass, fields
import configparser
from typing import Union
from pathlib import Path

@dataclass(kw_only=True)
class Config:
    model: str
    model_type: str
    model_path: Path = None

    output_path: Path

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
        self.output_path = Path.cwd()/self.output_path
        self.model_path = Path.cwd()/self.model_path if self.model_path else None
        self.debug = self.debug == "True"

    def save_to_file(self, path: Path) -> None:
        printing_template = [
            '[Config]',
            'model',
            'model_type',
            'model_path',
            'output_path',
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

        if path.exists():
            if self.debug:
                os.remove(path)
            else:
                raise FileExistsError(f"Config file already exists! \nPath: {path}")

        save_to_file(self, path, printing_template, len(fields(Config)) )

    def get_list_fields(self) -> list[str]:
        """
        Returns fields, which are given multiple values.
        """

        fields_with_lists = []
        for field in fields(self):
            if isinstance(getattr(self, field.name), list):
                fields_with_lists.append(field.name)
        return fields_with_lists

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

    def save_to_file(self, path: Path) -> None:
        super().save_to_file(path)

        printing_template = [
            '[TrainingConfig]',
            'perform_training',
            'learning_rate',
            'num_train_epochs',
            'warmup_steps',
        ]

        save_to_file(self, path, printing_template, len(fields(type(self))) - len(fields(Config) ))

@dataclass(kw_only=True)
class InferenceConfig(Config):
    extract_logits: bool = True
    word_timestamps: bool = True
    beam_size: int | list[int] = 5


    def __post_init__(self):
        super().__post_init__()
        self.extract_logits = self.extract_logits == "True"
        self.word_timestamps = self.word_timestamps == "True"

        if "," in self.beam_size:
            self.beam_size = list(map(int, self.beam_size.split(",")))
        else:
            self.beam_size = int(self.beam_size)

    def save_to_file(self, path: Path) -> None:
        super().save_to_file(path)

        printing_template = [
            '[InferenceConfig]',
            'extract_logits',
            'word_timestamps',
            'beam_size',
        ]

        save_to_file(self, path, printing_template, len(fields(type(self))) - len(fields(Config)) )

def get_config(path: Path) -> TrainingConfig | InferenceConfig:
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

def save_to_file(config: Config | TrainingConfig | InferenceConfig,
                 path: Path,
                 printing_template: list[str],
                 num_args_to_write: int) -> None:

    value_counter = 0
    with open(path, "a") as f:

        for entry in printing_template:
            if entry == "":
                line = "\n"
            elif any(word in entry for word in ["[", "#"]):
                line = entry + "\n"
            else:
                value = getattr(config, entry)
                if isinstance(value, list):
                    value = str(value)[1:-1]
                elif "path" in entry:
                    value = Path(value).relative_to(Path.cwd()) if value else ""


                line = f"{entry} = {value}" + "\n"
                value_counter = value_counter + 1

            f.write(line)

    assert num_args_to_write == value_counter, "Wrong number of arguments written to config file!"

if __name__ == '__main__':
    pass