import os
from dataclasses import dataclass, fields
import configparser
from pathlib import Path
import copy
from itertools import product
from typing import cast

@dataclass(kw_only=True)
class Config:
    model: str
    model_type: str | list
    model_path: Path | None = None

    output_path: Path

    dataset_type: str
    dataset_path: Path | None = None
    train_split: int | float
    test_split: int | float
    val_split: int | float = None

    #for debugging
    dataset_scaling: float = 1
    debug: bool = False

    def __post_init__(self):
        self.model_type = cast(str, self.model_type).replace(" ", "").split(",") if "," in self.model_type else self.model_type
        self.dataset_path = Path.cwd()/self.dataset_path if self.dataset_path else None
        self.train_split = to_int_or_float(self.train_split)
        self.test_split = to_int_or_float(self.test_split)
        self.val_split = to_int_or_float(self.val_split) if (self.val_split is not None) else None
        self.dataset_scaling = float(self.dataset_scaling)
        self.output_path = Path.cwd()/self.output_path
        self.model_path = Path.cwd()/self.model_path if self.model_path else None

        if isinstance(self.debug, str):
            self.debug = self.debug == "True"

        if not (all(isinstance(t, int) for t in [self.train_split, self.test_split, self.val_split]) or
                all(isinstance(t, float) for t in [self.train_split, self.test_split, self.val_split])):
            raise ValueError("self.train_split, self.test_split, self.val_split must all be either int of float!")

    def save_to_file(self, path: Path) -> None:
        printing_template = [
            '[Config]',
            'model',
            'model_type',
            'model_path',
            'output_path',
            '',
            'dataset_type',
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
        if isinstance(self.perform_training, str):
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
    extract_logprobs: bool = True
    word_timestamps: bool = True
    beam_size: int | list[int] = 5


    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.extract_logprobs, str):
            self.extract_logprobs = self.extract_logprobs == "True"
        if isinstance(self.word_timestamps, str):
            self.word_timestamps = self.word_timestamps == "True"

        if isinstance(self.beam_size, str):
            if "," in self.beam_size:
                self.beam_size = list(map(int, self.beam_size.replace(" ", "").split(",")))
            else:
                self.beam_size = int(self.beam_size)

    def save_to_file(self, path: Path) -> None:
        super().save_to_file(path)

        printing_template = [
            '[InferenceConfig]',
            'extract_logprobs',
            'word_timestamps',
            'beam_size',
        ]

        save_to_file(self, path, printing_template, len(fields(type(self))) - len(fields(Config)) )

def get_config(path: Path) -> TrainingConfig | InferenceConfig:
    config_parser = configparser.RawConfigParser()
    # configParser.optionxform = str  # preserve original case
    config_parser.read(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist! \nPath: {path}")

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

def unfold_config(config: TrainingConfig | InferenceConfig) -> list[Config]:
    """
    If a config file has multiple values for specific parameters, this function will unfold them into multiple individual configs.
    If it is a 'normal' config file, it should be returned as only object in the list.
    config: TrainingConfig | InferenceConfig

    Returns: list[Config], the list of individual configs
    """

    fields_with_lists = config.get_list_fields()
    configs: list[Config] = []

    # make individual, independent configs files of the group file with lists as values
    for v in product(*[getattr(config, field) for field in
                       fields_with_lists]):  # returns each possible combination of list fields

        updated_config = copy.deepcopy(config)
        name_tail = []
        for field, value in zip(fields_with_lists, v):
            setattr(updated_config, field, value)
            name_tail.append(f"{field}_{value}")

        updated_config.output_path = Path(updated_config.output_path) / "_".join(name_tail)
        configs.append(updated_config)

    return configs

def to_int_or_float(string: str | int | float):
    if not isinstance(string, str):
        return string
    try:
        num = int(string)
    except ValueError:
        num = float(string)
    return num


if __name__ == '__main__':
    pass