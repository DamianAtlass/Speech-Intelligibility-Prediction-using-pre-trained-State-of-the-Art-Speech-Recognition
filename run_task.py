print("Import libraries...")
import sys

try:
    import sip_whisper
except ModuleNotFoundError as e:
    print(e)
    print("sip_whisper module not found, please install it! (see readme.md)")
    sys.exit(1)

from shutil import copyfile
import copy
from itertools import product
from pathlib import Path

# custom imports
from utils.config_dataclasses import get_config, TrainingConfig, InferenceConfig, Config
from utils.grid_utils import get_grid, apply_split
from train_whisper import train_whisper
from inference import inference

#logging
import logging
import sys


print("Imports done!")

def create_logger(config: InferenceConfig | TrainingConfig) -> logging.Logger:
    # create logger
    logging.basicConfig(
        level=logging.INFO, #dont set to DEBUG
        format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(config.output_dir / "logfile.log", mode='w'),
            #logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # handle uncaught exceptions, see: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    return logger

def main():
    if sys.version_info[0] < 3 and sys.version_info[1] < 12:
        raise Exception("Must be using Python 3.12 or later!")

    config_path = Path(Path.cwd()/"tmp_training_config.ini")
    config = get_config(config_path)

    Path.mkdir(config.output_dir.parent, exist_ok=True)
    Path.mkdir(config.output_dir, exist_ok=config.debug)

    logger = create_logger(config)

    # check for fields that are lists, implying multiple tasks / group task
    fields_with_lists = config.get_list_fields()
    configs: list[Config] = []

    # make individual, independent configs files of the group file with lists as values
    for v in product(*[getattr(config, field) for field in fields_with_lists]): #returns each possible combination of list fields

        updated_config = copy.deepcopy(config)
        name_tail = []
        for field, value in zip(fields_with_lists, v):
            setattr(updated_config, field, value)
            name_tail.append(f"{field}_{value}")

        updated_config.output_dir = Path(updated_config.output_dir) / "_".join(name_tail)
        configs.append(updated_config)


    if len(configs) > 1:
        copyfile(Path.cwd()/config_path, config.output_dir/"config.ini")

    for current_config in configs:
        logger.info(f"Execute new task, save to {config.output_dir.relative_to(Path.cwd())}")
        Path.mkdir(current_config.output_dir, exist_ok=current_config.debug)
        current_config.save_to_file(current_config.output_dir/"config.ini")

        dataset = get_grid()
        dataset = apply_split(dataset, current_config)

        if isinstance(current_config, TrainingConfig):
            train_whisper(current_config, dataset)
        if isinstance(current_config, InferenceConfig):
            inference(current_config, dataset)


if __name__ == '__main__':
    main()
