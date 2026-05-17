print("Import libraries...")
import sys

try:
    import sip_whisper
except ModuleNotFoundError as e:
    print(e)
    print("sip_whisper module not found, please install it! (see readme.md)")
    sys.exit(1)

from shutil import copyfile

from pathlib import Path
import argparse
from dotenv import load_dotenv
import os

#logging
import logging
import sys

# custom imports
from utils.config_dataclasses import get_config, TrainingConfig, InferenceConfig, unfold_config
from utils.grid_utils import get_grid, apply_split
from train_whisper import train_whisper
from inference import inference
from utils.cuda_utils import manage_device

print("Imports done!")

def create_logger(config: InferenceConfig | TrainingConfig) -> logging.Logger:
    # create logger
    logging.basicConfig(
        level=logging.INFO, #dont set to DEBUG
        format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(config.output_path / "logfile.log", mode='w'),
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help="config file path")

    load_dotenv()

    if not (config_path:=parser.parse_args().f):
        config_path = "tmp_inference_config.ini"

    config = get_config(config_path)
    logger = create_logger(config)


    Path.mkdir(config.output_path.parent, exist_ok=True)
    Path.mkdir(config.output_path, exist_ok=config.debug)


    # check for fields that are lists, implying multiple tasks / group task

        updated_config = copy.deepcopy(config)
        name_tail = []
        for field, value in zip(fields_with_lists, v):
            setattr(updated_config, field, value)
            name_tail.append(f"{field}_{value}")

        updated_config.output_path = Path(updated_config.output_path) / "_".join(name_tail)
        configs.append(updated_config)

    configs = unfold_config(config)

    if len(configs) > 1:
        copyfile(Path.cwd()/config_path, config.output_path/"config.ini")

    manage_device(os.getenv("GPU_DEVICE"))

    for current_config in configs:
        logger.info(f"Execute new task, save to {config.output_path.relative_to(Path.cwd())}")
        if len(configs) > 1:
            Path.mkdir(current_config.output_path, exist_ok=config.debug)

        current_config.save_to_file(current_config.output_path/"config.ini")

        dataset = get_grid()
        dataset = apply_split(dataset, current_config)

        if isinstance(current_config, TrainingConfig):
            train_whisper(current_config, dataset)
        if isinstance(current_config, InferenceConfig):
            inference(current_config, dataset)


if __name__ == '__main__':
    main()
