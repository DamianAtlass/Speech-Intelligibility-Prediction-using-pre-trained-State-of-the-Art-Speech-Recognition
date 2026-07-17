import shutil

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
from datasets import Dataset, DatasetDict

#logging
import logging
import sys

load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch

# custom imports
from utils.config_dataclasses import get_config, TrainingConfig, InferenceConfig, unfold_config
from train_whisper import train_whisper
from inference import inference
from utils.cuda_utils import select_device
from utils.logging_utils import catch_time
from utils.dataset_utils import get_dataset, apply_split
from evaluate_run import evaluate_run

INDIVIDUAL_WHISPER_MODELS = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo']

print("Imports done!")

def create_logger(config: InferenceConfig | TrainingConfig) -> logging.Logger:
    """
    Create a logger. Needs to happen in this file!
    config: InferenceConfig | TrainingConfig, contains output path
    """
    # create logger
    logging.basicConfig(
        level=logging.INFO, #dont set to DEBUG
        format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(config.output_path / "logfile.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # handle uncaught exceptions, see: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    def handle_exception(exc_type, exc_value, exc_traceback):
        print(exc_type, exc_value, exc_traceback)
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
    parser.add_argument('-f', default="training_config_template.ini", help="config file path")

    load_dotenv()

    config_path = Path(parser.parse_args().f)

    config = get_config(config_path)

    Path.mkdir(config.output_path.parent, exist_ok=True)
    if "delete_me" in config.output_path.name and config.debug:
        shutil.rmtree(config.output_path)
    Path.mkdir(config.output_path, exist_ok=config.debug)

    logger = create_logger(config)
    logger.info("Logger instantiated")

    # check for fields that are lists, implying multiple tasks / group task

    logger.info("Unfold configs")
    configs = unfold_config(config)
    logger.info(f"New configs: {len(configs)}")


    if len(configs) > 1:
        copyfile(Path.cwd()/config_path, config.output_path/"config.ini")

    logger.info("Set devices")
    device = select_device()

    with catch_time() as t:
        for i, current_config in enumerate(configs):
            logger.info(f"Task: {i+1}/{len(configs)}, save to {current_config.output_path.relative_to(Path.cwd())}")
            logger.info(f"Task config: {current_config}")
            if len(configs) > 1:
                Path.mkdir(current_config.output_path, exist_ok=current_config.debug)

            current_config.save_to_file(current_config.output_path/"config.ini")

            logger.info(f"Task: {i+1}/{len(configs)}, get dataset")
            dataset: Dataset= get_dataset(current_config.dataset_type)

            logger.info(f"Task: {i+1}/{len(configs)}, apply split")
            dataset: DatasetDict= apply_split(dataset,
                                              current_config.train_split,
                                              current_config.test_split,
                                              current_config.val_split,
                                              current_config.dataset_scaling)

            with catch_time() as t2:
                if isinstance(current_config, TrainingConfig):
                    logger.info(f"Task: {i + 1}/{len(configs)}, enter training")
                    train_whisper(current_config, dataset, device)
                if isinstance(current_config, InferenceConfig):
                    logger.info(f"Task: {i + 1}/{len(configs)}, enter inference")
                    inference(current_config, dataset, device)
            logger.info(f"Task: {i+1} - execution time: {i + 1}: {t2()/3600:.2f} h")

    logger.info(f"Execution time of all tasks: {t()/3600:.2f} h")
    logger.info(f"All tasks are finished!")
    if isinstance(config, InferenceConfig):
        logger.info("Evalute run.")
        evaluate_run(config.output_path, device)


if __name__ == '__main__':
    main()
