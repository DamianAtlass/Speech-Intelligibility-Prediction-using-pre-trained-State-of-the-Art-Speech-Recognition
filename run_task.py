print("Import libraries...")
import sys

try:
    import sip_whisper
except ModuleNotFoundError as e:
    print(e)
    print("sip_whisper module not found, please install it! (see readme.md)")
    sys.exit(1)

from utils.config_dataclasses import get_config, TrainingConfig, InferenceConfig
from dataclasses import fields
from pathlib import Path
from utils.grid_utils import get_grid, apply_split
from train_whisper import train_whisper
from shutil import copyfile
from inference import inference
from itertools import product
import copy

print("Imports done!")

def main():
    if sys.version_info[0] < 3 and sys.version_info[1] < 12:
        raise Exception("Must be using Python 3.12 or later!")

    config_file_path = "tmp_group_inference_config.ini"
    config = get_config(config_file_path)

    fields_with_lists = []
    for field in fields(config):
        if isinstance(getattr(config, field.name), list):
            fields_with_lists.append(field.name)

    configs = []
    counter = 0

    for v in product(*[getattr(config, field) for field in fields_with_lists]):
        current_config = copy.deepcopy(config)
        for field, value in zip(fields_with_lists, v):
            setattr(current_config, field, value)

        current_config.output_dir = Path(current_config.output_dir) / f"{counter}"
        configs.append(current_config)
        counter = counter + 1

    if len(configs) == 0:
        configs.append(config)
    else:
        Path.mkdir(config.output_dir.parent, exist_ok=True)
        Path.mkdir(config.output_dir, exist_ok=config.debug)
        copyfile(Path.cwd() / config_file_path, config.output_dir / config_file_path)

        config.save_to_file(Path.cwd() / "tmp.ini")

    for current_config in configs:
        Path.mkdir(current_config.output_dir.parent, exist_ok=True)
        Path.mkdir(current_config.output_dir, exist_ok=current_config.debug)
        copyfile(Path.cwd() / config_file_path, current_config.output_dir / config_file_path)

        dataset = get_grid()
        dataset = apply_split(dataset, current_config)

        if isinstance(current_config, TrainingConfig):
            train_whisper(current_config, dataset)
        if isinstance(current_config, InferenceConfig):
            inference(current_config, dataset)


    print()

if __name__ == '__main__':
    main()
