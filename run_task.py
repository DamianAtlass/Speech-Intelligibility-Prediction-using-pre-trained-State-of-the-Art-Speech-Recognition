print("Import libraries...")
import sys

try:
    import sip_whisper
except ModuleNotFoundError as e:
    print(e)
    print("sip_whisper module not found, please install it! (see readme.md)")
    sys.exit(1)

from utils.config_dataclasses import get_config, TrainingConfig, InferenceConfig, Config
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

    config_path = Path(Path.cwd()/"tmp_training_config.ini")
    config = get_config(config_path)

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
        Path.mkdir(config.output_dir.parent, exist_ok=True)
        Path.mkdir(config.output_dir, exist_ok=config.debug)
        copyfile(Path.cwd()/config_path, config.output_dir/"config.ini")

    for current_config in configs:
        Path.mkdir(current_config.output_dir, exist_ok=current_config.debug)
        current_config.save_to_file(current_config.output_dir/"config.ini")

        dataset = get_grid()
        dataset = apply_split(dataset, current_config)

        if isinstance(current_config, TrainingConfig):
            train_whisper(current_config, dataset)
        if isinstance(current_config, InferenceConfig):
            inference(current_config, dataset)


    print()

if __name__ == '__main__':
    main()
