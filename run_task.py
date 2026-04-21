print("imports start")

import sys
from utils.config_dataclasses import get_config, TrainingConfig
from pathlib import Path
from utils.grid_utils import get_grid, apply_split
print("----")
from train_whisper import train_whisper
print("----")
from shutil import copyfile
print("imports done")

def main():
    if sys.version_info[0] < 3 and sys.version_info[1] < 12:
        raise Exception("Must be using Python 3.12 or later!")

    config_file_path = "tmp_config.ini"
    config = get_config(config_file_path)

    Path.mkdir(config.output_dir.parent, exist_ok=True)
    Path.mkdir(config.output_dir)
    copyfile(Path.cwd() / config_file_path, config.output_dir / config_file_path)

    dataset = get_grid()
    dataset = apply_split(dataset, config)

    if isinstance(config, TrainingConfig):
        train_whisper(config, dataset)

    print()

if __name__ == '__main__':
    main()
