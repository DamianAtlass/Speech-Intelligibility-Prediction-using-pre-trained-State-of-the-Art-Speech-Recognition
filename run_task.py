import sys
from utils.config_dataclasses import get_config
from pathlib import Path
from utils.grid_utils import get_grid


def main():
    if sys.version_info[0] < 3 and sys.version_info[1] < 12:
        raise Exception("Must be using Python 3.12 or later!")

if __name__ == '__main__':
    main()
    config = get_config("tmp_config.ini")

    cwd = Path.cwd()
    grid = get_grid()
    print()