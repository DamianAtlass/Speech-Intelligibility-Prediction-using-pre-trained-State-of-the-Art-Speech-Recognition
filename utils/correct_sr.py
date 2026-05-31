import json
from pathlib import Path
import os
from tqdm import tqdm

def convert_alignment_from_25kH_to_16kH(s: int) -> int:
    """
    Convert an alignment value from 25kH to 16kH using a simple formula

    s: original alignment value
    :return:
    """
    new = s/(25_000/16_000)

    return int(new)

def replace_alignment(t: dict) -> None:
    """
    Replaces the alignment section of a json dict. Happens in-place.
    """
    alignments = t["alignment"]

    for i in range(len(alignments)):
        start, end, word = alignments[i]
        start = convert_alignment_from_25kH_to_16kH(int(start))
        end = convert_alignment_from_25kH_to_16kH(int(end))

        alignments[i] = [str(t) for t in [start, end, word]]

def correct_file(path: Path) -> None:
    """
    Load, correct, and save a json file with alignments in the wrong 'format'.
    """

    if not path.exists():
        raise RuntimeError(f"File {path} not found")

    with open(path, 'r') as f:
        j = json.load(f)
        replace_alignment(j)
    os.remove(path)
    with open(path, 'w') as f:
        json.dump(j, f, indent=4)

if __name__ == '__main__':
    #path_single_file = Path("s3_bwbg6n.json")
    #replace_file(path_single_file)

    path = Path("inferences/_")

    #if not group
    if (path/"data").is_dir():
        raise RuntimeError() # is here to prevent accidental execution

        for json_path in tqdm((path/"data").iterdir()):
            correct_file(json_path)
    #if group
    else:
        raise RuntimeError() # is here to prevent accidental execution
        for file_or_dir in path.iterdir():
            if file_or_dir.is_dir():
                if (file_or_dir / "data").is_dir():
                    data_folder = file_or_dir / "data"
                    for json_path in tqdm(data_folder.iterdir()):
                        replace_file(file_or_dir)
